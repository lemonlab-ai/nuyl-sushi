import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from nuyl_sushi.config import ProjectPaths
from nuyl_sushi.data.metadata_candidates import (
    build_label_distribution,
    build_sensor_candidates,
    infer_view,
)


def build_parser(paths: Optional[ProjectPaths] = None) -> argparse.ArgumentParser:
    project = paths or ProjectPaths.discover()
    audit_dir = project.artifacts / "data_audit"
    parser = argparse.ArgumentParser(
        description=(
            "Build reviewable metadata and sensor-link candidates from the NUYL Sushi audit. "
            "The command never assigns subject identity, skill level, or a final sensor link."
        )
    )
    parser.add_argument("--video-manifest", default=str(audit_dir / "video_manifest.csv"))
    parser.add_argument("--sensor-manifest", default=str(audit_dir / "sensor_manifest.csv"))
    parser.add_argument(
        "--master-json", default=str(audit_dir / "master" / "master_annotations.json")
    )
    parser.add_argument("--output-dir", default=str(audit_dir / "metadata"))
    return parser


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: Dict[str, Any]) -> str:
    confidence = summary["first_rank_sensor_confidence"]
    return f"""# NUYL Sushi 中繼資料候選報告

產生時間：{summary['generated_at_utc']}

## 結果

- Canonical 已標註影片：{summary['referenced_videos']}
- 已有來源路徑證據的視角：{summary['direct_views']}
- 由檔名與原始檔 lineage 補出的視角：{summary['inferred_views']}
- 尚未解析的視角：{summary['unresolved_views']}
- 最終 view1 / view2：{summary['view1']} / {summary['view2']}
- 有時間戳、可列感測器候選的影片：{summary['videos_with_sensor_candidates']}
- 感測器候選列（每片最多 3 筆）：{summary['sensor_candidate_rows']}
- 第一候選為 high / medium / low / weak / segmented-parent：{confidence['high_candidate']} / {confidence['medium_candidate']} / {confidence['low_candidate']} / {confidence['weak_candidate']} / {confidence['manual_only_segmented_parent']}
- Coarse / fine 標籤種類：{summary['coarse_labels']} / {summary['fine_labels']}

## 使用限制

- `view_type` 是來源 lineage 推定，不等於人工驗證；每筆均保留 evidence 與 confidence。
- `sensor_link_candidates.csv` 只依同日時間接近程度排序，全部維持 `review_required`，不可直接當同步結果。
- 分段影片可能共用母檔時間戳，因此一律標示 `manual_only_segmented_parent`。
- `subject_id`、`skill_level`、`consent_scope` 未找到可靠依據，維持空白等待人工補錄。
- 感測器資料含 GPS 欄位；公開資料集前必須去識別化或排除。

## 輸出

- `metadata_candidates.csv`：99 支影片的 session、view 與待人工補錄欄位。
- `sensor_link_candidates.csv`：依時間排序的候選，不是最終關聯。
- `label_distribution.csv`：coarse/fine 的片段數、影片數與標註秒數。
- `summary.json`：機器可讀摘要。
"""


def run(args: argparse.Namespace) -> Dict[str, Any]:
    video_manifest = Path(args.video_manifest).resolve()
    sensor_manifest = Path(args.sensor_manifest).resolve()
    master_path = Path(args.master_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path in (video_manifest, sensor_manifest, master_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    videos = [
        row
        for row in read_csv(video_manifest)
        if row.get("is_referenced", "").lower() == "true"
    ]
    sensors = read_csv(sensor_manifest)
    master = json.loads(master_path.read_text(encoding="utf-8"))

    metadata_rows: List[Dict[str, Any]] = []
    direct_views = 0
    inferred_views = 0
    for row in sorted(videos, key=lambda item: item["video_id"]):
        view, evidence, confidence = infer_view(
            row["filename"], row.get("view_guess", ""), row.get("source_locations", "")
        )
        if row.get("view_guess", "").lower() in {"view1", "view2"}:
            direct_views += 1
        elif view != "unknown":
            inferred_views += 1
        metadata_rows.append(
            {
                "video_id": row["video_id"],
                "filename": row["filename"],
                "session_id": row.get("session_guess", "unknown"),
                "session_source": "source directory or filename date",
                "view_type": view,
                "view_source": evidence,
                "view_confidence": confidence,
                "subject_id": "",
                "skill_level": "",
                "sensor_capture_id": "",
                "consent_scope": "",
                "notes": "",
            }
        )

    sensor_candidates = build_sensor_candidates(videos, sensors)
    label_rows = build_label_distribution(master)
    first_rank = [row for row in sensor_candidates if row["candidate_rank"] == 1]
    confidence_names = (
        "high_candidate",
        "medium_candidate",
        "low_candidate",
        "weak_candidate",
        "manual_only_segmented_parent",
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "referenced_videos": len(metadata_rows),
        "direct_views": direct_views,
        "inferred_views": inferred_views,
        "unresolved_views": sum(row["view_type"] == "unknown" for row in metadata_rows),
        "view1": sum(row["view_type"] == "view1" for row in metadata_rows),
        "view2": sum(row["view_type"] == "view2" for row in metadata_rows),
        "videos_with_sensor_candidates": len({row["video_id"] for row in sensor_candidates}),
        "sensor_candidate_rows": len(sensor_candidates),
        "first_rank_sensor_confidence": {
            name: sum(row["confidence"] == name for row in first_rank)
            for name in confidence_names
        },
        "coarse_labels": sum(row["level"] == "coarse" for row in label_rows),
        "fine_labels": sum(row["level"] == "fine" for row in label_rows),
        "split_readiness": "blocked_missing_subject_id",
        "source_files": {
            "video_manifest": str(video_manifest),
            "sensor_manifest": str(sensor_manifest),
            "master_json": str(master_path),
        },
    }

    write_csv(
        output_dir / "metadata_candidates.csv",
        metadata_rows,
        [
            "video_id", "filename", "session_id", "session_source", "view_type",
            "view_source", "view_confidence", "subject_id", "skill_level",
            "sensor_capture_id", "consent_scope", "notes",
        ],
    )
    write_csv(
        output_dir / "sensor_link_candidates.csv",
        sensor_candidates,
        [
            "video_id", "filename", "session_id", "candidate_rank", "sensor_capture_id",
            "sensor_relative_path", "absolute_clock_delta_seconds", "confidence",
            "decision", "warning",
        ],
    )
    write_csv(
        output_dir / "label_distribution.csv",
        label_rows,
        ["level", "label", "segment_count", "video_count", "total_annotated_seconds"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "REPORT_ZH.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
