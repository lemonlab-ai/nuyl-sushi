import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "data_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a read-only NUYL Sushi source dataset against canonical master annotations."
    )
    parser.add_argument("--source-root", required=True, help="Folder containing NUYLSushi-1M and raw sessions.")
    parser.add_argument("--master-json", required=True, help="Canonical master_annotations.json to audit.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-hash", action="store_true", help="Skip SHA-256 generation.")
    parser.add_argument("--ffprobe", default="", help="Optional explicit ffprobe executable path.")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_session(parts: Iterable[str]) -> str:
    patterns = (
        (re.compile(r"^20221210$"), "2022-12-10"),
        (re.compile(r"^20221215$"), "2022-12-15"),
        (re.compile(r"^20230204$"), "2023-02-04"),
        (re.compile(r"^20230208$"), "2023-02-08"),
        (re.compile(r"^221210$"), "2022-12-10"),
        (re.compile(r"^221215$"), "2022-12-15"),
        (re.compile(r"^230204$"), "2023-02-04"),
        (re.compile(r"^230208$"), "2023-02-08"),
    )
    for part in parts:
        for pattern, value in patterns:
            if pattern.match(part):
                return value
    return "unknown"


def infer_source_metadata(source_paths: List[Path]) -> Dict[str, str]:
    views = set()
    sessions = set()
    for path in source_paths:
        lower_parts = [part.lower() for part in path.parts]
        if "view1" in lower_parts:
            views.add("view1")
        if "view2" in lower_parts:
            views.add("view2")
        session = normalize_session(path.parts)
        if session != "unknown":
            sessions.add(session)
    return {
        "view_guess": next(iter(views)) if len(views) == 1 else "unknown",
        "session_guess": next(iter(sessions)) if len(sessions) == 1 else "unknown",
    }


def find_ffprobe(explicit: str) -> Optional[str]:
    if explicit:
        candidate = Path(explicit)
        return str(candidate) if candidate.is_file() else None
    return shutil.which("ffprobe")


def probe_video_cv2(path: Path) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError:
        return {"status": "not_checked", "backend": "", "error": "ffprobe and OpenCV unavailable"}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"status": "failed", "backend": "opencv", "error": "VideoCapture open failed"}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    first_ok, _ = capture.read()
    last_ok = False
    if frames > 1:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
        last_ok, _ = capture.read()
    capture.release()
    valid = fps > 0 and frames > 0 and width > 0 and height > 0 and first_ok and last_ok
    return {
        "status": "ok" if valid else "failed",
        "backend": "opencv",
        "error": "" if valid else "invalid metadata or unreadable boundary frame",
        "duration": (frames / fps) if fps > 0 else "",
        "codec": "",
        "width": width,
        "height": height,
        "frame_rate": fps,
        "num_frames": frames,
    }


def probe_video(path: Path, ffprobe: Optional[str]) -> Dict[str, Any]:
    if not ffprobe:
        return probe_video_cv2(path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "backend": "ffprobe", "error": str(exc)}
    if result.returncode != 0:
        return {"status": "failed", "backend": "ffprobe", "error": result.stderr.strip()[:500]}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "failed", "backend": "ffprobe", "error": f"invalid ffprobe JSON: {exc}"}
    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), {}
    )
    return {
        "status": "ok" if video_stream else "failed",
        "backend": "ffprobe",
        "error": "" if video_stream else "no video stream",
        "duration": payload.get("format", {}).get("duration", ""),
        "codec": video_stream.get("codec_name", ""),
        "width": video_stream.get("width", ""),
        "height": video_stream.get("height", ""),
        "frame_rate": video_stream.get("r_frame_rate", ""),
        "num_frames": video_stream.get("nb_frames", ""),
    }


def read_csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            return next(csv.reader(handle), [])
    except OSError:
        return []


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_sensors(source_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sensor_root in sorted(path for path in source_root.rglob("sensor") if path.is_dir()):
        for capture_dir in sorted(path for path in sensor_root.iterdir() if path.is_dir()):
            csv_files = sorted(capture_dir.glob("*.csv"))
            names = {path.name for path in csv_files}
            rows.append(
                {
                    "session": normalize_session(capture_dir.parts),
                    "capture_id": capture_dir.name,
                    "relative_path": capture_dir.relative_to(source_root).as_posix(),
                    "csv_count": len(csv_files),
                    "has_motion": "Motion.csv" in names,
                    "has_gravity_attitude": "GravityAndAttitude.csv" in names,
                    "has_gps": "GPS.csv" in names,
                    "contains_location_pii": "GPS.csv" in names,
                    "motion_columns": "|".join(read_csv_header(capture_dir / "Motion.csv")),
                    "gravity_columns": "|".join(
                        read_csv_header(capture_dir / "GravityAndAttitude.csv")
                    ),
                }
            )
    return rows


def markdown_report(summary: Dict[str, Any]) -> str:
    media = summary["media_probe"]
    return f"""# NUYL Sushi 資料稽核報告

產生時間：{summary['generated_at_utc']}

來源資料（唯讀）：`{summary['source_root']}`

## 結論

- Canonical VIA records：{summary['annotation_records']}
- 動作片段：{summary['annotation_segments']}
- Dataset MP4：{summary['dataset_videos']}
- 標註引用且存在：{summary['referenced_present']}
- 標註引用但缺檔：{summary['referenced_missing']}
- 未被標註引用的影片：{summary['unreferenced_videos']}
- 零位元影片：{summary['zero_length_videos']}
- SHA-256 已完成：{summary['hashed_videos']}
- 重複內容 hash groups：{summary['duplicate_hash_groups']}
- 可直接由來源路徑判斷 view1：{summary['view_counts'].get('view1', 0)}
- 可直接由來源路徑判斷 view2：{summary['view_counts'].get('view2', 0)}
- 視角仍未知：{summary['view_counts'].get('unknown', 0)}
- Sensor captures：{summary['sensor_captures']}
- 含 GPS 的 captures：{summary['gps_captures']}

## 尚未閉合

- subject metadata 缺失：{summary['unknown_subject_records']} / {summary['annotation_records']}
- skill-level metadata 缺失：{summary['unknown_skill_records']} / {summary['annotation_records']}
- 媒體探測：{media['status']}（ok={media['ok']}、failed={media['failed']}、not_checked={media['not_checked']}）
- `20221216.xlsx`：{summary['spreadsheet_status']}
- Sensor capture 尚未對應到 canonical video_id。
- Subject-wise split readiness：{summary['split_readiness']}
- GPS 是精確位置資料，進入任何公開 dataset 前必須移除或去識別化。

## 輸出

- `summary.json`：機器可讀摘要。
- `video_manifest.csv`：全部 dataset videos、引用狀態、來源推定與 checksum。
- `missing_references.csv`：標註引用但缺少的影片。
- `unreferenced_videos.csv`：存在但未被 canonical annotations 引用的影片。
- `sensor_manifest.csv`：sensor captures 與欄位/PII 狀態。
- `metadata_worklist.csv`：99 支 canonical videos 的人工補值表。

此報告只描述檔案與資料契約完整性，不代表已取得公開或散布授權。
"""


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    master_path = Path(args.master_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_root = source_root / "NUYLSushi-1M"
    video_dir = dataset_root / "videos"

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Dataset video folder not found: {video_dir}")
    if not master_path.is_file():
        raise FileNotFoundError(f"Master JSON not found: {master_path}")

    master = json.loads(master_path.read_text(encoding="utf-8"))
    records: Dict[str, Dict[str, Any]] = master.get("videos", {})
    referenced_names = {str(record.get("filename", "")) for record in records.values()}
    annotation_by_name = {
        str(record.get("filename", "")): record for record in records.values()
    }

    source_index: Dict[str, List[Path]] = {}
    for candidate in source_root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() != ".mp4":
            continue
        if video_dir in candidate.parents:
            continue
        source_index.setdefault(candidate.name, []).append(candidate)

    ffprobe = find_ffprobe(args.ffprobe)
    video_rows: List[Dict[str, Any]] = []
    videos = sorted(video_dir.glob("*.mp4"), key=lambda path: path.name.lower())
    for index, video_path in enumerate(videos, start=1):
        record = annotation_by_name.get(video_path.name, {})
        source_paths = source_index.get(video_path.name, [])
        inferred = infer_source_metadata(source_paths)
        digest = "" if args.skip_hash else sha256_file(video_path)
        probe = probe_video(video_path, ffprobe)
        video_rows.append(
            {
                "video_id": record.get("video_id", video_path.stem),
                "filename": video_path.name,
                "size_bytes": video_path.stat().st_size,
                "sha256": digest,
                "is_referenced": video_path.name in referenced_names,
                "annotation_count": len(record.get("annotations", [])),
                "coarse_labels": "|".join(
                    sorted({str(item.get("coarse_label", "")) for item in record.get("annotations", [])})
                ),
                "view_guess": inferred["view_guess"],
                "session_guess": inferred["session_guess"],
                "source_match_count": len(source_paths),
                "source_locations": "|".join(
                    path.relative_to(source_root).as_posix() for path in source_paths
                ),
                "subject_id": record.get("subject_id", "unknown"),
                "skill_level": record.get("skill_level", "unknown"),
                "probe_status": probe.get("status", "not_checked"),
                "probe_error": probe.get("error", ""),
                "probe_backend": probe.get("backend", ""),
                "codec": probe.get("codec", ""),
                "width": probe.get("width", ""),
                "height": probe.get("height", ""),
                "duration_probe": probe.get("duration", ""),
            }
        )
        if index % 10 == 0 or index == len(videos):
            print(f"[AUDIT] videos {index}/{len(videos)}")

    available_names = {row["filename"] for row in video_rows}
    missing_rows = [
        {
            "video_id": record.get("video_id", video_id),
            "filename": record.get("filename", ""),
            "source_path": record.get("source_path", ""),
        }
        for video_id, record in records.items()
        if str(record.get("filename", "")) not in available_names
    ]
    unreferenced_rows = [row for row in video_rows if not row["is_referenced"]]
    sensor_rows = audit_sensors(source_root)

    view_counts = Counter(row["view_guess"] for row in video_rows if row["is_referenced"])
    probe_counts = Counter(row["probe_status"] for row in video_rows)
    hash_counts = Counter(row["sha256"] for row in video_rows if row["sha256"])
    xlsx_files = sorted(path.relative_to(source_root).as_posix() for path in source_root.glob("*.xlsx"))
    summary = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "master_json": str(master_path),
        "annotation_records": len(records),
        "annotation_segments": sum(len(record.get("annotations", [])) for record in records.values()),
        "coarse_classes": len(master.get("label_space", {}).get("coarse", [])),
        "fine_classes": len(master.get("label_space", {}).get("fine", [])),
        "dataset_videos": len(video_rows),
        "dataset_bytes": sum(int(row["size_bytes"]) for row in video_rows),
        "referenced_present": len(records) - len(missing_rows),
        "referenced_missing": len(missing_rows),
        "unreferenced_videos": len(unreferenced_rows),
        "zero_length_videos": sum(int(row["size_bytes"]) == 0 for row in video_rows),
        "hashed_videos": sum(bool(row["sha256"]) for row in video_rows),
        "duplicate_hash_groups": sum(count > 1 for count in hash_counts.values()),
        "view_counts": dict(sorted(view_counts.items())),
        "unknown_subject_records": sum(
            str(record.get("subject_id", "unknown")).lower() in {"", "unknown", "none", "null"}
            for record in records.values()
        ),
        "unknown_skill_records": sum(
            str(record.get("skill_level", "unknown")).lower() in {"", "unknown", "none", "null"}
            for record in records.values()
        ),
        "sensor_captures": len(sensor_rows),
        "gps_captures": sum(bool(row["has_gps"]) for row in sensor_rows),
        "sensor_video_links": 0,
        "split_readiness": "blocked: subject_id is unknown for every canonical record",
        "media_probe": {
            "status": (
                "complete"
                if probe_counts.get("not_checked", 0) == 0
                else "blocked_ffprobe_and_opencv_unavailable"
            ),
            "executable": ffprobe or "",
            "ok": probe_counts.get("ok", 0),
            "failed": probe_counts.get("failed", 0),
            "not_checked": probe_counts.get("not_checked", 0),
        },
        "spreadsheets_found": xlsx_files,
        "spreadsheet_status": "not inspected: required spreadsheet runtime unavailable",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    video_fields = list(video_rows[0].keys()) if video_rows else ["filename"]
    write_csv(output_dir / "video_manifest.csv", video_rows, video_fields)
    write_csv(
        output_dir / "missing_references.csv",
        missing_rows,
        ["video_id", "filename", "source_path"],
    )
    write_csv(output_dir / "unreferenced_videos.csv", unreferenced_rows, video_fields)
    sensor_fields = list(sensor_rows[0].keys()) if sensor_rows else ["capture_id"]
    write_csv(output_dir / "sensor_manifest.csv", sensor_rows, sensor_fields)
    metadata_rows = [
        {
            "video_id": row["video_id"],
            "filename": row["filename"],
            "subject_id": "" if str(row["subject_id"]).lower() == "unknown" else row["subject_id"],
            "session_id": "" if row["session_guess"] == "unknown" else row["session_guess"],
            "view_type": "" if row["view_guess"] == "unknown" else row["view_guess"],
            "skill_level": "" if str(row["skill_level"]).lower() == "unknown" else row["skill_level"],
            "sensor_capture_id": "",
            "consent_scope": "",
            "notes": "",
        }
        for row in video_rows
        if row["is_referenced"]
    ]
    write_csv(
        output_dir / "metadata_worklist.csv",
        metadata_rows,
        [
            "video_id",
            "filename",
            "subject_id",
            "session_id",
            "view_type",
            "skill_level",
            "sensor_capture_id",
            "consent_scope",
            "notes",
        ],
    )
    (output_dir / "REPORT_ZH.md").write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
