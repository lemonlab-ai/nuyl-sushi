import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_JSON = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT_CSV = ROOT_DIR / "artifacts" / "skill_level" / "skill_level_manifest.csv"

SKILL_LEVEL_ALIASES = {
    "beginner": "beginner",
    "novice": "beginner",
    "newbie": "beginner",
    "entry": "beginner",
    "entry_level": "beginner",
    "junior": "beginner",
    "intermediate": "intermediate",
    "advanced": "intermediate",
    "mid": "intermediate",
    "mid_level": "intermediate",
    "expert": "expert",
    "senior": "expert",
    "master": "expert",
    "pro": "expert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-video skill-level manifest from master_annotations.json."
    )
    parser.add_argument("--master-json", default=str(DEFAULT_MASTER_JSON))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--default-sequence-dir",
        default="",
        help="Optional directory to auto-fill sequence_path as <dir>/<video_id>.json",
    )
    parser.add_argument(
        "--drop-unknown-skill",
        action="store_true",
        help="Drop rows with skill_level=unknown.",
    )
    return parser.parse_args()


def normalize_skill_level(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    canon = text.replace("-", "_").replace(" ", "_")
    return SKILL_LEVEL_ALIASES.get(canon, canon)


def main() -> None:
    args = parse_args()
    master = json.loads(Path(args.master_json).read_text(encoding="utf-8"))
    videos: Dict[str, Dict[str, Any]] = master.get("videos", {})

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    seq_dir = Path(args.default_sequence_dir) if args.default_sequence_dir else None

    rows: List[Dict[str, Any]] = []
    for video_id in sorted(videos.keys()):
        rec = videos.get(video_id, {})
        skill_level = normalize_skill_level(rec.get("skill_level", "unknown"))
        if args.drop_unknown_skill and skill_level == "unknown":
            continue

        sequence_path = ""
        if seq_dir is not None:
            sequence_path = str((seq_dir / f"{video_id}.json").resolve())

        anns = rec.get("annotations", [])
        rows.append(
            {
                "video_id": str(video_id),
                "subject_id": str(rec.get("subject_id", "unknown")),
                "session_id": str(rec.get("session_id", "unknown")),
                "view_type": str(rec.get("view_type", "unknown")),
                "subset": str(rec.get("subset", "")),
                "skill_level": skill_level,
                "source_path": str(rec.get("source_path", "")),
                "duration": float(rec.get("duration", 0.0) or 0.0),
                "num_annotations": len(anns) if isinstance(anns, list) else 0,
                "sequence_path": sequence_path,
            }
        )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "video_id",
                    "subject_id",
                    "session_id",
                    "view_type",
                    "subset",
                    "skill_level",
                    "source_path",
                    "duration",
                    "num_annotations",
                    "sequence_path",
                ]
            )

    print(f"[DONE] skill-level manifest -> {output_csv}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
