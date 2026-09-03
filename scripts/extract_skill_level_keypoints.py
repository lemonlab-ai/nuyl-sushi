import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_JSON = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "skill_level" / "keypoints"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.gesture_coach.coach import extract_keypoints_mediapipe, save_sequence_json

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
        description="Batch extract MediaPipe hand keypoint sequences for skill-level experiments."
    )
    parser.add_argument("--master-json", default=str(DEFAULT_MASTER_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subset", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--max-videos", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--drop-unknown-skill", action="store_true")
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
    if not videos:
        raise RuntimeError("No videos found in master json.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    index_csv = out_dir / "skill_level_keypoint_index.csv"

    rows: List[Dict[str, Any]] = []
    processed = 0

    for video_id in sorted(videos.keys()):
        rec = videos.get(video_id, {})
        subset = str(rec.get("subset", "")).strip().lower()
        if args.subset != "all" and subset != args.subset:
            continue

        skill_level = normalize_skill_level(rec.get("skill_level", "unknown"))
        if args.drop_unknown_skill and skill_level == "unknown":
            continue

        src = Path(str(rec.get("source_path", "")))
        if not src.is_absolute():
            src = (ROOT_DIR / src).resolve()
        seq_path = seq_dir / f"{video_id}.json"

        status = "ok"
        error = ""
        if args.skip_existing and seq_path.exists():
            status = "skipped_existing"
        elif not src.exists():
            status = "missing_video"
            error = f"source_path not found: {src}"
        else:
            try:
                seq = extract_keypoints_mediapipe(
                    video_path=src,
                    max_frames=int(args.max_frames),
                    min_detection_confidence=float(args.min_detection_confidence),
                    min_tracking_confidence=float(args.min_tracking_confidence),
                )
                save_sequence_json(seq_path, seq)
            except Exception as exc:
                status = "extract_failed"
                error = str(exc)

        rows.append(
            {
                "sample_id": str(video_id),
                "video_id": str(video_id),
                "subject_id": str(rec.get("subject_id", "unknown")),
                "skill_level": skill_level,
                "subset": subset,
                "source_path": str(src),
                "sequence_path": str(seq_path.resolve()),
                "status": status,
                "error": error,
            }
        )
        processed += 1
        if args.max_videos > 0 and processed >= int(args.max_videos):
            break

    with index_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "video_id",
            "subject_id",
            "skill_level",
            "subset",
            "source_path",
            "sequence_path",
            "status",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_count = sum(1 for r in rows if r.get("status") in {"ok", "skipped_existing"})
    print(f"[DONE] keypoint extraction index -> {index_csv}")
    print(f"rows={len(rows)}, usable={ok_count}")


if __name__ == "__main__":
    main()
