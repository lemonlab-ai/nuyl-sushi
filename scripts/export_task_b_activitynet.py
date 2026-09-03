import argparse
import json
from pathlib import Path
from typing import Any, Dict


SUBSET_MAP = {"train": "training", "val": "validation", "test": "testing"}
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_JSON = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "task_b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Task B ActivityNet-style JSON from master annotations")
    parser.add_argument(
        "--master-json",
        default=str(DEFAULT_MASTER_JSON),
        help=f"Path to master_annotations.json (default: {DEFAULT_MASTER_JSON})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output folder for Task B artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--level", choices=["coarse", "fine", "both"], default="both")
    return parser.parse_args()


def export_level(videos: Dict[str, Dict[str, Any]], level: str, output_path: Path) -> None:
    database: Dict[str, Dict[str, Any]] = {}
    labels = set()

    for video_id, record in videos.items():
        subset = SUBSET_MAP.get(record.get("subset", "train"), "training")
        duration = float(record.get("duration") or 0.0)
        anns = []
        for ann in record.get("annotations", []):
            label = ann[f"{level}_label"]
            labels.add(label)
            start = float(ann["start"])
            end = float(ann["end"])
            anns.append({"segment": [start, end], "label": label})
            duration = max(duration, end)

        database[video_id] = {
            "duration": duration,
            "subset": subset,
            "fps": record.get("fps", 0.0),
            "source_path": record.get("source_path", ""),
            "annotations": anns,
        }

    payload = {
        "version": "1.0",
        "task": f"temporal_localization_{level}",
        "labels": sorted(labels),
        "database": database,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    master = json.loads(Path(args.master_json).read_text(encoding="utf-8"))
    videos = master["videos"]
    output_dir = Path(args.output_dir)
    levels = ["coarse", "fine"] if args.level == "both" else [args.level]

    for level in levels:
        out = output_dir / f"activitynet_{level}.json"
        export_level(videos, level, out)
        print(f"[DONE] {level}: {out}")


if __name__ == "__main__":
    main()
