import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = ROOT_DIR / "data" / "raw" / "via_csv"
DEFAULT_VIDEO_DIR = ROOT_DIR / "data" / "raw" / "videos"
DEFAULT_META_JSON = ROOT_DIR / "data" / "meta" / "video_meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock VIA CSV + dummy videos for pipeline smoke tests.")
    parser.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR))
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--video-meta-json", default=str(DEFAULT_META_JSON))
    parser.add_argument("--num-videos", type=int, default=6)
    parser.add_argument("--segments-per-video", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(csv_dir: Path, video_dir: Path, meta_path: Path) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)


def create_dummy_video_file(path: Path) -> None:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        width, height = 64, 64
        fps = 10.0
        n_frames = 120
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 5) % 255
            frame[:, :, 1] = (i * 3) % 255
            frame[:, :, 2] = (i * 7) % 255
            writer.write(frame)
        writer.release()
    except Exception:
        path.write_bytes(b"")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    csv_dir = Path(args.csv_dir)
    video_dir = Path(args.video_dir)
    meta_path = Path(args.video_meta_json)
    ensure_dirs(csv_dir, video_dir, meta_path)

    coarse_labels = ["PREPARE", "CUTTING", "ASSEMBLE"]
    fine_labels = ["WASH", "TRIM", "SLICE", "NIGIRI_SHAPE", "PLATE"]
    view_types = ["front_view", "side_view"]
    skill_levels = ["beginner", "intermediate", "expert"]

    metadata: Dict[str, Dict] = {}
    csv_rows_by_coarse: Dict[str, List[Dict[str, str]]] = {k: [] for k in coarse_labels}

    for i in range(args.num_videos):
        filename = f"mock_video_{i:03d}.mp4"
        video_id = Path(filename).stem
        create_dummy_video_file(video_dir / filename)

        subject_id = f"S{(i % 5) + 1:02d}"
        metadata[filename] = {
            "subject_id": subject_id,
            "session_id": f"mock-session-{(i % 2) + 1}",
            "view_type": view_types[i % len(view_types)],
            "skill_level": skill_levels[i % len(skill_levels)],
            "subset": "",
            "has_gyro": (i % 2 == 0),
            "gyro_path": "" if i % 2 else f"gyro/{video_id}.csv",
            "weight_path": f"weight/{video_id}.csv",
            "weight_g": 180 + i,
        }

        base = 0.0
        for _ in range(args.segments_per_video):
            coarse = coarse_labels[i % len(coarse_labels)]
            fine = fine_labels[rng.randrange(len(fine_labels))]
            start = base + rng.uniform(0.1, 0.5)
            end = start + rng.uniform(0.4, 1.4)
            base = end
            csv_rows_by_coarse[coarse].append(
                {
                    "file_list": json.dumps([filename], ensure_ascii=False),
                    "metadata": json.dumps({"1": fine}),
                    "temporal_coordinates": json.dumps([round(start, 3), round(end, 3)]),
                }
            )

    for coarse, rows in csv_rows_by_coarse.items():
        csv_path = csv_dir / f"{coarse.lower()}_mock.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_list", "metadata", "temporal_coordinates"])
            writer.writeheader()
            writer.writerows(rows)

    meta_payload = {"videos": metadata}
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Mock dataset generated.")
    print(f"CSV dir:   {csv_dir}")
    print(f"Video dir: {video_dir}")
    print(f"Meta:      {meta_path}")


if __name__ == "__main__":
    main()
