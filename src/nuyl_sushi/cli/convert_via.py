import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from nuyl_sushi.config import ProjectPaths
from nuyl_sushi.data.conversion import convert_via_dataset, load_optional_meta
from nuyl_sushi.data.master import save_master


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Convert VIA CSV annotations to canonical master_annotations.json"
    )
    parser.add_argument("--csv-dir", default=str(paths.data / "raw" / "via_csv"))
    parser.add_argument("--video-dir", default=str(paths.data / "raw" / "videos"))
    parser.add_argument("--output-dir", default=str(paths.artifacts / "master"))
    parser.add_argument(
        "--video-meta-json",
        default=str(paths.data / "meta" / "video_meta.json"),
        help="Optional JSON mapping filename/video_id to metadata",
    )
    parser.add_argument("--split-by", choices=["subject_id", "video_id"], default="subject_id")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    csv_dir = Path(args.csv_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    metadata_path = Path(args.video_meta_json) if args.video_meta_json else None
    dataset, stats = convert_via_dataset(
        csv_dir,
        video_dir,
        video_meta=load_optional_meta(metadata_path),
        split_by=args.split_by,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_master(dataset, output_dir / "master_annotations.json")
    for level in ("coarse", "fine"):
        labels = dataset.label_space[level]
        (output_dir / f"label_map_{level}.txt").write_text(
            "\n".join(f"{index}\t{label}" for index, label in enumerate(labels)) + "\n",
            encoding="utf-8",
        )
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[DONE] master_annotations.json saved to: {output_dir}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
