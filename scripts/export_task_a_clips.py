import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_JSON = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "task_a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Task A manifests and optional trimmed clips from master_annotations.json"
    )
    parser.add_argument(
        "--master-json",
        default=str(DEFAULT_MASTER_JSON),
        help=f"Path to master_annotations.json (default: {DEFAULT_MASTER_JSON})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output folder for Task A artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--level", choices=["coarse", "fine", "both"], default="both")
    parser.add_argument("--trim-clips", action="store_true", help="Trim each annotation segment into a clip file")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary path/name")
    parser.add_argument("--video-root", default="", help="Fallback root if source_path is relative")
    parser.add_argument("--reencode", action="store_true", help="Re-encode clips instead of stream-copy")
    return parser.parse_args()


def safe_name(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def trim_clip(
    ffmpeg_bin: str,
    src: Path,
    dst: Path,
    start_sec: float,
    end_sec: float,
    reencode: bool,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end_sec - start_sec)
    if reencode:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-an",
            str(dst),
        ]
    else:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-c",
            "copy",
            str(dst),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src} -> {dst}\n{proc.stderr}")


def resolve_ffmpeg_bin(ffmpeg_arg: str) -> str:
    direct = shutil.which(ffmpeg_arg)
    if direct:
        return direct
    candidates = []
    for base in (Path(sys.prefix), Path(os.environ.get("CONDA_PREFIX", ""))):
        if not str(base):
            continue
        candidates.extend(
            [
                base / "Library" / "bin" / "ffmpeg.exe",
                base / "Scripts" / "ffmpeg.exe",
                base / "bin" / "ffmpeg",
            ]
        )
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"ffmpeg not found: {ffmpeg_arg}. Try passing --ffmpeg <absolute-path-to-ffmpeg.exe>."
    )


def main() -> None:
    args = parse_args()
    master = json.loads(Path(args.master_json).read_text(encoding="utf-8"))
    videos: Dict[str, Dict[str, Any]] = master["videos"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    levels = ["coarse", "fine"] if args.level == "both" else [args.level]

    ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg) if args.trim_clips else args.ffmpeg

    for level in levels:
        labels = sorted(
            {
                ann[f"{level}_label"]
                for record in videos.values()
                for ann in record.get("annotations", [])
                if f"{level}_label" in ann
            }
        )
        label_to_id = {name: idx for idx, name in enumerate(labels)}
        (output_dir / f"class_map_{level}.json").write_text(
            json.dumps(label_to_id, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        rows: List[Dict[str, Any]] = []
        list_by_subset = defaultdict(list)

        for video_id, record in videos.items():
            src = Path(record.get("source_path", ""))
            if not src.is_absolute() and args.video_root:
                src = Path(args.video_root) / src
            subset = record.get("subset", "train")
            for ann_idx, ann in enumerate(record.get("annotations", [])):
                label = ann[f"{level}_label"]
                label_id = label_to_id[label]
                start_sec = float(ann["start"])
                end_sec = float(ann["end"])
                clip_id = f"{video_id}_{ann_idx:04d}"

                clip_path = ""
                if args.trim_clips:
                    clip_rel = Path("clips") / level / subset / safe_name(label) / f"{clip_id}.mp4"
                    clip_abs = output_dir / clip_rel
                    trim_clip(ffmpeg_bin, src, clip_abs, start_sec, end_sec, args.reencode)
                    clip_path = str(clip_abs.resolve())
                    list_by_subset[subset].append(f"{clip_rel.as_posix()} {label_id}")

                rows.append(
                    {
                        "clip_id": clip_id,
                        "video_id": video_id,
                        "video_path": str(src),
                        "clip_path": clip_path,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "label": label,
                        "label_id": label_id,
                        "subset": subset,
                        "view_type": record.get("view_type", "unknown"),
                        "subject_id": record.get("subject_id", "unknown"),
                        "session_id": record.get("session_id", "unknown"),
                        "skill_level": record.get("skill_level", "unknown"),
                        "has_gyro": bool(record.get("has_gyro", False)),
                        "weight_g": record.get("weight_g"),
                    }
                )

        csv_path = output_dir / f"task_a_{level}_manifest.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

        if args.trim_clips:
            for subset, lines in list_by_subset.items():
                list_path = output_dir / f"mmaction_{level}_{subset}.txt"
                list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(f"[DONE] {level}: manifest -> {csv_path}")
        if args.trim_clips:
            print(f"       clips -> {output_dir / 'clips' / level}")


if __name__ == "__main__":
    main()
