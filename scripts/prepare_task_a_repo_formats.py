import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "artifacts" / "task_a" / "task_a_fine_manifest.csv"
DEFAULT_CLASS_MAP = ROOT_DIR / "artifacts" / "task_a" / "class_map_fine.json"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "artifacts" / "task_a" / "repo_formats"
DEFAULT_TASK_A_ROOT = ROOT_DIR / "artifacts" / "task_a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Task A exports for VideoMAEv2 / SlowFast (+ optional TSM frame lists)."
    )
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--task-a-root",
        default=str(DEFAULT_TASK_A_ROOT),
        help="Base folder for relative paths in CSV (used by VideoMAEv2 --data_root and SlowFast PATH_PREFIX).",
    )
    parser.add_argument(
        "--path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Write clip path as relative-to-task-a-root or absolute path.",
    )
    parser.add_argument(
        "--allow-video-fallback",
        action="store_true",
        help="Allow using full video_path when clip_path is empty (not recommended for clip classification).",
    )
    parser.add_argument(
        "--prepare-tsm",
        action="store_true",
        help="Also extract frames and write TSM list files.",
    )
    parser.add_argument(
        "--tsm-frames-dir",
        default="",
        help="TSM frame root (default: <output-root>/tsm/frames).",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Re-extract frames if target frame folder already exists.",
    )
    return parser.parse_args()


def ensure_subset(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"train", "val", "test"}:
        return v
    return "train"


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def load_class_map(path: Path) -> Dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"class_map must be a dict: {path}")
    return {str(k): int(v) for k, v in data.items()}


def resolve_media_path(row: Dict[str, str], allow_video_fallback: bool) -> Path:
    clip_path = (row.get("clip_path") or "").strip()
    if clip_path:
        return Path(clip_path)
    if allow_video_fallback:
        video_path = (row.get("video_path") or "").strip()
        if video_path:
            return Path(video_path)
    raise RuntimeError(
        f"clip_path missing for clip_id={row.get('clip_id')} (run export_task_a_clips.py --trim-clips)"
    )


def to_output_path(media_path: Path, task_a_root: Path, path_mode: str) -> str:
    abs_path = media_path.resolve()
    if path_mode == "absolute":
        return abs_path.as_posix()
    try:
        rel = abs_path.relative_to(task_a_root.resolve())
        return rel.as_posix()
    except ValueError:
        return abs_path.as_posix()


def write_kinetics_style_csv(out_file: Path, rows: Iterable[Tuple[str, int]]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{p} {label}" for p, label in rows]
    out_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def probe_num_frames(video_path: Path) -> int:
    try:
        import cv2  # type: ignore
    except Exception:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(0, frames)


def safe_name(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value)


def extract_frames(ffmpeg_bin: str, src_video: Path, dst_dir: Path, overwrite: bool) -> int:
    if dst_dir.exists():
        existing = sorted(dst_dir.glob("img_*.jpg"))
        if existing and not overwrite:
            return len(existing)
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src_video),
        str(dst_dir / "img_%05d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src_video}\n{proc.stderr}")
    return len(list(dst_dir.glob("img_*.jpg")))


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


def prepare_tsm(
    rows: List[Dict[str, str]],
    tsm_root: Path,
    frames_root: Path,
    ffmpeg_bin: str,
    overwrite_frames: bool,
) -> None:
    tsm_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    labels = {}
    split_lines = defaultdict(list)
    skipped = 0

    for row in rows:
        clip_path = (row.get("clip_path") or "").strip()
        if not clip_path:
            skipped += 1
            continue
        src = Path(clip_path)
        if not src.exists():
            skipped += 1
            continue
        label = str(row["label"])
        label_id = int(row["label_id"])
        labels[label] = label_id
        subset = ensure_subset(row.get("subset", "train"))
        clip_id = str(row["clip_id"])
        rel_dir = Path(subset) / safe_name(label) / clip_id
        frame_dir = frames_root / rel_dir
        n_frames = extract_frames(ffmpeg_bin, src, frame_dir, overwrite_frames)
        if n_frames < 3:
            n_frames = max(n_frames, probe_num_frames(src), 3)
        split_lines[subset].append(f"{rel_dir.as_posix()} {n_frames} {label_id}")

    if not labels:
        raise RuntimeError("No valid clip_path rows found for TSM preparation.")

    ordered = sorted(labels.items(), key=lambda kv: kv[1])
    (tsm_root / "category.txt").write_text(
        "\n".join(name for name, _ in ordered) + "\n",
        encoding="utf-8",
    )
    for subset in ("train", "val", "test"):
        out = tsm_root / f"{subset}_videofolder.txt"
        lines = split_lines.get(subset, [])
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    info = {
        "root_path": str(frames_root.resolve()),
        "train_list": str((tsm_root / "train_videofolder.txt").resolve()),
        "val_list": str((tsm_root / "val_videofolder.txt").resolve()),
        "test_list": str((tsm_root / "test_videofolder.txt").resolve()),
        "num_classes": len(labels),
        "skipped_rows": skipped,
    }
    (tsm_root / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[DONE] TSM data prepared at: {tsm_root}")
    print(json.dumps(info, indent=2))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_csv)
    class_map_path = Path(args.class_map)
    output_root = Path(args.output_root)
    task_a_root = Path(args.task_a_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(manifest_path)
    class_map = load_class_map(class_map_path)

    by_subset = defaultdict(list)
    missing_media = 0
    non_relative_paths = 0
    for row in rows:
        subset = ensure_subset(row.get("subset", "train"))
        label = str(row["label"])
        label_id = int(row["label_id"])
        expected = class_map.get(label)
        if expected is None:
            raise RuntimeError(f"Label {label} not in class_map")
        if expected != label_id:
            raise RuntimeError(f"Label id mismatch for {label}: manifest={label_id}, class_map={expected}")
        try:
            media = resolve_media_path(row, args.allow_video_fallback)
        except RuntimeError:
            missing_media += 1
            continue
        if not media.exists():
            missing_media += 1
            continue
        out_path = to_output_path(media, task_a_root, args.path_mode)
        if args.path_mode == "relative" and (":" in out_path[:4] or out_path.startswith("/")):
            non_relative_paths += 1
        by_subset[subset].append((out_path, label_id))

    if sum(len(v) for v in by_subset.values()) == 0:
        raise RuntimeError("No valid rows to export. Check manifest paths and --allow-video-fallback.")

    for target in ("videomaev2", "slowfast"):
        target_dir = output_root / target
        for subset in ("train", "val", "test"):
            write_kinetics_style_csv(target_dir / f"{subset}.csv", by_subset.get(subset, []))
        (target_dir / "class_map.json").write_text(
            json.dumps(class_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (target_dir / "README.txt").write_text(
            "Use with Kinetics-style loaders.\n"
            f"path_mode={args.path_mode}\n"
            f"task_a_root={task_a_root.resolve()}\n"
            "VideoMAEv2: set --data_path to this folder and --data_root to task_a_root.\n"
            "SlowFast: set DATA.PATH_TO_DATA_DIR to this folder and DATA.PATH_PREFIX to task_a_root.\n",
            encoding="utf-8",
        )
        print(f"[DONE] {target}: {target_dir}")

    stats = {
        "rows_in_manifest": len(rows),
        "rows_exported": sum(len(v) for v in by_subset.values()),
        "missing_or_invalid_media_rows": missing_media,
        "subset_counts": {k: len(v) for k, v in by_subset.items()},
        "path_mode": args.path_mode,
        "task_a_root": str(task_a_root.resolve()),
        "absolute_path_rows_in_relative_mode": non_relative_paths,
    }
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    if args.prepare_tsm:
        tsm_root = output_root / "tsm"
        frames_root = Path(args.tsm_frames_dir) if args.tsm_frames_dir else (tsm_root / "frames")
        ffmpeg_bin = resolve_ffmpeg_bin(args.ffmpeg)
        prepare_tsm(
            rows=rows,
            tsm_root=tsm_root,
            frames_root=frames_root,
            ffmpeg_bin=ffmpeg_bin,
            overwrite_frames=args.overwrite_frames,
        )


if __name__ == "__main__":
    main()
