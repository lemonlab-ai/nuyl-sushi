import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GT_JSON = ROOT_DIR / "artifacts" / "task_b" / "actionformer_fine.json"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "task_b" / "features_basic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract basic temporal .npy features per video for ActionFormer (CPU-only baseline)."
    )
    parser.add_argument("--gt-json", default=str(DEFAULT_GT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--target-fps", type=float, default=2.0, help="Sampling fps for feature extraction")
    parser.add_argument("--window-size", type=int, default=16, help="Frames per feature vector window")
    parser.add_argument("--window-hop", type=int, default=8, help="Window hop in sampled frames")
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="Optional cap for sampled frames per video (0 means no cap)",
    )
    return parser.parse_args()


def sample_frames(video_path: Path, target_fps: float, max_sampled_frames: int) -> Tuple[List[np.ndarray], float]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Activate conda env `ntnu-sushi` first.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], 0.0
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 30.0
    step = max(1, int(round(src_fps / max(target_fps, 0.1))))

    frames: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            frames.append(frame)
            if max_sampled_frames > 0 and len(frames) >= max_sampled_frames:
                break
        frame_idx += 1
    cap.release()
    return frames, src_fps


def window_feature(chunk: List[np.ndarray]) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Activate conda env `ntnu-sushi` first.")
    arr = np.stack(chunk).astype(np.float32) / 255.0  # T,H,W,C (BGR)
    bgr_mean = arr.mean(axis=(0, 1, 2))
    bgr_std = arr.std(axis=(0, 1, 2))

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for f in chunk]
    gray_stack = np.stack(grays, axis=0)

    if gray_stack.shape[0] > 1:
        diffs = np.abs(np.diff(gray_stack, axis=0))
        motion_mean = float(diffs.mean())
        motion_std = float(diffs.std())
    else:
        motion_mean = 0.0
        motion_std = 0.0

    mid_gray = (gray_stack[gray_stack.shape[0] // 2] * 255.0).astype(np.uint8)
    edges = cv2.Canny(mid_gray, 100, 200)
    edge_density = float((edges > 0).mean())

    hist, _ = np.histogram(gray_stack, bins=4, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)

    feat = np.concatenate(
        [
            bgr_mean,  # 3
            bgr_std,  # 3
            np.array([motion_mean, motion_std, edge_density], dtype=np.float32),  # 3
            hist,  # 4
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def build_video_features(frames: List[np.ndarray], window_size: int, window_hop: int) -> np.ndarray:
    if not frames:
        return np.zeros((1, 13), dtype=np.float32)
    win = max(2, window_size)
    hop = max(1, window_hop)

    feats: List[np.ndarray] = []
    n = len(frames)
    if n < win:
        feats.append(window_feature(frames))
    else:
        i = 0
        while i < n:
            j = min(i + win, n)
            chunk = frames[i:j]
            if len(chunk) < 2:
                break
            feats.append(window_feature(chunk))
            if j == n:
                break
            i += hop
    if not feats:
        feats.append(window_feature(frames[: min(n, win)]))
    return np.stack(feats, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    gt_json = Path(args.gt_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not gt_json.exists():
        raise FileNotFoundError(gt_json)
    payload = json.loads(gt_json.read_text(encoding="utf-8"))
    database: Dict[str, Dict] = payload.get("database", {})
    if not isinstance(database, dict):
        raise RuntimeError("Invalid gt-json: database must be a dict")

    stats = {
        "num_videos": 0,
        "num_ok": 0,
        "num_missing": 0,
        "target_fps": args.target_fps,
        "window_size": args.window_size,
        "window_hop": args.window_hop,
        "input_dim": 13,
        "videos": {},
    }

    for video_id, item in database.items():
        src = Path(str(item.get("source_path", "")))
        stats["num_videos"] += 1
        if not src.exists():
            stats["num_missing"] += 1
            stats["videos"][video_id] = {"status": "missing", "source_path": str(src)}
            continue

        frames, src_fps = sample_frames(src, args.target_fps, args.max_sampled_frames)
        feats = build_video_features(frames, args.window_size, args.window_hop)
        np.save(out_dir / f"{video_id}.npy", feats)
        stats["num_ok"] += 1
        stats["videos"][video_id] = {
            "status": "ok",
            "source_path": str(src),
            "src_fps": src_fps,
            "sampled_frames": len(frames),
            "feature_shape": list(feats.shape),
        }

    (out_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "feature_meta.json").write_text(
        json.dumps(
            {
                "input_dim": 13,
                "target_fps": args.target_fps,
                "window_size": args.window_size,
                "window_hop": args.window_hop,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[DONE] Basic ActionFormer features extracted.")
    print(json.dumps({k: v for k, v in stats.items() if k != "videos"}, indent=2))


if __name__ == "__main__":
    main()
