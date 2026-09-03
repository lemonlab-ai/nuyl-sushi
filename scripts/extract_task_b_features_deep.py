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
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "task_b" / "features_deep_r3d18"

# Kinetics-400 normalization (common for torchvision video backbones)
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deep temporal .npy features per video for ActionFormer using torchvision 3D backbones."
    )
    parser.add_argument("--gt-json", default=str(DEFAULT_GT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument(
        "--weights",
        choices=["kinetics400", "none"],
        default="kinetics400",
        help="Backbone weights source.",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--target-fps", type=float, default=8.0, help="Sampling FPS before clip windowing.")
    parser.add_argument("--clip-len", type=int, default=16, help="Frames per clip for 3D backbone.")
    parser.add_argument("--clip-hop", type=int, default=8, help="Hop size between clips (sampled frames).")
    parser.add_argument("--resize-short", type=int, default=128, help="Resize shorter side before center crop.")
    parser.add_argument("--crop-size", type=int, default=112, help="Center crop size.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for clip feature inference.")
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=0,
        help="Optional cap for sampled frames per video (0 means no cap).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone(model_name: str, weights_name: str):
    try:
        import torch.nn as nn  # type: ignore
        import torchvision.models.video as tv_video  # type: ignore
    except Exception as exc:
        raise RuntimeError("torch + torchvision are required for deep feature extraction.") from exc

    builder = getattr(tv_video, model_name)
    kwargs = {}
    if weights_name == "kinetics400":
        weights_attr = {
            "r3d_18": "R3D_18_Weights",
            "mc3_18": "MC3_18_Weights",
            "r2plus1d_18": "R2Plus1D_18_Weights",
        }[model_name]
        if hasattr(tv_video, weights_attr):
            weights_enum = getattr(tv_video, weights_attr)
            kwargs["weights"] = weights_enum.DEFAULT
        else:
            # torchvision<0.13 style
            kwargs["pretrained"] = True

    model = builder(**kwargs)
    if not hasattr(model, "fc"):
        raise RuntimeError(f"Unexpected torchvision video model structure: {model_name}")
    model.fc = nn.Identity()
    return model


def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    ch = min(crop_size, h)
    cw = min(crop_size, w)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    crop = img[y0 : y0 + ch, x0 : x0 + cw]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return crop


def preprocess_frame(frame_bgr: np.ndarray, resize_short: int, crop_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.float32)

    scale = float(resize_short) / float(min(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    crop = center_crop(resized, crop_size)

    x = crop.astype(np.float32) / 255.0
    x = (x - KINETICS_MEAN) / KINETICS_STD
    return x


def sample_and_preprocess_frames(
    video_path: Path,
    target_fps: float,
    resize_short: int,
    crop_size: int,
    max_sampled_frames: int,
) -> Tuple[np.ndarray, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Activate conda env first.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, crop_size, crop_size, 3), dtype=np.float32), 0.0

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
            frames.append(preprocess_frame(frame, resize_short, crop_size))
            if max_sampled_frames > 0 and len(frames) >= max_sampled_frames:
                break
        frame_idx += 1
    cap.release()
    if not frames:
        return np.zeros((0, crop_size, crop_size, 3), dtype=np.float32), src_fps
    return np.stack(frames, axis=0).astype(np.float32), src_fps


def build_clip_starts(num_frames: int, clip_len: int, clip_hop: int) -> List[int]:
    if num_frames <= 0:
        return [0]
    if num_frames <= clip_len:
        return [0]
    starts = list(range(0, num_frames - clip_len + 1, max(1, clip_hop)))
    last = num_frames - clip_len
    if starts[-1] != last:
        starts.append(last)
    return starts


def clip_tensor_from_start(frames: np.ndarray, start: int, clip_len: int):
    import torch  # type: ignore

    t = frames.shape[0]
    if t == 0:
        clip = np.zeros((clip_len, frames.shape[1], frames.shape[2], frames.shape[3]), dtype=np.float32)
    else:
        end = min(start + clip_len, t)
        clip = frames[start:end]
        if clip.shape[0] < clip_len:
            pad = np.repeat(clip[-1][None, ...], clip_len - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)
    x = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()  # C,T,H,W
    return x


def infer_feature_dim(model, device: str, clip_len: int, crop_size: int) -> int:
    import torch  # type: ignore

    with torch.no_grad():
        x = torch.zeros((1, 3, clip_len, crop_size, crop_size), dtype=torch.float32, device=device)
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        out = out.reshape(out.shape[0], -1)
        return int(out.shape[1])


def extract_video_features(
    model,
    device: str,
    frames: np.ndarray,
    clip_len: int,
    clip_hop: int,
    batch_size: int,
) -> np.ndarray:
    import torch  # type: ignore

    starts = build_clip_starts(frames.shape[0], clip_len, clip_hop)
    clips = [clip_tensor_from_start(frames, s, clip_len) for s in starts]
    feats: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(clips), max(1, batch_size)):
            batch = torch.stack(clips[i : i + max(1, batch_size)], dim=0).to(device=device, dtype=torch.float32)
            out = model(batch)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.reshape(out.shape[0], -1)
            feats.append(out.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feats, axis=0) if feats else np.zeros((1, 1), dtype=np.float32)


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

    device = resolve_device(args.device)
    model = load_backbone(args.model, args.weights)
    model.eval()
    model.to(device)
    feature_dim = infer_feature_dim(model, device=device, clip_len=args.clip_len, crop_size=args.crop_size)

    stats = {
        "num_videos": 0,
        "num_ok": 0,
        "num_missing": 0,
        "num_empty": 0,
        "model": args.model,
        "weights": args.weights,
        "device": device,
        "target_fps": args.target_fps,
        "clip_len": args.clip_len,
        "clip_hop": args.clip_hop,
        "resize_short": args.resize_short,
        "crop_size": args.crop_size,
        "input_dim": feature_dim,
        "videos": {},
    }

    for video_id, item in database.items():
        src = Path(str(item.get("source_path", "")))
        stats["num_videos"] += 1

        if not src.exists():
            stats["num_missing"] += 1
            stats["videos"][video_id] = {"status": "missing", "source_path": str(src)}
            continue

        frames, src_fps = sample_and_preprocess_frames(
            video_path=src,
            target_fps=args.target_fps,
            resize_short=args.resize_short,
            crop_size=args.crop_size,
            max_sampled_frames=args.max_sampled_frames,
        )
        if frames.shape[0] == 0:
            stats["num_empty"] += 1
            feats = np.zeros((1, feature_dim), dtype=np.float32)
            np.save(out_dir / f"{video_id}.npy", feats)
            stats["videos"][video_id] = {
                "status": "empty_video",
                "source_path": str(src),
                "src_fps": src_fps,
                "sampled_frames": 0,
                "feature_shape": list(feats.shape),
            }
            continue

        feats = extract_video_features(
            model=model,
            device=device,
            frames=frames,
            clip_len=args.clip_len,
            clip_hop=args.clip_hop,
            batch_size=args.batch_size,
        )
        np.save(out_dir / f"{video_id}.npy", feats)
        stats["num_ok"] += 1
        stats["videos"][video_id] = {
            "status": "ok",
            "source_path": str(src),
            "src_fps": src_fps,
            "sampled_frames": int(frames.shape[0]),
            "feature_shape": list(feats.shape),
        }

    (out_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "feature_meta.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "weights": args.weights,
                "device": device,
                "input_dim": feature_dim,
                "target_fps": args.target_fps,
                "clip_len": args.clip_len,
                "clip_hop": args.clip_hop,
                "resize_short": args.resize_short,
                "crop_size": args.crop_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[DONE] Deep ActionFormer features extracted.")
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k != "videos"},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

