import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from extract_task_b_features_deep import (  # type: ignore
        extract_video_features,
        infer_feature_dim,
        load_backbone,
        resolve_device,
        sample_and_preprocess_frames,
    )
except Exception:
    from scripts.extract_task_b_features_deep import (  # type: ignore
        extract_video_features,
        infer_feature_dim,
        load_backbone,
        resolve_device,
        sample_and_preprocess_frames,
    )


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GT_JSON = ROOT_DIR / "artifacts" / "task_b" / "actionformer_fine.json"
DEFAULT_OUT_DIR = ROOT_DIR / "artifacts" / "task_b" / "features_deep_ensemble"

SUPPORTED_MODELS = {"r3d_18", "mc3_18", "r2plus1d_18"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract stronger Task B features by concatenating multiple deep backbones per video."
    )
    parser.add_argument("--gt-json", default=str(DEFAULT_GT_JSON))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--models",
        default="r3d_18,mc3_18,r2plus1d_18",
        help="Comma-separated backbones from {r3d_18,mc3_18,r2plus1d_18}.",
    )
    parser.add_argument("--weights", choices=["kinetics400", "none"], default="kinetics400")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--target-fps", type=float, default=8.0)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--clip-hop", type=int, default=8)
    parser.add_argument("--resize-short", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sampled-frames", type=int, default=0)
    return parser.parse_args()


def parse_models(raw: str) -> List[str]:
    models = [x.strip() for x in raw.split(",") if x.strip()]
    if not models:
        raise ValueError("No valid models parsed from --models")
    bad = [m for m in models if m not in SUPPORTED_MODELS]
    if bad:
        raise ValueError(f"Unsupported models: {bad}. Allowed={sorted(SUPPORTED_MODELS)}")
    return models


def resample_feature_time(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={x.shape}")
    t, d = x.shape
    if target_len <= 0:
        return np.zeros((1, d), dtype=np.float32)
    if t == target_len:
        return x.astype(np.float32)
    if t <= 1:
        return np.repeat(x[:1], target_len, axis=0).astype(np.float32)
    src_idx = np.linspace(0.0, 1.0, t, dtype=np.float32)
    dst_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.zeros((target_len, d), dtype=np.float32)
    for j in range(d):
        out[:, j] = np.interp(dst_idx, src_idx, x[:, j].astype(np.float32))
    return out


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

    models = parse_models(args.models)
    device = resolve_device(args.device)

    backbones = {}
    dims = {}
    for model_name in models:
        model = load_backbone(model_name, args.weights)
        model.eval()
        model.to(device)
        dim = infer_feature_dim(model, device=device, clip_len=args.clip_len, crop_size=args.crop_size)
        backbones[model_name] = model
        dims[model_name] = dim

    total_dim = int(sum(dims.values()))
    stats = {
        "num_videos": 0,
        "num_ok": 0,
        "num_missing": 0,
        "num_empty": 0,
        "models": models,
        "weights": args.weights,
        "device": device,
        "target_fps": args.target_fps,
        "clip_len": args.clip_len,
        "clip_hop": args.clip_hop,
        "input_dim": total_dim,
        "model_dims": dims,
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
            feats = np.zeros((1, total_dim), dtype=np.float32)
            np.save(out_dir / f"{video_id}.npy", feats)
            stats["videos"][video_id] = {
                "status": "empty_video",
                "source_path": str(src),
                "src_fps": src_fps,
                "sampled_frames": 0,
                "feature_shape": list(feats.shape),
            }
            continue

        per_model_feats: List[np.ndarray] = []
        max_t = 1
        for model_name in models:
            feats_i = extract_video_features(
                model=backbones[model_name],
                device=device,
                frames=frames,
                clip_len=args.clip_len,
                clip_hop=args.clip_hop,
                batch_size=args.batch_size,
            ).astype(np.float32)
            per_model_feats.append(feats_i)
            max_t = max(max_t, int(feats_i.shape[0]))

        aligned = [resample_feature_time(f, max_t) for f in per_model_feats]
        feats = np.concatenate(aligned, axis=1).astype(np.float32)
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
                "models": models,
                "weights": args.weights,
                "device": device,
                "input_dim": total_dim,
                "model_dims": dims,
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
    print("[DONE] Ensemble ActionFormer features extracted.")
    print(json.dumps({k: v for k, v in stats.items() if k != "videos"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
