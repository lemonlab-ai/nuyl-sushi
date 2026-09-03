import argparse
import atexit
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    from extract_task_b_features_deep import (  # type: ignore
        extract_video_features,
        infer_feature_dim,
        load_backbone,
        preprocess_frame,
        resolve_device,
    )
except Exception:
    from scripts.extract_task_b_features_deep import (  # type: ignore
        extract_video_features,
        infer_feature_dim,
        load_backbone,
        preprocess_frame,
        resolve_device,
    )

try:
    from train_multimodal_baseline import (  # type: ignore
        balanced_accuracy,
        gyro_feature_from_csv,
        load_manifest,
        load_master_index,
        macro_f1,
        one_hot,
        parse_bool,
        safe_float,
        softmax,
        subset_mask,
    )
except Exception:
    from scripts.train_multimodal_baseline import (  # type: ignore
        balanced_accuracy,
        gyro_feature_from_csv,
        load_manifest,
        load_master_index,
        macro_f1,
        one_hot,
        parse_bool,
        safe_float,
        softmax,
        subset_mask,
    )

try:
    from wandb_utils import finish_wandb_run, init_wandb_run, parse_wandb_tags, wandb_log  # type: ignore
except Exception:
    from scripts.wandb_utils import finish_wandb_run, init_wandb_run, parse_wandb_tags, wandb_log  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "artifacts" / "task_a" / "task_a_fine_manifest.csv"
DEFAULT_MASTER = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "multimodal_fusion_deepvideo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multimodal fusion using deep video features + gyro/weight with missing-modality augmentation."
    )
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--master-json", default=str(DEFAULT_MASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument("--video-backbone", choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument("--video-weights", choices=["kinetics400", "none"], default="kinetics400")
    parser.add_argument("--video-device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--target-fps", type=float, default=6.0)
    parser.add_argument("--clip-len", type=int, default=16)
    parser.add_argument("--clip-hop", type=int, default=8)
    parser.add_argument("--resize-short", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sampled-frames", type=int, default=0)

    parser.add_argument("--disable-weight", action="store_true", help="Ignore weight modality and use zeros.")
    parser.add_argument("--disable-gyro", action="store_true", help="Ignore gyro modality and use zeros.")
    parser.add_argument("--mod-drop-video", type=float, default=0.10)
    parser.add_argument("--mod-drop-weight", type=float, default=0.20)
    parser.add_argument("--mod-drop-gyro", type=float, default=0.20)
    parser.add_argument(
        "--use-reliability-gating",
        action="store_true",
        help="Enable modality reliability gating (video/weight/gyro) before fusion MLP.",
    )
    parser.add_argument("--gating-weight-decay", type=float, default=1e-4)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="ntnu-sushi")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="task_a")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_clip_preprocessed_frames(
    media_path: Path,
    start_sec: float,
    end_sec: float,
    target_fps: float,
    resize_short: int,
    crop_size: int,
    max_sampled_frames: int,
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Activate conda env first.")
    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        return np.zeros((0, crop_size, crop_size, 3), dtype=np.float32)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if end_sec <= start_sec:
        start_idx = 0
        end_idx = total if total > 0 else int(1e9)
    else:
        start_idx = max(0, int(round(start_sec * src_fps)))
        end_idx = max(start_idx + 1, int(round(end_sec * src_fps)))
        if total > 0:
            end_idx = min(end_idx, total)
    step = max(1, int(round(src_fps / max(target_fps, 0.1))))

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_idx))
    frames: List[np.ndarray] = []
    frame_idx = start_idx
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx >= end_idx:
            break
        if (frame_idx - start_idx) % step == 0:
            frames.append(preprocess_frame(frame, resize_short=resize_short, crop_size=crop_size))
            if max_sampled_frames > 0 and len(frames) >= max_sampled_frames:
                break
        frame_idx += 1
    cap.release()

    if not frames:
        return np.zeros((0, crop_size, crop_size, 3), dtype=np.float32)
    return np.stack(frames, axis=0).astype(np.float32)


def build_sensor_feature(
    row: Dict[str, str],
    master_index: Dict[str, Dict],
    use_weight: bool = True,
    use_gyro: bool = True,
) -> np.ndarray:
    if use_weight:
        weight_g_raw = row.get("weight_g", "")
        if weight_g_raw not in ("", None):
            weight_feat = np.array([safe_float(weight_g_raw, 0.0), 1.0], dtype=np.float32)
        else:
            weight_feat = np.array([0.0, 0.0], dtype=np.float32)
    else:
        weight_feat = np.array([0.0, 0.0], dtype=np.float32)

    video_id = str(row.get("video_id", ""))
    video_record = master_index.get(video_id, {})
    gyro_feat = np.zeros((7,), dtype=np.float32)
    if use_gyro:
        has_gyro = parse_bool(str(row.get("has_gyro", "false"))) or bool(video_record.get("has_gyro", False))
        gyro_path_raw = video_record.get("gyro_path")
        if has_gyro and gyro_path_raw:
            gyro_path = Path(str(gyro_path_raw))
            if not gyro_path.is_absolute():
                gyro_path = (ROOT_DIR / gyro_path).resolve()
            gyro_feat = gyro_feature_from_csv(gyro_path)
        if has_gyro and np.allclose(gyro_feat, 0.0):
            gyro_feat[-1] = 1.0

    return np.concatenate([weight_feat, gyro_feat], axis=0).astype(np.float32)  # 9-dim


def build_deep_video_feature(
    row: Dict[str, str],
    model,
    device: str,
    target_fps: float,
    clip_len: int,
    clip_hop: int,
    resize_short: int,
    crop_size: int,
    batch_size: int,
    max_sampled_frames: int,
    video_dim: int,
) -> np.ndarray:
    clip_path = Path((row.get("clip_path") or "").strip()) if (row.get("clip_path") or "").strip() else None
    video_path = Path(row.get("video_path", ""))
    start_sec = safe_float(row.get("start_sec", "0"), 0.0)
    end_sec = safe_float(row.get("end_sec", "0"), 0.0)

    if clip_path is not None and clip_path.exists():
        media = clip_path
        s, e = 0.0, 0.0
    else:
        media = video_path
        s, e = start_sec, end_sec
    if not media.exists():
        return np.zeros((video_dim,), dtype=np.float32)

    frames = sample_clip_preprocessed_frames(
        media_path=media,
        start_sec=s,
        end_sec=e,
        target_fps=target_fps,
        resize_short=resize_short,
        crop_size=crop_size,
        max_sampled_frames=max_sampled_frames,
    )
    if frames.shape[0] == 0:
        return np.zeros((video_dim,), dtype=np.float32)

    feats_t = extract_video_features(
        model=model,
        device=device,
        frames=frames,
        clip_len=clip_len,
        clip_hop=clip_hop,
        batch_size=batch_size,
    )
    if feats_t.ndim != 2 or feats_t.shape[0] == 0:
        return np.zeros((video_dim,), dtype=np.float32)
    return feats_t.mean(axis=0).astype(np.float32)


def apply_group_dropout(
    x: np.ndarray,
    rng: np.random.Generator,
    video_slice: slice,
    weight_slice: slice,
    gyro_slice: slice,
    p_video: float,
    p_weight: float,
    p_gyro: float,
) -> np.ndarray:
    out = x.copy()
    for i in range(out.shape[0]):
        dv = bool(rng.random() < p_video)
        dw = bool(rng.random() < p_weight)
        dg = bool(rng.random() < p_gyro)
        if dv and dw and dg:
            keep = int(rng.integers(0, 3))
            dv = keep != 0
            dw = keep != 1
            dg = keep != 2
        if dv:
            out[i, video_slice] = 0.0
        if dw:
            out[i, weight_slice] = 0.0
        if dg:
            out[i, gyro_slice] = 0.0
    return out


def build_quality_matrix(x_raw: np.ndarray, video_slice: slice, weight_slice: slice, gyro_slice: slice) -> np.ndarray:
    if x_raw.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={x_raw.shape}")
    q_video = (np.linalg.norm(x_raw[:, video_slice], axis=1) > 1e-8).astype(np.float32)
    q_weight = (x_raw[:, weight_slice.start + 1] > 0.5).astype(np.float32)
    q_gyro = (x_raw[:, gyro_slice.stop - 1] > 0.5).astype(np.float32)
    return np.stack([q_video, q_weight, q_gyro], axis=1).astype(np.float32)


def apply_group_dropout_with_quality(
    x: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
    video_slice: slice,
    weight_slice: slice,
    gyro_slice: slice,
    p_video: float,
    p_weight: float,
    p_gyro: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out = x.copy()
    q_out = q.copy()
    for i in range(out.shape[0]):
        dv = bool(rng.random() < p_video)
        dw = bool(rng.random() < p_weight)
        dg = bool(rng.random() < p_gyro)
        if dv and dw and dg:
            keep = int(rng.integers(0, 3))
            dv = keep != 0
            dw = keep != 1
            dg = keep != 2
        if dv:
            out[i, video_slice] = 0.0
            q_out[i, 0] = 0.0
        if dw:
            out[i, weight_slice] = 0.0
            q_out[i, 1] = 0.0
        if dg:
            out[i, gyro_slice] = 0.0
            q_out[i, 2] = 0.0
    return out, q_out


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def apply_reliability_gating(
    x: np.ndarray,
    q: np.ndarray,
    wg: np.ndarray,
    bg: np.ndarray,
    video_slice: slice,
    weight_slice: slice,
    gyro_slice: slice,
) -> Tuple[np.ndarray, np.ndarray]:
    gates = sigmoid(q @ wg + bg[None, :]).astype(np.float32)
    out = x.copy()
    out[:, video_slice] *= gates[:, 0:1]
    out[:, weight_slice] *= gates[:, 1:2]
    out[:, gyro_slice] *= gates[:, 2:3]
    return out, gates


def forward_mlp(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h_pre = x @ w1 + b1
    h = np.maximum(h_pre, 0.0)
    logits = h @ w2 + b2
    return h, logits


def train_fusion_mlp(
    x: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    use_reliability_gating: bool,
    gating_weight_decay: float,
    video_slice: slice,
    weight_slice: slice,
    gyro_slice: slice,
    p_video: float,
    p_weight: float,
    p_gyro: float,
    seed: int,
    wandb_run=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w1 = (rng.standard_normal((d, hidden_dim)) * np.sqrt(2.0 / max(d, 1))).astype(np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    w2 = (rng.standard_normal((hidden_dim, num_classes)) * np.sqrt(2.0 / max(hidden_dim, 1))).astype(np.float32)
    b2 = np.zeros((num_classes,), dtype=np.float32)
    wg = (np.eye(3, dtype=np.float32) * 4.0).astype(np.float32)
    bg = np.array([-2.0, -2.0, -2.0], dtype=np.float32)
    y_oh = one_hot(y, num_classes)
    losses: List[float] = []

    for epoch in range(epochs):
        x_drop, q_drop = apply_group_dropout_with_quality(
            x=x,
            q=q,
            rng=rng,
            video_slice=video_slice,
            weight_slice=weight_slice,
            gyro_slice=gyro_slice,
            p_video=p_video,
            p_weight=p_weight,
            p_gyro=p_gyro,
        )
        if use_reliability_gating:
            x_in, gates = apply_reliability_gating(
                x_drop, q_drop, wg, bg, video_slice=video_slice, weight_slice=weight_slice, gyro_slice=gyro_slice
            )
        else:
            x_in = x_drop
            gates = np.ones((x_drop.shape[0], 3), dtype=np.float32)
        h, logits = forward_mlp(x_in, w1, b1, w2, b2)
        probs = softmax(logits)
        ce = -float(np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)))
        reg = 0.5 * weight_decay * float(np.sum(w1 * w1) + np.sum(w2 * w2))
        if use_reliability_gating:
            reg += 0.5 * gating_weight_decay * float(np.sum(wg * wg))
        losses.append(ce + reg)
        if wandb_run is not None and ((epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch + 1 == epochs)):
            payload = {"epoch": int(epoch + 1), "train/loss": float(ce + reg)}
            if use_reliability_gating:
                payload.update(
                    {
                        "train/gate_video_mean": float(gates[:, 0].mean()),
                        "train/gate_weight_mean": float(gates[:, 1].mean()),
                        "train/gate_gyro_mean": float(gates[:, 2].mean()),
                    }
                )
            wandb_log(wandb_run, payload)

        grad_logits = (probs - y_oh) / n
        grad_w2 = h.T @ grad_logits + weight_decay * w2
        grad_b2 = grad_logits.sum(axis=0)

        grad_h = grad_logits @ w2.T
        grad_h[h <= 0] = 0.0
        grad_x_in = grad_h @ w1.T
        grad_w1 = x_in.T @ grad_h + weight_decay * w1
        grad_b1 = grad_h.sum(axis=0)

        if use_reliability_gating:
            grad_g = np.zeros((x_drop.shape[0], 3), dtype=np.float32)
            grad_g[:, 0] = np.sum(grad_x_in[:, video_slice] * x_drop[:, video_slice], axis=1)
            grad_g[:, 1] = np.sum(grad_x_in[:, weight_slice] * x_drop[:, weight_slice], axis=1)
            grad_g[:, 2] = np.sum(grad_x_in[:, gyro_slice] * x_drop[:, gyro_slice], axis=1)
            grad_z = grad_g * gates * (1.0 - gates)
            grad_wg = q_drop.T @ grad_z + gating_weight_decay * wg
            grad_bg = grad_z.sum(axis=0)
        else:
            grad_wg = np.zeros_like(wg)
            grad_bg = np.zeros_like(bg)

        w1 -= lr * grad_w1.astype(np.float32)
        b1 -= lr * grad_b1.astype(np.float32)
        w2 -= lr * grad_w2.astype(np.float32)
        b2 -= lr * grad_b2.astype(np.float32)
        if use_reliability_gating:
            wg -= lr * grad_wg.astype(np.float32)
            bg -= lr * grad_bg.astype(np.float32)

    return w1, b1, w2, b2, wg, bg, losses


def predict_labels(
    x: np.ndarray,
    q: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wg: np.ndarray,
    bg: np.ndarray,
    video_slice: slice,
    weight_slice: slice,
    gyro_slice: slice,
    use_reliability_gating: bool = False,
    disable_gating: bool = False,
) -> np.ndarray:
    x_eval = x
    if use_reliability_gating and (not disable_gating):
        x_eval, _ = apply_reliability_gating(
            x, q, wg, bg, video_slice=video_slice, weight_slice=weight_slice, gyro_slice=gyro_slice
        )
    _, logits = forward_mlp(x_eval, w1, b1, w2, b2)
    return logits.argmax(axis=1)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    if y_true.shape[0] == 0:
        return {"count": 0.0, "top1_accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0}
    return {
        "count": float(y_true.shape[0]),
        "top1_accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
    }


def apply_eval_ablation(
    x: np.ndarray,
    q: np.ndarray,
    mode: str,
    video_slice: slice,
    weight_slice: slice,
    gyro_slice: slice,
) -> Tuple[np.ndarray, np.ndarray]:
    out = x.copy()
    q_out = q.copy()
    if mode == "no_gyro":
        out[:, gyro_slice] = 0.0
        q_out[:, 2] = 0.0
    elif mode == "no_weight":
        out[:, weight_slice] = 0.0
        q_out[:, 1] = 0.0
    elif mode == "video_only":
        out[:, weight_slice] = 0.0
        out[:, gyro_slice] = 0.0
        q_out[:, 1] = 0.0
        q_out[:, 2] = 0.0
    elif mode == "sensor_only":
        out[:, video_slice] = 0.0
        q_out[:, 0] = 0.0
    return out, q_out


def eval_subset_with_modes(
    x: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    wg: np.ndarray,
    bg: np.ndarray,
    num_classes: int,
    video_slice: slice,
    weight_slice: slice,
    gyro_slice: slice,
    use_reliability_gating: bool,
) -> Dict[str, Dict[str, float]]:
    modes = ["clean", "no_gyro", "no_weight", "video_only", "sensor_only"]
    out: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        xm, qm = apply_eval_ablation(x, q, mode, video_slice, weight_slice, gyro_slice)
        yp = predict_labels(
            xm,
            qm,
            w1,
            b1,
            w2,
            b2,
            wg,
            bg,
            video_slice=video_slice,
            weight_slice=weight_slice,
            gyro_slice=gyro_slice,
            use_reliability_gating=use_reliability_gating,
        )
        out[mode] = evaluate(y, yp, num_classes)
    if use_reliability_gating:
        yp_no_gate = predict_labels(
            x,
            q,
            w1,
            b1,
            w2,
            b2,
            wg,
            bg,
            video_slice=video_slice,
            weight_slice=weight_slice,
            gyro_slice=gyro_slice,
            use_reliability_gating=True,
            disable_gating=True,
        )
        out["clean_no_gating"] = evaluate(y, yp_no_gate, num_classes)
    return out


def build_ablation_report(robustness: Dict[str, Dict[str, Dict[str, float]]], use_reliability_gating: bool) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for subset in ("val", "test"):
        sub = robustness.get(subset, {}) if isinstance(robustness, dict) else {}
        if not sub:
            out[subset] = {}
            continue
        plus_gyro = sub.get("clean_no_gating", sub.get("clean", {}))
        plus_gating = sub.get("clean", {})
        subset_out: Dict[str, Dict[str, float]] = {
            "video_only": sub.get("video_only", {}),
            "plus_weight": sub.get("no_gyro", {}),
            "plus_gyro": plus_gyro,
            "plus_gating": plus_gating if use_reliability_gating else plus_gyro,
        }
        if use_reliability_gating and sub.get("clean_no_gating"):
            subset_out["gating_gain_top1"] = {
                "value": float(plus_gating.get("top1_accuracy", 0.0) - plus_gyro.get("top1_accuracy", 0.0))
            }
        out[subset] = subset_out
    return out


def main() -> None:
    args = parse_args()
    wandb_run = init_wandb_run(
        enabled=bool(args.wandb),
        project=str(args.wandb_project),
        entity=str(args.wandb_entity),
        run_name=str(args.wandb_run_name),
        group=str(args.wandb_group),
        tags=parse_wandb_tags(args.wandb_tags),
        mode=str(args.wandb_mode),
        job_type="train_task_a_fusion_deepvideo",
        config={
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "use_reliability_gating": bool(args.use_reliability_gating),
            "gating_weight_decay": float(args.gating_weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "video_backbone": str(args.video_backbone),
            "video_weights": str(args.video_weights),
            "video_device": str(args.video_device),
            "target_fps": float(args.target_fps),
            "clip_len": int(args.clip_len),
            "clip_hop": int(args.clip_hop),
            "resize_short": int(args.resize_short),
            "crop_size": int(args.crop_size),
            "batch_size": int(args.batch_size),
            "max_sampled_frames": int(args.max_sampled_frames),
            "mod_drop_video": float(args.mod_drop_video),
            "mod_drop_weight": float(args.mod_drop_weight),
            "mod_drop_gyro": float(args.mod_drop_gyro),
            "use_weight": bool(not args.disable_weight),
            "use_gyro": bool(not args.disable_gyro),
            "seed": int(args.seed),
        },
    )
    if wandb_run is not None:
        atexit.register(lambda: finish_wandb_run(wandb_run))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(Path(args.manifest_csv))
    master_index = load_master_index(Path(args.master_json))

    device = resolve_device(args.video_device)
    model = load_backbone(args.video_backbone, args.video_weights)
    model.eval()
    model.to(device)
    video_dim = infer_feature_dim(model, device=device, clip_len=args.clip_len, crop_size=args.crop_size)
    sensor_dim = 9
    video_slice = slice(0, video_dim)
    weight_slice = slice(video_dim, video_dim + 2)
    gyro_slice = slice(video_dim + 2, video_dim + sensor_dim)

    features: List[np.ndarray] = []
    labels: List[int] = []
    subsets: List[str] = []
    clip_ids: List[str] = []
    for row in manifest_rows:
        video_feat = build_deep_video_feature(
            row=row,
            model=model,
            device=device,
            target_fps=args.target_fps,
            clip_len=args.clip_len,
            clip_hop=args.clip_hop,
            resize_short=args.resize_short,
            crop_size=args.crop_size,
            batch_size=args.batch_size,
            max_sampled_frames=args.max_sampled_frames,
            video_dim=video_dim,
        )
        sensor_feat = build_sensor_feature(
            row=row,
            master_index=master_index,
            use_weight=not bool(args.disable_weight),
            use_gyro=not bool(args.disable_gyro),
        )
        feat = np.concatenate([video_feat, sensor_feat], axis=0).astype(np.float32)
        features.append(feat)
        labels.append(int(row["label_id"]))
        subsets.append(str(row.get("subset", "train")).strip().lower())
        clip_ids.append(str(row.get("clip_id", "")))

    x = np.stack(features, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    num_classes = int(np.max(y)) + 1

    train_mask = subset_mask(subsets, ("train",))
    val_mask = subset_mask(subsets, ("val",))
    test_mask = subset_mask(subsets, ("test",))
    if not np.any(train_mask):
        raise RuntimeError("No train subset rows found in manifest.")

    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    x_norm = (x - mean) / std
    q = build_quality_matrix(x, video_slice=video_slice, weight_slice=weight_slice, gyro_slice=gyro_slice)

    w1, b1, w2, b2, wg, bg, losses = train_fusion_mlp(
        x=x_norm[train_mask],
        q=q[train_mask],
        y=y[train_mask],
        num_classes=num_classes,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        use_reliability_gating=bool(args.use_reliability_gating),
        gating_weight_decay=float(args.gating_weight_decay),
        video_slice=video_slice,
        weight_slice=weight_slice,
        gyro_slice=gyro_slice,
        p_video=float(args.mod_drop_video),
        p_weight=float(args.mod_drop_weight),
        p_gyro=float(args.mod_drop_gyro),
        seed=int(args.seed),
        wandb_run=wandb_run,
    )

    train_pred = predict_labels(
        x_norm[train_mask],
        q[train_mask],
        w1,
        b1,
        w2,
        b2,
        wg,
        bg,
        video_slice=video_slice,
        weight_slice=weight_slice,
        gyro_slice=gyro_slice,
        use_reliability_gating=bool(args.use_reliability_gating),
    )
    val_pred = (
        predict_labels(
            x_norm[val_mask],
            q[val_mask],
            w1,
            b1,
            w2,
            b2,
            wg,
            bg,
            video_slice=video_slice,
            weight_slice=weight_slice,
            gyro_slice=gyro_slice,
            use_reliability_gating=bool(args.use_reliability_gating),
        )
        if np.any(val_mask)
        else np.array([], dtype=np.int64)
    )
    test_pred = (
        predict_labels(
            x_norm[test_mask],
            q[test_mask],
            w1,
            b1,
            w2,
            b2,
            wg,
            bg,
            video_slice=video_slice,
            weight_slice=weight_slice,
            gyro_slice=gyro_slice,
            use_reliability_gating=bool(args.use_reliability_gating),
        )
        if np.any(test_mask)
        else np.array([], dtype=np.int64)
    )

    metrics = {
        "train": evaluate(y[train_mask], train_pred, num_classes),
        "val": evaluate(y[val_mask], val_pred, num_classes),
        "test": evaluate(y[test_mask], test_pred, num_classes),
        "robustness": {
            "val": eval_subset_with_modes(
                x_norm[val_mask],
                q[val_mask],
                y[val_mask],
                w1,
                b1,
                w2,
                b2,
                wg,
                bg,
                num_classes,
                video_slice,
                weight_slice,
                gyro_slice,
                use_reliability_gating=bool(args.use_reliability_gating),
            )
            if np.any(val_mask)
            else {},
            "test": eval_subset_with_modes(
                x_norm[test_mask],
                q[test_mask],
                y[test_mask],
                w1,
                b1,
                w2,
                b2,
                wg,
                bg,
                num_classes,
                video_slice,
                weight_slice,
                gyro_slice,
                use_reliability_gating=bool(args.use_reliability_gating),
            )
            if np.any(test_mask)
            else {},
        },
        "config": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "video_backbone": args.video_backbone,
            "video_weights": args.video_weights,
            "video_device": device,
            "target_fps": float(args.target_fps),
            "clip_len": int(args.clip_len),
            "clip_hop": int(args.clip_hop),
            "resize_short": int(args.resize_short),
            "crop_size": int(args.crop_size),
            "batch_size": int(args.batch_size),
            "max_sampled_frames": int(args.max_sampled_frames),
            "video_feature_dim": int(video_dim),
            "sensor_dim": int(sensor_dim),
            "feature_dim": int(x.shape[1]),
            "num_classes": num_classes,
            "train_count": int(train_mask.sum()),
            "val_count": int(val_mask.sum()),
            "test_count": int(test_mask.sum()),
            "mod_drop_video": float(args.mod_drop_video),
            "mod_drop_weight": float(args.mod_drop_weight),
            "mod_drop_gyro": float(args.mod_drop_gyro),
            "use_reliability_gating": bool(args.use_reliability_gating),
            "gating_weight_decay": float(args.gating_weight_decay),
            "use_weight": bool(not args.disable_weight),
            "use_gyro": bool(not args.disable_gyro),
        },
    }
    metrics["ablation_report"] = build_ablation_report(
        metrics.get("robustness", {}),
        use_reliability_gating=bool(args.use_reliability_gating),
    )

    np.savez_compressed(
        out_dir / "model.npz",
        model_type=np.array(["fusion_mlp_deepvideo"]),
        video_feature_type=np.array(["deep_backbone"]),
        video_backbone=np.array([args.video_backbone]),
        video_weights=np.array([args.video_weights]),
        video_device=np.array([device]),
        target_fps=np.array([float(args.target_fps)], dtype=np.float32),
        clip_len=np.array([int(args.clip_len)], dtype=np.int64),
        clip_hop=np.array([int(args.clip_hop)], dtype=np.int64),
        resize_short=np.array([int(args.resize_short)], dtype=np.int64),
        crop_size=np.array([int(args.crop_size)], dtype=np.int64),
        batch_size=np.array([int(args.batch_size)], dtype=np.int64),
        max_sampled_frames=np.array([int(args.max_sampled_frames)], dtype=np.int64),
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        num_classes=np.array([num_classes], dtype=np.int64),
        feature_dim=np.array([x.shape[1]], dtype=np.int64),
        video_feature_dim=np.array([video_dim], dtype=np.int64),
        sensor_dim=np.array([sensor_dim], dtype=np.int64),
        hidden_dim=np.array([int(args.hidden_dim)], dtype=np.int64),
        use_weight=np.array([int(not args.disable_weight)], dtype=np.int64),
        use_gyro=np.array([int(not args.disable_gyro)], dtype=np.int64),
        use_reliability_gating=np.array([int(bool(args.use_reliability_gating))], dtype=np.int64),
        gating_weight_decay=np.array([float(args.gating_weight_decay)], dtype=np.float32),
        wg=wg.astype(np.float32),
        bg=bg.astype(np.float32),
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "loss_curve.json").write_text(
        json.dumps({"loss": [float(v) for v in losses]}, indent=2), encoding="utf-8"
    )
    (out_dir / "ablation_report.json").write_text(
        json.dumps(metrics.get("ablation_report", {}), indent=2), encoding="utf-8"
    )
    ablation_rows = []
    for subset_name, subset_payload in metrics.get("ablation_report", {}).items():
        if not isinstance(subset_payload, dict):
            continue
        for scenario, vals in subset_payload.items():
            if not isinstance(vals, dict):
                continue
            if {"top1_accuracy", "macro_f1", "balanced_accuracy"}.issubset(vals.keys()):
                ablation_rows.append(
                    {
                        "subset": subset_name,
                        "scenario": scenario,
                        "top1_accuracy": float(vals.get("top1_accuracy", 0.0)),
                        "macro_f1": float(vals.get("macro_f1", 0.0)),
                        "balanced_accuracy": float(vals.get("balanced_accuracy", 0.0)),
                    }
                )
    if ablation_rows:
        with (out_dir / "ablation_report.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["subset", "scenario", "top1_accuracy", "macro_f1", "balanced_accuracy"],
            )
            writer.writeheader()
            writer.writerows(ablation_rows)

    pred_all = predict_labels(
        x_norm,
        q,
        w1,
        b1,
        w2,
        b2,
        wg,
        bg,
        video_slice=video_slice,
        weight_slice=weight_slice,
        gyro_slice=gyro_slice,
        use_reliability_gating=bool(args.use_reliability_gating),
    )
    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "subset", "label_id", "pred_label_id"])
        writer.writeheader()
        for cid, subset, gt, pd in zip(clip_ids, subsets, y.tolist(), pred_all.tolist()):
            writer.writerow({"clip_id": cid, "subset": subset, "label_id": gt, "pred_label_id": pd})

    wandb_log(
        wandb_run,
        {
            "summary/train_top1": float(metrics["train"]["top1_accuracy"]),
            "summary/val_top1": float(metrics["val"]["top1_accuracy"]),
            "summary/test_top1": float(metrics["test"]["top1_accuracy"]),
            "summary/train_macro_f1": float(metrics["train"]["macro_f1"]),
            "summary/val_macro_f1": float(metrics["val"]["macro_f1"]),
            "summary/test_macro_f1": float(metrics["test"]["macro_f1"]),
        },
    )
    print("[DONE] Multimodal deepvideo fusion training finished.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
