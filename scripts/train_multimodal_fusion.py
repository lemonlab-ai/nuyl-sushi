import argparse
import atexit
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from train_multimodal_baseline import (  # type: ignore
        balanced_accuracy,
        build_feature,
        load_manifest,
        load_master_index,
        macro_f1,
        one_hot,
        softmax,
        subset_mask,
    )
except Exception:
    from scripts.train_multimodal_baseline import (  # type: ignore
        balanced_accuracy,
        build_feature,
        load_manifest,
        load_master_index,
        macro_f1,
        one_hot,
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
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "multimodal_fusion"

VIDEO_SLICE = slice(0, 13)
WEIGHT_SLICE = slice(13, 15)
GYRO_SLICE = slice(15, 22)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multimodal fusion MLP (video + gyro + weight) for Task A with missing-modality augmentation."
    )
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--master-json", default=str(DEFAULT_MASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--target-fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--disable-weight", action="store_true", help="Ignore weight modality and use zeros.")
    parser.add_argument("--disable-gyro", action="store_true", help="Ignore gyro modality and use zeros.")
    parser.add_argument("--mod-drop-video", type=float, default=0.05)
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


def apply_group_dropout(
    x: np.ndarray,
    rng: np.random.Generator,
    p_video: float,
    p_weight: float,
    p_gyro: float,
) -> np.ndarray:
    dropped = x.copy()
    for i in range(dropped.shape[0]):
        dv = bool(rng.random() < p_video)
        dw = bool(rng.random() < p_weight)
        dg = bool(rng.random() < p_gyro)
        if dv and dw and dg:
            keep = int(rng.integers(0, 3))
            dv = keep != 0
            dw = keep != 1
            dg = keep != 2
        if dv:
            dropped[i, VIDEO_SLICE] = 0.0
        if dw:
            dropped[i, WEIGHT_SLICE] = 0.0
        if dg:
            dropped[i, GYRO_SLICE] = 0.0
    return dropped


def build_quality_matrix(x_raw: np.ndarray) -> np.ndarray:
    if x_raw.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={x_raw.shape}")
    q_video = (np.linalg.norm(x_raw[:, VIDEO_SLICE], axis=1) > 1e-8).astype(np.float32)
    q_weight = (x_raw[:, WEIGHT_SLICE.start + 1] > 0.5).astype(np.float32)
    q_gyro = (x_raw[:, GYRO_SLICE.stop - 1] > 0.5).astype(np.float32)
    return np.stack([q_video, q_weight, q_gyro], axis=1).astype(np.float32)


def apply_group_dropout_with_quality(
    x: np.ndarray,
    q: np.ndarray,
    rng: np.random.Generator,
    p_video: float,
    p_weight: float,
    p_gyro: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dropped = x.copy()
    q_out = q.copy()
    for i in range(dropped.shape[0]):
        dv = bool(rng.random() < p_video)
        dw = bool(rng.random() < p_weight)
        dg = bool(rng.random() < p_gyro)
        if dv and dw and dg:
            keep = int(rng.integers(0, 3))
            dv = keep != 0
            dw = keep != 1
            dg = keep != 2
        if dv:
            dropped[i, VIDEO_SLICE] = 0.0
            q_out[i, 0] = 0.0
        if dw:
            dropped[i, WEIGHT_SLICE] = 0.0
            q_out[i, 1] = 0.0
        if dg:
            dropped[i, GYRO_SLICE] = 0.0
            q_out[i, 2] = 0.0
    return dropped, q_out


def sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def apply_reliability_gating(
    x: np.ndarray,
    q: np.ndarray,
    wg: np.ndarray,
    bg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    gates = sigmoid(q @ wg + bg[None, :]).astype(np.float32)
    out = x.copy()
    out[:, VIDEO_SLICE] *= gates[:, 0:1]
    out[:, WEIGHT_SLICE] *= gates[:, 1:2]
    out[:, GYRO_SLICE] *= gates[:, 2:3]
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
        x_drop, q_drop = apply_group_dropout_with_quality(x, q, rng, p_video=p_video, p_weight=p_weight, p_gyro=p_gyro)
        if use_reliability_gating:
            x_in, gates = apply_reliability_gating(x_drop, q_drop, wg, bg)
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
            grad_g[:, 0] = np.sum(grad_x_in[:, VIDEO_SLICE] * x_drop[:, VIDEO_SLICE], axis=1)
            grad_g[:, 1] = np.sum(grad_x_in[:, WEIGHT_SLICE] * x_drop[:, WEIGHT_SLICE], axis=1)
            grad_g[:, 2] = np.sum(grad_x_in[:, GYRO_SLICE] * x_drop[:, GYRO_SLICE], axis=1)
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
    use_reliability_gating: bool = False,
    disable_gating: bool = False,
) -> np.ndarray:
    x_eval = x
    if use_reliability_gating and (not disable_gating):
        x_eval, _ = apply_reliability_gating(x, q, wg, bg)
    _, logits = forward_mlp(x_eval, w1, b1, w2, b2)
    return logits.argmax(axis=1)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    if y_true.shape[0] == 0:
        return {"count": 0.0, "top1_accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0}
    acc = float(np.mean(y_true == y_pred))
    mf1 = macro_f1(y_true, y_pred, num_classes)
    bacc = balanced_accuracy(y_true, y_pred, num_classes)
    return {
        "count": float(y_true.shape[0]),
        "top1_accuracy": acc,
        "macro_f1": mf1,
        "balanced_accuracy": bacc,
    }


def apply_eval_ablation(x: np.ndarray, q: np.ndarray, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    out = x.copy()
    q_out = q.copy()
    if mode == "no_gyro":
        out[:, GYRO_SLICE] = 0.0
        q_out[:, 2] = 0.0
    elif mode == "no_weight":
        out[:, WEIGHT_SLICE] = 0.0
        q_out[:, 1] = 0.0
    elif mode == "video_only":
        out[:, WEIGHT_SLICE] = 0.0
        out[:, GYRO_SLICE] = 0.0
        q_out[:, 1] = 0.0
        q_out[:, 2] = 0.0
    elif mode == "sensor_only":
        out[:, VIDEO_SLICE] = 0.0
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
    use_reliability_gating: bool,
) -> Dict[str, Dict[str, float]]:
    modes = ["clean", "no_gyro", "no_weight", "video_only", "sensor_only"]
    out: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        xm, qm = apply_eval_ablation(x, q, mode)
        yp = predict_labels(
            xm,
            qm,
            w1,
            b1,
            w2,
            b2,
            wg,
            bg,
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
        job_type="train_task_a_fusion",
        config={
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "use_reliability_gating": bool(args.use_reliability_gating),
            "gating_weight_decay": float(args.gating_weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "target_fps": float(args.target_fps),
            "max_frames": int(args.max_frames),
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

    features: List[np.ndarray] = []
    labels: List[int] = []
    subsets: List[str] = []
    clip_ids: List[str] = []
    for row in manifest_rows:
        feat = build_feature(
            row,
            master_index,
            args.target_fps,
            args.max_frames,
            use_weight=not bool(args.disable_weight),
            use_gyro=not bool(args.disable_gyro),
        )
        features.append(feat.astype(np.float32))
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
    q = build_quality_matrix(x)

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
            "target_fps": float(args.target_fps),
            "max_frames": int(args.max_frames),
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
        model_type=np.array(["fusion_mlp"]),
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        num_classes=np.array([num_classes], dtype=np.int64),
        feature_dim=np.array([x.shape[1]], dtype=np.int64),
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
    print("[DONE] Multimodal fusion training finished.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
