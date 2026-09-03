import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from wandb_utils import finish_wandb_run, init_wandb_run, parse_wandb_tags, wandb_log  # type: ignore
except Exception:
    from scripts.wandb_utils import finish_wandb_run, init_wandb_run, parse_wandb_tags, wandb_log  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "artifacts" / "task_a" / "task_a_fine_manifest.csv"
DEFAULT_MASTER = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "multimodal_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multimodal baseline (video + gyro + weight) for Task A")
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--master-json", default=str(DEFAULT_MASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-fps", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--disable-weight", action="store_true", help="Ignore weight modality and use zeros.")
    parser.add_argument("--disable-gyro", action="store_true", help="Ignore gyro modality and use zeros.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="ntnu-sushi")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="task_a")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def load_master_index(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload.get("videos", {})
    return videos if isinstance(videos, dict) else {}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s: List[float] = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    recalls: List[float] = []
    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    return float(np.mean(recalls))


def sample_video_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    target_fps: float,
    max_frames: int,
) -> List[np.ndarray]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Activate conda env `ntnu-sushi` first.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_idx = max(0, int(start_sec * fps))
    end_idx = int(end_sec * fps) if end_sec > start_sec else total
    if end_idx <= start_idx:
        end_idx = total
    end_idx = min(total, end_idx)
    step = max(1, int(round(fps / max(target_fps, 0.1))))

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx < start_idx:
            idx += 1
            continue
        if idx >= end_idx:
            break
        if ((idx - start_idx) % step) == 0:
            frames.append(frame)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def video_feature_from_frames(frames: List[np.ndarray]) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Activate conda env `ntnu-sushi` first.")
    if not frames:
        return np.zeros((13,), dtype=np.float32)
    arr = np.stack(frames).astype(np.float32) / 255.0
    bgr_mean = arr.mean(axis=(0, 1, 2))
    bgr_std = arr.std(axis=(0, 1, 2))
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for f in frames]
    gray_stack = np.stack(grays, axis=0)
    if gray_stack.shape[0] > 1:
        diffs = np.abs(np.diff(gray_stack, axis=0))
        motion_mean = float(diffs.mean())
        motion_std = float(diffs.std())
    else:
        motion_mean, motion_std = 0.0, 0.0
    mid_gray = (gray_stack[gray_stack.shape[0] // 2] * 255.0).astype(np.uint8)
    edges = cv2.Canny(mid_gray, 100, 200)
    edge_density = float((edges > 0).mean())
    hist, _ = np.histogram(gray_stack, bins=4, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= max(float(hist.sum()), 1.0)
    return np.concatenate(
        [
            bgr_mean,
            bgr_std,
            np.array([motion_mean, motion_std, edge_density], dtype=np.float32),
            hist,
        ],
        axis=0,
    ).astype(np.float32)


def gyro_feature_from_csv(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((7,), dtype=np.float32)
    numeric_rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            vals: List[float] = []
            for item in row:
                try:
                    vals.append(float(item))
                except Exception:
                    continue
            if vals:
                numeric_rows.append(vals)
    if not numeric_rows:
        return np.zeros((7,), dtype=np.float32)
    width = min(3, max(len(r) for r in numeric_rows))
    arr = np.zeros((len(numeric_rows), width), dtype=np.float32)
    for i, r in enumerate(numeric_rows):
        for j in range(min(width, len(r))):
            arr[i, j] = r[j]
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    out = np.zeros((7,), dtype=np.float32)
    out[0:width] = means[:width]
    out[3 : 3 + width] = stds[:width]
    out[-1] = 1.0
    return out


def build_feature(
    row: Dict[str, str],
    master_index: Dict[str, Dict],
    target_fps: float,
    max_frames: int,
    use_weight: bool = True,
    use_gyro: bool = True,
) -> np.ndarray:
    clip_path = Path((row.get("clip_path") or "").strip()) if (row.get("clip_path") or "").strip() else None
    video_path = Path(row.get("video_path", ""))
    start_sec = safe_float(row.get("start_sec", "0"), 0.0)
    end_sec = safe_float(row.get("end_sec", "0"), 0.0)

    frames: List[np.ndarray]
    if clip_path and clip_path.exists():
        frames = sample_video_frames(clip_path, 0.0, 1e9, target_fps, max_frames)
    else:
        frames = sample_video_frames(video_path, start_sec, end_sec, target_fps, max_frames)
    video_feat = video_feature_from_frames(frames)  # 13

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

    return np.concatenate([video_feat, weight_feat, gyro_feat], axis=0).astype(np.float32)


def train_softmax(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    wandb_run=None,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    rng = np.random.default_rng(seed)
    n, d = x.shape
    w = (rng.standard_normal((d, num_classes)) * 0.01).astype(np.float32)
    b = np.zeros((num_classes,), dtype=np.float32)
    y_oh = one_hot(y, num_classes)
    losses: List[float] = []

    for epoch in range(epochs):
        logits = x @ w + b
        probs = softmax(logits)
        ce = -np.mean(np.sum(y_oh * np.log(np.clip(probs, 1e-12, 1.0)), axis=1))
        reg = 0.5 * weight_decay * float(np.sum(w * w))
        loss = ce + reg
        losses.append(loss)
        if wandb_run is not None and ((epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch + 1 == epochs)):
            wandb_log(wandb_run, {"epoch": int(epoch + 1), "train/loss": float(loss)})

        grad = (probs - y_oh) / n
        grad_w = x.T @ grad + weight_decay * w
        grad_b = grad.mean(axis=0)
        w -= lr * grad_w.astype(np.float32)
        b -= lr * grad_b.astype(np.float32)
    return w, b, losses


def evaluate(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, num_classes: int) -> Dict[str, float]:
    if x.shape[0] == 0:
        return {"count": 0.0, "top1_accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0}
    logits = x @ w + b
    pred = logits.argmax(axis=1)
    acc = float(np.mean(pred == y))
    mf1 = macro_f1(y, pred, num_classes)
    bacc = balanced_accuracy(y, pred, num_classes)
    return {
        "count": float(x.shape[0]),
        "top1_accuracy": acc,
        "macro_f1": mf1,
        "balanced_accuracy": bacc,
    }


def subset_mask(subsets: List[str], allow: Tuple[str, ...]) -> np.ndarray:
    allow_set = set(allow)
    return np.array([s in allow_set for s in subsets], dtype=bool)


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
        job_type="train_task_a_linear",
        config={
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "target_fps": float(args.target_fps),
            "max_frames": int(args.max_frames),
            "use_weight": bool(not args.disable_weight),
            "use_gyro": bool(not args.disable_gyro),
            "seed": int(args.seed),
        },
    )

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

    try:
        w, b, losses = train_softmax(
            x=x_norm[train_mask],
            y=y[train_mask],
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            wandb_run=wandb_run,
        )

        metrics = {
            "train": evaluate(x_norm[train_mask], y[train_mask], w, b, num_classes),
            "val": evaluate(x_norm[val_mask], y[val_mask], w, b, num_classes),
            "test": evaluate(x_norm[test_mask], y[test_mask], w, b, num_classes),
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "target_fps": args.target_fps,
                "max_frames": args.max_frames,
                "feature_dim": int(x.shape[1]),
                "num_classes": num_classes,
                "train_count": int(train_mask.sum()),
                "val_count": int(val_mask.sum()),
                "test_count": int(test_mask.sum()),
                "use_weight": bool(not args.disable_weight),
                "use_gyro": bool(not args.disable_gyro),
            },
        }

        np.savez_compressed(
            out_dir / "model.npz",
            w=w,
            b=b,
            model_type=np.array(["linear_softmax"]),
            use_weight=np.array([int(not args.disable_weight)], dtype=np.int64),
            use_gyro=np.array([int(not args.disable_gyro)], dtype=np.int64),
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            num_classes=np.array([num_classes], dtype=np.int64),
        )
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (out_dir / "loss_curve.json").write_text(
            json.dumps({"loss": [float(v) for v in losses]}, indent=2), encoding="utf-8"
        )

        pred_logits = x_norm @ w + b
        pred = pred_logits.argmax(axis=1)
        pred_csv = out_dir / "predictions.csv"
        with pred_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "subset", "label_id", "pred_label_id"])
            writer.writeheader()
            for cid, subset, gt, pd in zip(clip_ids, subsets, y.tolist(), pred.tolist()):
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

        print("[DONE] Multimodal baseline training finished.")
        print(json.dumps(metrics, indent=2))
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
