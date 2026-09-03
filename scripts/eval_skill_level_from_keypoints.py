import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "skill_level"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from apps.gesture_coach.coach import _frame_to_vec, load_sequence_json
from apps.gesture_coach.mano_adapter import ensure_mano_sequence

SKILL_LEVEL_ALIASES = {
    "beginner": "beginner",
    "novice": "beginner",
    "newbie": "beginner",
    "entry": "beginner",
    "entry_level": "beginner",
    "junior": "beginner",
    "intermediate": "intermediate",
    "advanced": "intermediate",
    "mid": "intermediate",
    "mid_level": "intermediate",
    "expert": "expert",
    "senior": "expert",
    "master": "expert",
    "pro": "expert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate skill-level classification from gesture keypoint sequences "
            "(2D / 3D image / 3D world / 3D MANO-compatible)."
        )
    )
    parser.add_argument("--index-path", required=True, help="CSV/JSON index with sequence_path + skill_level.")
    parser.add_argument(
        "--coord-modes",
        default="2d,3d_image,3d_world,3d_mano",
        help="Comma-separated coordinate modes.",
    )
    parser.add_argument("--hand-mode", choices=["left", "right", "both"], default="both")
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument(
        "--allow-mano-proxy",
        action="store_true",
        help="Allow world_3d fallback when coord_mode=3d_mano and mano joints are missing.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def normalize_skill_level(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    canon = text.replace("-", "_").replace(" ", "_")
    return SKILL_LEVEL_ALIASES.get(canon, canon)


def parse_coord_modes(raw: str) -> List[str]:
    out = []
    for token in str(raw).split(","):
        mode = token.strip().lower()
        if not mode:
            continue
        if mode not in {"2d", "3d_image", "3d_world", "3d_mano"}:
            raise ValueError(f"Unsupported coord mode: {mode}")
        out.append(mode)
    if not out:
        raise ValueError("--coord-modes is empty")
    return out


def load_index(index_path: Path) -> List[Dict[str, Any]]:
    suffix = index_path.suffix.lower()
    if suffix == ".csv":
        with index_path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return [dict(x) for x in payload["samples"]]
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, dict)]
    raise RuntimeError("Index JSON must be a list or a dict with key 'samples'.")


def prepare_samples(raw_rows: List[Dict[str, Any]], index_path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        status = str(row.get("status", "")).strip().lower()
        if status and status not in {"ok", "skipped_existing"}:
            continue
        seq_raw = str(row.get("sequence_path", "")).strip()
        if not seq_raw:
            continue
        seq_path = Path(seq_raw)
        if not seq_path.is_absolute():
            seq_path = (index_path.parent / seq_path).resolve()
        if not seq_path.exists():
            continue
        skill_level = normalize_skill_level(row.get("skill_level", "unknown"))
        if skill_level == "unknown":
            continue
        sample_id = str(row.get("sample_id", "")).strip() or f"s{i:05d}"
        subject_id = str(row.get("subject_id", "")).strip() or f"sample::{sample_id}"
        out.append(
            {
                "sample_id": sample_id,
                "subject_id": subject_id,
                "skill_level": skill_level,
                "sequence_path": str(seq_path),
            }
        )
    return out


def hand_mode_flags(hand_mode: str) -> Tuple[bool, bool]:
    mode = str(hand_mode).strip().lower()
    return (mode in {"left", "both"}, mode in {"right", "both"})


def extract_feature(
    seq: Dict[str, Any],
    coord_mode: str,
    use_left: bool,
    use_right: bool,
    max_frames: int,
) -> np.ndarray:
    frames = seq.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    if max_frames > 0:
        frames = frames[: int(max_frames)]

    template_vec = _frame_to_vec({}, use_left=use_left, use_right=use_right, coord_mode=coord_mode)
    dim = int(template_vec.shape[0])

    if not frames:
        arr = np.zeros((1, dim), dtype=np.float32)
    else:
        vecs = [_frame_to_vec(frame, use_left=use_left, use_right=use_right, coord_mode=coord_mode) for frame in frames]
        arr = np.stack(vecs, axis=0).astype(np.float32)

    feat_mean = np.mean(arr, axis=0)
    feat_std = np.std(arr, axis=0)
    if arr.shape[0] > 1:
        feat_motion = np.mean(np.abs(np.diff(arr, axis=0)), axis=0)
    else:
        feat_motion = np.zeros_like(feat_mean)
    feat_presence = np.mean((np.abs(arr) > 1e-8).astype(np.float32), axis=0)
    return np.concatenate([feat_mean, feat_std, feat_motion, feat_presence], axis=0).astype(np.float32)


def build_group_folds(
    samples: List[Dict[str, Any]],
    labels: np.ndarray,
    num_folds: int,
    seed: int,
    num_classes: int,
) -> List[List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        groups[str(sample["subject_id"])].append(idx)

    group_keys = sorted(groups.keys())
    if len(group_keys) < 2:
        raise RuntimeError("Need at least 2 distinct subject_id groups for cross-validation.")
    k = min(int(num_folds), len(group_keys))
    if k < 2:
        raise RuntimeError("num_folds must be >= 2")

    # Group-wise stratification: assign each subject-group to a fold while
    # minimizing label-count deviation from expected per-fold targets.
    group_label_counts: Dict[str, Dict[int, int]] = {}
    global_label_totals: Dict[int, int] = defaultdict(int)
    for gk in group_keys:
        counts: Dict[int, int] = defaultdict(int)
        for idx in groups[gk]:
            lb = int(labels[idx])
            counts[lb] += 1
            global_label_totals[lb] += 1
        group_label_counts[gk] = dict(counts)

    rng = random.Random(seed)
    shuffled_keys = list(group_keys)
    rng.shuffle(shuffled_keys)
    shuffled_keys.sort(
        key=lambda gk: (
            -sum(group_label_counts[gk].values()),
            -len(group_label_counts[gk]),
            str(gk),
        )
    )

    fold_map: Dict[str, int] = {}
    fold_label_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(k)]
    fold_weights = [0 for _ in range(k)]
    total_weight = max(1, int(len(samples)))

    for i, gk in enumerate(shuffled_keys):
        counts = group_label_counts[gk]
        if i < k:
            best_fold = i
        else:
            best_fold = 0
            best_score = None
            for fold_idx in range(k):
                score = 0.0
                for cid in range(num_classes):
                    target = float(global_label_totals.get(cid, 0)) / float(k)
                    before = abs(float(fold_label_counts[fold_idx].get(cid, 0)) - target)
                    after = abs(float(fold_label_counts[fold_idx].get(cid, 0) + counts.get(cid, 0)) - target)
                    score += (after - before)

                group_weight = int(sum(counts.values()))
                load_target = float(total_weight) / float(k)
                load_before = abs(float(fold_weights[fold_idx]) - load_target)
                load_after = abs(float(fold_weights[fold_idx] + group_weight) - load_target)
                score += 0.1 * (load_after - load_before)

                if best_score is None or score < best_score - 1e-12:
                    best_score = score
                    best_fold = fold_idx
                elif best_score is not None and abs(score - best_score) <= 1e-12:
                    if fold_weights[fold_idx] < fold_weights[best_fold]:
                        best_fold = fold_idx

        fold_map[gk] = best_fold
        fold_weights[best_fold] += int(sum(counts.values()))
        for cid, c in counts.items():
            fold_label_counts[best_fold][cid] += int(c)

    folds: List[List[int]] = [[] for _ in range(k)]
    for gk, fold_idx in fold_map.items():
        folds[fold_idx].extend(groups[gk])
    return folds


def standardize_train_test(train_x: np.ndarray, test_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(train_x, axis=0, keepdims=True)
    sigma = np.std(train_x, axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def fit_centroids(train_x: np.ndarray, train_y: np.ndarray, class_ids: List[int]) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for cid in class_ids:
        mask = train_y == cid
        if not np.any(mask):
            continue
        centroids[cid] = np.mean(train_x[mask], axis=0)
    return centroids


def predict_centroid(test_x: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    classes = sorted(centroids.keys())
    if not classes:
        raise RuntimeError("No class centroid available for prediction.")
    dmat = []
    for cid in classes:
        c = centroids[cid][None, :]
        d = np.sum((test_x - c) ** 2, axis=1)
        dmat.append(d)
    all_dist = np.stack(dmat, axis=1)
    best = np.argmin(all_dist, axis=1)
    return np.asarray([classes[int(i)] for i in best], dtype=np.int64)


def compute_metrics(
    true_y: np.ndarray,
    pred_y: np.ndarray,
    class_ids: List[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = int(true_y.shape[0])
    out["n"] = n
    out["accuracy"] = float(np.mean((true_y == pred_y).astype(np.float32))) if n > 0 else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    recalls = []
    f1s = []
    for cid in class_ids:
        tp = int(np.sum((true_y == cid) & (pred_y == cid)))
        fp = int(np.sum((true_y != cid) & (pred_y == cid)))
        fn = int(np.sum((true_y == cid) & (pred_y != cid)))
        support = int(np.sum(true_y == cid))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall <= 1e-12:
            f1 = 0.0
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))
        per_class[str(cid)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
        if support > 0:
            recalls.append(recall)
            f1s.append(f1)

    out["balanced_accuracy"] = float(np.mean(recalls)) if recalls else 0.0
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    out["per_class"] = per_class
    return out


def mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def main() -> None:
    args = parse_args()
    index_path = Path(args.index_path)
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    coord_modes = parse_coord_modes(args.coord_modes)
    raw_rows = load_index(index_path)
    samples = prepare_samples(raw_rows, index_path)
    if len(samples) < 4:
        raise RuntimeError("Need at least 4 valid samples after filtering unknown skill levels.")

    label_names = sorted({str(s["skill_level"]) for s in samples})
    if len(label_names) < 2:
        raise RuntimeError("Need at least 2 skill_level classes.")
    label_to_id = {name: i for i, name in enumerate(label_names)}
    y_all = np.asarray([label_to_id[str(s["skill_level"])] for s in samples], dtype=np.int64)

    folds = build_group_folds(
        samples,
        labels=y_all,
        num_folds=args.num_folds,
        seed=args.seed,
        num_classes=len(label_names),
    )
    use_left, use_right = hand_mode_flags(args.hand_mode)

    fold_label_distributions: List[Dict[str, int]] = []
    fold_coverage_warnings: List[str] = []
    for fold_id, test_indices in enumerate(folds):
        counts: Dict[str, int] = {}
        for lb_name, lb_id in label_to_id.items():
            c = int(np.sum(y_all[test_indices] == lb_id)) if test_indices else 0
            counts[lb_name] = c
            if c == 0:
                fold_coverage_warnings.append(
                    f"fold={fold_id} has zero test samples for class={lb_name}"
                )
        fold_label_distributions.append(counts)
    if fold_coverage_warnings:
        print("[WARN] Imperfect label coverage across folds:")
        for msg in fold_coverage_warnings:
            print(f"  - {msg}")

    seq_cache: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        p = str(sample["sequence_path"])
        if p in seq_cache:
            continue
        seq_cache[p] = load_sequence_json(Path(p))

    result_modes: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for coord_mode in coord_modes:
        features: List[np.ndarray] = []
        for sample in samples:
            seq = seq_cache[str(sample["sequence_path"])]
            seq_use = seq
            if coord_mode == "3d_mano":
                seq_use = ensure_mano_sequence(seq_use, allow_world_fallback=bool(args.allow_mano_proxy))
            feat = extract_feature(
                seq=seq_use,
                coord_mode=coord_mode,
                use_left=use_left,
                use_right=use_right,
                max_frames=int(args.max_frames),
            )
            features.append(feat)
        x_all = np.stack(features, axis=0).astype(np.float32)

        fold_metrics = []
        for fold_id, test_indices in enumerate(folds):
            test_set = set(test_indices)
            train_indices = [i for i in range(len(samples)) if i not in test_set]
            if not train_indices or not test_indices:
                continue

            x_train = x_all[train_indices]
            y_train = y_all[train_indices]
            x_test = x_all[test_indices]
            y_test = y_all[test_indices]

            x_train_norm, x_test_norm = standardize_train_test(x_train, x_test)
            centroids = fit_centroids(x_train_norm, y_train, class_ids=list(range(len(label_names))))
            pred = predict_centroid(x_test_norm, centroids)
            metric = compute_metrics(y_test, pred, class_ids=list(range(len(label_names))))
            metric["fold"] = fold_id
            metric["num_train"] = int(len(train_indices))
            metric["num_test"] = int(len(test_indices))
            fold_metrics.append(metric)

        if not fold_metrics:
            raise RuntimeError(f"No valid fold result for coord_mode={coord_mode}")

        acc_stats = mean_std([float(m["accuracy"]) for m in fold_metrics])
        f1_stats = mean_std([float(m["macro_f1"]) for m in fold_metrics])
        bal_stats = mean_std([float(m["balanced_accuracy"]) for m in fold_metrics])

        result_modes[coord_mode] = {
            "accuracy": acc_stats,
            "macro_f1": f1_stats,
            "balanced_accuracy": bal_stats,
            "folds": fold_metrics,
        }

        csv_rows.append(
            {
                "coord_mode": coord_mode,
                "accuracy_mean": acc_stats["mean"],
                "accuracy_std": acc_stats["std"],
                "macro_f1_mean": f1_stats["mean"],
                "macro_f1_std": f1_stats["std"],
                "balanced_accuracy_mean": bal_stats["mean"],
                "balanced_accuracy_std": bal_stats["std"],
            }
        )

    summary = {
        "index_path": str(index_path.resolve()),
        "num_samples": int(len(samples)),
        "num_classes": int(len(label_names)),
        "labels": label_names,
        "coord_modes": coord_modes,
        "hand_mode": args.hand_mode,
        "num_folds": int(len(folds)),
        "fold_test_label_distributions": fold_label_distributions,
        "fold_coverage_warnings": fold_coverage_warnings,
        "seed": int(args.seed),
        "max_frames": int(args.max_frames),
        "allow_mano_proxy": bool(args.allow_mano_proxy),
        "classifier": "nearest_centroid",
        "feature_summary": "concat(mean, std, mean_abs_velocity, presence_ratio) on normalized keypoint vectors",
        "results_by_mode": result_modes,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "skill_level_keypoint_benchmark.json"
    out_csv = output_dir / "skill_level_keypoint_benchmark.csv"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "coord_mode",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "balanced_accuracy_mean",
                "balanced_accuracy_std",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    best_mode = None
    best_f1 = -math.inf
    for row in csv_rows:
        v = float(row["macro_f1_mean"])
        if v > best_f1:
            best_f1 = v
            best_mode = str(row["coord_mode"])

    print(f"[DONE] skill-level benchmark saved to: {out_json}")
    print(f"[DONE] summary table saved to: {out_csv}")
    print(f"Best mode by macro_f1_mean: {best_mode} ({best_f1:.4f})")


if __name__ == "__main__":
    main()
