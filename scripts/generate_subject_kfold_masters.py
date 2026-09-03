import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "splits_kfold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate subject-wise k-fold master_annotations files."
    )
    parser.add_argument("--master-json", default=str(DEFAULT_MASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation group ratio sampled from non-test groups in each fold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-level",
        choices=["none", "coarse", "fine", "both"],
        default="fine",
        help="Label level used for group-wise stratified fold assignment.",
    )
    parser.add_argument(
        "--subject-leakage-policy",
        choices=["warn", "fail"],
        default="fail",
        help="How to handle subject leakage across train/val/test in each fold.",
    )
    parser.add_argument(
        "--imbalance-policy",
        choices=["off", "warn", "fail"],
        default="warn",
        help="How to handle label imbalance across test folds.",
    )
    parser.add_argument(
        "--max-label-relative-deviation",
        type=float,
        default=0.8,
        help=(
            "Allowed max relative deviation from expected per-fold label count "
            "for imbalance check. Example: 0.8 means 80%% deviation tolerance."
        ),
    )
    parser.add_argument(
        "--imbalance-min-total",
        type=int,
        default=5,
        help="Ignore labels whose total test count across all folds is below this value.",
    )
    parser.add_argument(
        "--min-test-label-count-per-fold",
        type=int,
        default=0,
        help="Minimum required test count per label per fold (0 disables threshold).",
    )
    parser.add_argument(
        "--min-count-policy",
        choices=["off", "warn", "fail"],
        default="off",
        help="How to handle min-test-label-count-per-fold violations.",
    )
    return parser.parse_args()


def group_key(video_id: str, rec: Dict) -> str:
    subject_id = str(rec.get("subject_id", "")).strip().lower()
    if subject_id and subject_id not in {"unknown", "none", "null", "nan"}:
        return f"subject::{subject_id}"
    return f"video::{video_id}"


def build_groups(videos: Dict[str, Dict]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for vid, rec in videos.items():
        groups[group_key(vid, rec)].append(vid)
    return dict(groups)


def collect_group_label_counts(
    videos: Dict[str, Dict],
    groups: Dict[str, List[str]],
    stratify_level: str,
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for gk, vids in groups.items():
        label_counts: Dict[str, int] = defaultdict(int)
        for vid in vids:
            rec = videos.get(vid, {})
            anns = rec.get("annotations", [])
            if not isinstance(anns, list):
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                if stratify_level in {"coarse", "both"}:
                    coarse = str(ann.get("coarse_label", "")).strip()
                    if coarse:
                        label_counts[f"coarse::{coarse}"] += 1
                if stratify_level in {"fine", "both"}:
                    fine = str(ann.get("fine_label", "")).strip()
                    if fine:
                        label_counts[f"fine::{fine}"] += 1
        if not label_counts:
            label_counts["__NO_LABEL__"] = 1
        out[gk] = dict(label_counts)
    return out


def assign_fold_ids_random(group_keys: List[str], num_folds: int, seed: int) -> Dict[str, int]:
    keys = list(group_keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    out: Dict[str, int] = {}
    for idx, key in enumerate(keys):
        out[key] = idx % num_folds
    return out


def assign_fold_ids_stratified(
    group_keys: List[str],
    group_label_counts: Dict[str, Dict[str, int]],
    num_folds: int,
    seed: int,
) -> Dict[str, int]:
    rng = random.Random(seed)
    keys = list(group_keys)
    rng.shuffle(keys)
    keys.sort(
        key=lambda k: (
            -sum(group_label_counts.get(k, {}).values()),
            -len(group_label_counts.get(k, {})),
            str(k),
        )
    )
    if not keys:
        return {}

    global_label_totals: Dict[str, int] = defaultdict(int)
    total_weight = 0
    for key in keys:
        for label, c in group_label_counts.get(key, {}).items():
            global_label_totals[label] += int(c)
            total_weight += int(c)
    if total_weight <= 0:
        total_weight = len(keys)

    fold_label_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(num_folds)]
    fold_weights = [0 for _ in range(num_folds)]
    out: Dict[str, int] = {}

    for idx, key in enumerate(keys):
        counts = group_label_counts.get(key, {})
        if idx < num_folds:
            best_fold = idx
        else:
            best_fold = 0
            best_score = None
            for fold_idx in range(num_folds):
                score = 0.0
                for label, c in counts.items():
                    target = float(global_label_totals.get(label, 0)) / float(num_folds)
                    before = abs(float(fold_label_counts[fold_idx].get(label, 0)) - target)
                    after = abs(float(fold_label_counts[fold_idx].get(label, 0) + int(c)) - target)
                    score += (after - before)

                group_weight = int(sum(counts.values()))
                if group_weight <= 0:
                    group_weight = 1
                load_target = float(total_weight) / float(num_folds)
                load_before = abs(float(fold_weights[fold_idx]) - load_target)
                load_after = abs(float(fold_weights[fold_idx] + group_weight) - load_target)
                score += 0.1 * (load_after - load_before)

                if best_score is None or score < best_score - 1e-12:
                    best_score = score
                    best_fold = fold_idx
                elif best_score is not None and abs(score - best_score) <= 1e-12:
                    if fold_weights[fold_idx] < fold_weights[best_fold]:
                        best_fold = fold_idx

        out[key] = best_fold
        group_weight = int(sum(counts.values()))
        if group_weight <= 0:
            group_weight = 1
        fold_weights[best_fold] += group_weight
        for label, c in counts.items():
            fold_label_counts[best_fold][label] += int(c)

    return out


def primary_label_for_group(group_counts: Dict[str, int]) -> str:
    if not group_counts:
        return "__NO_LABEL__"
    return sorted(group_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]


def split_groups_for_fold_stratified(
    group_keys: List[str],
    fold_map: Dict[str, int],
    group_label_counts: Dict[str, Dict[str, int]],
    fold_idx: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    test_keys = [k for k in group_keys if fold_map[k] == fold_idx]
    remain = [k for k in group_keys if fold_map[k] != fold_idx]
    if not remain:
        return [], [], test_keys

    rng = random.Random(seed + fold_idx * 1009)
    rng.shuffle(remain)

    n_remain = len(remain)
    n_val = int(round(n_remain * val_ratio))
    if val_ratio > 0 and n_val == 0 and n_remain >= 2:
        n_val = 1
    while n_remain - n_val < 1:
        if n_val > 0:
            n_val -= 1
        else:
            break
    if n_val <= 0:
        return remain, [], test_keys

    buckets: Dict[str, List[str]] = defaultdict(list)
    for key in remain:
        buckets[primary_label_for_group(group_label_counts.get(key, {}))].append(key)
    labels = sorted(buckets.keys())
    total = float(len(remain))

    quota: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    assigned = 0
    for lb in labels:
        keys = buckets[lb]
        raw = (float(len(keys)) / total) * float(n_val)
        q = min(len(keys), int(raw))
        quota[lb] = q
        assigned += q
        remainders.append((raw - float(q), lb))

    remainders.sort(key=lambda x: (-x[0], x[1]))
    idx = 0
    while assigned < n_val and remainders:
        lb = remainders[idx % len(remainders)][1]
        if quota[lb] < len(buckets[lb]):
            quota[lb] += 1
            assigned += 1
        idx += 1
        if idx > len(remainders) * 4 and assigned < n_val:
            break

    val_keys: List[str] = []
    for lb in labels:
        keys = list(buckets[lb])
        rng.shuffle(keys)
        val_keys.extend(keys[: quota.get(lb, 0)])

    val_set = set(val_keys)
    train_keys = [k for k in remain if k not in val_set]
    return train_keys, val_keys, test_keys


def split_groups_for_fold(
    group_keys: List[str],
    fold_map: Dict[str, int],
    fold_idx: int,
    val_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    test_keys = [k for k in group_keys if fold_map[k] == fold_idx]
    remain = [k for k in group_keys if fold_map[k] != fold_idx]
    rng = random.Random(seed + fold_idx * 1009)
    rng.shuffle(remain)
    n_remain = len(remain)
    n_val = int(round(n_remain * val_ratio))
    if val_ratio > 0 and n_val == 0 and n_remain >= 2:
        n_val = 1
    while n_remain - n_val < 1:
        if n_val > 0:
            n_val -= 1
        else:
            break
    return remain[n_val:], remain[:n_val], test_keys


def normalize_subject_id(rec: Dict) -> str:
    sid = str(rec.get("subject_id", "")).strip().lower()
    if sid and sid not in {"unknown", "none", "null", "nan"}:
        return sid
    return ""


def detect_subject_leakage(
    videos: Dict[str, Dict],
    groups: Dict[str, List[str]],
    train_keys: List[str],
    val_keys: List[str],
    test_keys: List[str],
) -> List[Dict[str, object]]:
    key_to_subset: Dict[str, str] = {}
    for key in train_keys:
        key_to_subset[key] = "train"
    for key in val_keys:
        key_to_subset[key] = "val"
    for key in test_keys:
        key_to_subset[key] = "test"

    subject_subsets: Dict[str, set] = defaultdict(set)
    subject_videos: Dict[str, List[str]] = defaultdict(list)
    for key, vids in groups.items():
        subset = key_to_subset.get(key, "train")
        for vid in vids:
            sid = normalize_subject_id(videos.get(vid, {}))
            if not sid:
                continue
            subject_subsets[sid].add(subset)
            subject_videos[sid].append(vid)

    leaks: List[Dict[str, object]] = []
    for sid in sorted(subject_subsets.keys()):
        subsets = sorted(subject_subsets[sid])
        if len(subsets) <= 1:
            continue
        vids = sorted(subject_videos.get(sid, []))
        leaks.append(
            {
                "subject_id": sid,
                "subsets": subsets,
                "video_count": len(vids),
                "video_ids_sample": vids[:10],
            }
        )
    return leaks


def compute_test_label_imbalance(
    fold_stats: Dict[str, Dict],
    num_folds: int,
    min_total: int,
) -> Dict[str, object]:
    fold_names = sorted(fold_stats.keys())
    labels = sorted(
        {
            label
            for fold_name in fold_names
            for label in fold_stats.get(fold_name, {}).get("test_label_counts", {}).keys()
        }
    )

    per_label: Dict[str, Dict[str, object]] = {}
    worst_label = ""
    worst_dev = -1.0
    for label in labels:
        fold_counts = [int(fold_stats.get(f, {}).get("test_label_counts", {}).get(label, 0)) for f in fold_names]
        total = int(sum(fold_counts))
        if total < int(min_total):
            continue
        expected = float(total) / float(num_folds)
        if expected <= 0.0:
            continue
        rel_devs = [abs(float(c) - expected) / expected for c in fold_counts]
        max_dev = float(max(rel_devs)) if rel_devs else 0.0
        if max_dev > worst_dev:
            worst_dev = max_dev
            worst_label = label
        per_label[label] = {
            "total_test_count": total,
            "expected_per_fold": expected,
            "max_relative_deviation": max_dev,
            "fold_counts": {f: int(c) for f, c in zip(fold_names, fold_counts)},
        }

    fold_totals = {
        f: int(sum(fold_stats.get(f, {}).get("test_label_counts", {}).values()))
        for f in fold_names
    }
    return {
        "num_labels_checked": int(len(per_label)),
        "worst_label": worst_label,
        "worst_label_max_relative_deviation": float(max(worst_dev, 0.0)),
        "per_label": per_label,
        "fold_total_test_labels": fold_totals,
    }


def compute_min_test_count_violations(label_imbalance: Dict[str, object], min_count: int) -> Dict[str, object]:
    threshold = max(int(min_count), 0)
    if threshold <= 0:
        return {
            "threshold": 0,
            "num_violations": 0,
            "violations": [],
        }
    per_label = label_imbalance.get("per_label", {})
    if not isinstance(per_label, dict):
        return {
            "threshold": threshold,
            "num_violations": 0,
            "violations": [],
        }

    violations: List[Dict[str, object]] = []
    for label in sorted(per_label.keys()):
        payload = per_label.get(label, {})
        if not isinstance(payload, dict):
            continue
        fold_counts = payload.get("fold_counts", {})
        if not isinstance(fold_counts, dict):
            continue
        bad_folds = []
        for fold_name in sorted(fold_counts.keys()):
            c = int(fold_counts.get(fold_name, 0))
            if c < threshold:
                bad_folds.append({"fold": fold_name, "count": c})
        if bad_folds:
            violations.append(
                {
                    "label": label,
                    "bad_folds": bad_folds,
                }
            )
    return {
        "threshold": threshold,
        "num_violations": int(len(violations)),
        "violations": violations,
    }


def write_fold_master(
    payload: Dict,
    groups: Dict[str, List[str]],
    train_keys: List[str],
    val_keys: List[str],
    test_keys: List[str],
    fold_idx: int,
    num_folds: int,
    seed: int,
    val_ratio: float,
    output_dir: Path,
) -> Dict[str, int]:
    copied = copy.deepcopy(payload)
    videos = copied["videos"]
    train_set = set(train_keys)
    val_set = set(val_keys)
    test_set = set(test_keys)

    subset_counts = {"train": 0, "val": 0, "test": 0}
    for key, vids in groups.items():
        subset = "train"
        if key in test_set:
            subset = "test"
        elif key in val_set:
            subset = "val"
        for vid in vids:
            videos[vid]["subset"] = subset
            subset_counts[subset] += 1

    copied.setdefault("meta", {})
    copied["meta"]["split_mode"] = "subject_kfold"
    copied["meta"]["fold_index"] = fold_idx
    copied["meta"]["num_folds"] = num_folds
    copied["meta"]["kfold_seed"] = seed
    copied["meta"]["kfold_val_ratio"] = val_ratio

    fold_name = f"fold_{fold_idx:02d}"
    fold_dir = output_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "master_annotations.json").write_text(
        json.dumps(copied, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return subset_counts


def main() -> None:
    args = parse_args()
    src = Path(args.master_json)
    out_dir = Path(args.output_dir)
    if not src.exists():
        raise FileNotFoundError(src)
    if args.num_folds < 2:
        raise ValueError("--num-folds must be >= 2")

    payload = json.loads(src.read_text(encoding="utf-8"))
    videos = payload.get("videos", {})
    if not isinstance(videos, dict) or not videos:
        raise RuntimeError("master json has no videos")

    groups = build_groups(videos)
    group_keys = sorted(groups.keys())
    n_groups = len(group_keys)
    if args.num_folds > n_groups:
        raise ValueError(f"--num-folds ({args.num_folds}) > number of groups ({n_groups})")

    group_label_counts = collect_group_label_counts(videos, groups, args.stratify_level)
    quality_label_level = args.stratify_level if args.stratify_level != "none" else "fine"
    quality_label_counts = collect_group_label_counts(videos, groups, quality_label_level)
    if args.stratify_level == "none":
        fold_map = assign_fold_ids_random(group_keys, args.num_folds, args.seed)
    else:
        fold_map = assign_fold_ids_stratified(group_keys, group_label_counts, args.num_folds, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fold_stats = {}
    leakage_by_fold: Dict[str, List[Dict[str, object]]] = {}
    for fold_idx in range(args.num_folds):
        if args.stratify_level == "none":
            train_keys, val_keys, test_keys = split_groups_for_fold(
                group_keys=group_keys,
                fold_map=fold_map,
                fold_idx=fold_idx,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
        else:
            train_keys, val_keys, test_keys = split_groups_for_fold_stratified(
                group_keys=group_keys,
                fold_map=fold_map,
                group_label_counts=group_label_counts,
                fold_idx=fold_idx,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )

        leaks = detect_subject_leakage(
            videos=videos,
            groups=groups,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
        )
        if leaks:
            fold_name = f"fold_{fold_idx:02d}"
            leakage_by_fold[fold_name] = leaks
            print(f"[WARN] {fold_name}: detected {len(leaks)} subject leakage entries.")
            if args.subject_leakage_policy == "fail":
                raise RuntimeError(
                    f"{fold_name}: subject leakage detected; use --subject-leakage-policy warn to continue."
                )

        def count_labels(keys: List[str]) -> Dict[str, int]:
            counts: Dict[str, int] = defaultdict(int)
            for key in keys:
                for lb, c in quality_label_counts.get(key, {}).items():
                    counts[lb] += int(c)
            return dict(sorted(counts.items()))

        subset_counts = write_fold_master(
            payload=payload,
            groups=groups,
            train_keys=train_keys,
            val_keys=val_keys,
            test_keys=test_keys,
            fold_idx=fold_idx,
            num_folds=args.num_folds,
            seed=args.seed,
            val_ratio=args.val_ratio,
            output_dir=out_dir,
        )
        fold_stats[f"fold_{fold_idx:02d}"] = {
            "subset_video_counts": subset_counts,
            "num_train_groups": len(train_keys),
            "num_val_groups": len(val_keys),
            "num_test_groups": len(test_keys),
            "train_label_counts": count_labels(train_keys),
            "val_label_counts": count_labels(val_keys),
            "test_label_counts": count_labels(test_keys),
        }
        for key, vids in groups.items():
            subset = "train"
            if key in test_keys:
                subset = "test"
            elif key in val_keys:
                subset = "val"
            for vid in vids:
                rows.append(
                    {
                        "fold": fold_idx,
                        "group_key": key,
                        "video_id": vid,
                        "subject_id": videos[vid].get("subject_id", "unknown"),
                        "subset": subset,
                    }
                )

    label_imbalance = compute_test_label_imbalance(
        fold_stats=fold_stats,
        num_folds=args.num_folds,
        min_total=max(int(args.imbalance_min_total), 1),
    )
    worst_dev = float(label_imbalance.get("worst_label_max_relative_deviation", 0.0))
    imbalance_violation = (
        args.imbalance_policy in {"warn", "fail"}
        and int(label_imbalance.get("num_labels_checked", 0)) > 0
        and worst_dev > float(args.max_label_relative_deviation)
    )
    if imbalance_violation:
        msg = (
            "Label imbalance exceeds threshold: "
            f"worst_dev={worst_dev:.4f} > max_label_relative_deviation={float(args.max_label_relative_deviation):.4f} "
            f"(worst_label={label_imbalance.get('worst_label', '')})"
        )
        print(f"[WARN] {msg}")
        if args.imbalance_policy == "fail":
            raise RuntimeError(msg)

    min_count_check = compute_min_test_count_violations(
        label_imbalance=label_imbalance,
        min_count=int(args.min_test_label_count_per_fold),
    )
    min_count_violation = args.min_count_policy in {"warn", "fail"} and int(min_count_check.get("num_violations", 0)) > 0
    if min_count_violation:
        msg = (
            "Min test label count violation: "
            f"threshold={int(args.min_test_label_count_per_fold)}, "
            f"num_labels_violated={int(min_count_check.get('num_violations', 0))}"
        )
        print(f"[WARN] {msg}")
        if args.min_count_policy == "fail":
            raise RuntimeError(msg)

    with (out_dir / "fold_assignments.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "group_key", "video_id", "subject_id", "subset"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "master_json": str(src.resolve()),
        "output_dir": str(out_dir.resolve()),
        "num_videos": len(videos),
        "num_groups": n_groups,
        "num_folds": args.num_folds,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "stratify_level": args.stratify_level,
        "split_quality": {
            "subject_leakage_policy": args.subject_leakage_policy,
            "imbalance_policy": args.imbalance_policy,
            "max_label_relative_deviation": float(args.max_label_relative_deviation),
            "imbalance_min_total": int(args.imbalance_min_total),
            "label_quality_level": quality_label_level,
            "min_test_label_count_per_fold": int(args.min_test_label_count_per_fold),
            "min_count_policy": args.min_count_policy,
            "num_folds_with_subject_leakage": int(len(leakage_by_fold)),
            "subject_leakage_by_fold": leakage_by_fold,
            "label_imbalance": label_imbalance,
            "min_test_count_check": min_count_check,
        },
        "fold_stats": fold_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[DONE] subject-wise k-fold masters generated.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
