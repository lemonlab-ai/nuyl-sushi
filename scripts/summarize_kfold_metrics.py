import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = ROOT_DIR / "artifacts" / "splits_kfold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize k-fold metrics (mean/std) from fold metric JSON files.")
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--fold-pattern", default="fold_*")
    parser.add_argument("--task", choices=["task_a", "task_b", "all"], default="all")
    parser.add_argument("--task-a-metrics-rel", default="task_a/metrics.json")
    parser.add_argument("--task-b-metrics-rel", default="task_b/metrics.json")
    parser.add_argument(
        "--include-detailed",
        action="store_true",
        help="Include detailed per-class metric keys (can be very verbose).",
    )
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--output-csv", default="", help="Optional output CSV path.")
    return parser.parse_args()


def flatten_numeric_dict(payload: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in payload.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_numeric_dict(v, key))
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            fv = float(v)
            if math.isfinite(fv):
                out[key] = fv
    return out


def filter_metrics(metric_map: Dict[str, float], include_detailed: bool) -> Dict[str, float]:
    if include_detailed:
        return metric_map
    out = {}
    for k, v in metric_map.items():
        if k.startswith("per_class_metrics."):
            continue
        if k.startswith("per_class_support."):
            continue
        if k.startswith("per_class_AP_by_IoU."):
            continue
        out[k] = v
    return out


def collect_fold_dirs(split_root: Path, pattern: str) -> List[Path]:
    return sorted([p for p in split_root.glob(pattern) if p.is_dir()])


def aggregate_metrics(rows: List[Tuple[str, float]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = {}
    for metric, value in rows:
        grouped.setdefault(metric, []).append(float(value))

    out: Dict[str, Dict[str, float]] = {}
    for metric, values in sorted(grouped.items()):
        out[metric] = {
            "count": float(len(values)),
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return out


def build_task_summary(fold_dirs: List[Path], rel_path: str, include_detailed: bool) -> Dict[str, Any]:
    per_fold: Dict[str, Dict[str, float]] = {}
    flat_rows: List[Tuple[str, float]] = []
    missing_folds: List[str] = []

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        metric_path = fold_dir / Path(rel_path)
        if not metric_path.exists():
            missing_folds.append(fold_name)
            continue

        payload = json.loads(metric_path.read_text(encoding="utf-8"))
        flat = flatten_numeric_dict(payload)
        flat = filter_metrics(flat, include_detailed=include_detailed)
        per_fold[fold_name] = flat
        for k, v in flat.items():
            flat_rows.append((k, v))

    return {
        "num_folds_with_metrics": len(per_fold),
        "missing_folds": missing_folds,
        "per_fold_metrics": per_fold,
        "aggregate": aggregate_metrics(flat_rows),
    }


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = []
    for task_name, task_data in payload.get("tasks", {}).items():
        aggregate = task_data.get("aggregate", {})
        for metric, stats in aggregate.items():
            rows.append(
                {
                    "task": task_name,
                    "metric": metric,
                    "count": int(stats["count"]),
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "metric", "count", "mean", "std", "min", "max"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    if not split_root.exists():
        raise FileNotFoundError(split_root)

    fold_dirs = collect_fold_dirs(split_root, args.fold_pattern)
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found at {split_root} with pattern {args.fold_pattern}")

    tasks: Dict[str, Any] = {}
    if args.task in {"task_a", "all"}:
        tasks["task_a"] = build_task_summary(
            fold_dirs,
            args.task_a_metrics_rel,
            include_detailed=args.include_detailed,
        )
    if args.task in {"task_b", "all"}:
        tasks["task_b"] = build_task_summary(
            fold_dirs,
            args.task_b_metrics_rel,
            include_detailed=args.include_detailed,
        )

    payload = {
        "split_root": str(split_root.resolve()),
        "fold_pattern": args.fold_pattern,
        "num_fold_dirs": len(fold_dirs),
        "folds": [p.name for p in fold_dirs],
        "include_detailed": bool(args.include_detailed),
        "tasks": tasks,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_csv:
        write_csv(Path(args.output_csv), payload)


if __name__ == "__main__":
    main()
