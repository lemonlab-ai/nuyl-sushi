import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT_DIR / "outputs" / "kfold_experiments" / "multimodal_baseline" / "experiment_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export paper-ready CSV/Markdown tables from k-fold summary and optional seed-sweep summary."
    )
    parser.add_argument("--experiment-summary", default="")
    parser.add_argument("--seed-sweep-summary", default="")
    parser.add_argument("--ablation-glob", default="fold_*/ablation_report.json")
    parser.add_argument("--ablation-subset", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-ablation", action="store_true", help="Do not include ablation table.")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()
    if not str(args.experiment_summary).strip() and not str(args.seed_sweep_summary).strip():
        args.experiment_summary = str(DEFAULT_SUMMARY)
    return args


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def get_stats(summary: Dict, task: str, metric: str) -> Dict[str, float]:
    item = summary.get(task, {}).get(metric, {})
    if not isinstance(item, dict):
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    out: Dict[str, float] = {}
    for key in ("count", "mean", "std", "min", "max"):
        value = item.get(key, 0.0)
        out[key] = float(value) if isinstance(value, (int, float)) else 0.0
    return out


def build_main_rows(summary: Dict) -> List[Dict[str, object]]:
    mapping = [
        ("task_a", "top1_accuracy"),
        ("task_a", "macro_f1"),
        ("task_a", "balanced_accuracy"),
        ("task_b", "average_mAP"),
        ("task_b", "average_Recall"),
    ]
    rows: List[Dict[str, object]] = []
    for task, metric in mapping:
        stats = get_stats(summary, task, metric)
        rows.append(
            {
                "section": "main",
                "setting": "kfold",
                "metric": f"{task}.{metric}",
                "count": int(stats["count"]),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "ci95_low": "",
                "ci95_high": "",
                "best_seed": "",
                "worst_seed": "",
            }
        )
    return rows


def build_ablation_rows(summary_path: Path, ablation_glob: str, subset: str) -> Tuple[List[Dict[str, object]], int]:
    fold_reports = sorted(summary_path.parent.glob(ablation_glob))
    bucket: Dict[Tuple[str, str], List[float]] = {}
    for report_path in fold_reports:
        payload = load_json(report_path)
        subset_payload = payload.get(subset, {})
        if not isinstance(subset_payload, dict):
            continue
        for scenario, vals in subset_payload.items():
            if not isinstance(vals, dict):
                continue
            for metric, value in vals.items():
                if not isinstance(value, (int, float)):
                    continue
                key = (str(scenario), str(metric))
                bucket.setdefault(key, []).append(float(value))

    rows: List[Dict[str, object]] = []
    for (scenario, metric), values in sorted(bucket.items()):
        stats = summarize(values)
        rows.append(
            {
                "section": "ablation",
                "setting": scenario,
                "metric": metric,
                "count": int(stats["count"]),
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "ci95_low": "",
                "ci95_high": "",
                "best_seed": "",
                "worst_seed": "",
            }
        )
    return rows, len(fold_reports)


def build_seed_sweep_rows(seed_summary: Dict) -> List[Dict[str, object]]:
    aggregate = seed_summary.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return []
    rows: List[Dict[str, object]] = []
    for metric, stats in sorted(aggregate.items()):
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "section": "seed_sweep",
                "setting": "multi_seed",
                "metric": str(metric),
                "count": int(float(stats.get("count", 0.0))),
                "mean": float(stats.get("mean", 0.0)),
                "std": float(stats.get("std", 0.0)),
                "min": float(stats.get("min", 0.0)),
                "max": float(stats.get("max", 0.0)),
                "ci95_low": float(stats.get("ci95_low", 0.0)),
                "ci95_high": float(stats.get("ci95_high", 0.0)),
                "best_seed": int(float(stats.get("best_seed", -1.0))),
                "worst_seed": int(float(stats.get("worst_seed", -1.0))),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "section",
        "setting",
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "max",
        "ci95_low",
        "ci95_high",
        "best_seed",
        "worst_seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_num(v: object) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    s = str(v).strip()
    return s


def build_markdown(rows: List[Dict[str, object]]) -> str:
    section_titles = {
        "main": "Main Results",
        "ablation": "Ablation Results",
        "seed_sweep": "Seed Sweep Results",
    }
    lines: List[str] = []
    for section in ("main", "ablation", "seed_sweep"):
        sec_rows = [r for r in rows if r.get("section") == section]
        if not sec_rows:
            continue
        lines.append(f"## {section_titles.get(section, section)}")
        lines.append("| Setting | Metric | Mean | Std | 95% CI | Min | Max | Count | BestSeed | WorstSeed |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sec_rows:
            ci = ""
            if str(r.get("ci95_low", "")).strip() != "" and str(r.get("ci95_high", "")).strip() != "":
                ci = f"[{fmt_num(r.get('ci95_low'))}, {fmt_num(r.get('ci95_high'))}]"
            lines.append(
                "| {setting} | {metric} | {mean} | {std} | {ci} | {minv} | {maxv} | {count} | {best} | {worst} |".format(
                    setting=str(r.get("setting", "")),
                    metric=str(r.get("metric", "")),
                    mean=fmt_num(r.get("mean", "")),
                    std=fmt_num(r.get("std", "")),
                    ci=ci,
                    minv=fmt_num(r.get("min", "")),
                    maxv=fmt_num(r.get("max", "")),
                    count=str(r.get("count", "")),
                    best=str(r.get("best_seed", "")),
                    worst=str(r.get("worst_seed", "")),
                )
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()

    rows: List[Dict[str, object]] = []
    num_ablation_reports = 0
    summary_path: Optional[Path] = None
    seed_summary_path: Optional[Path] = None

    if str(args.experiment_summary).strip():
        summary_path = Path(args.experiment_summary)
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = load_json(summary_path)
        rows.extend(build_main_rows(summary))
        if not args.no_ablation:
            ablation_rows, num_ablation_reports = build_ablation_rows(
                summary_path=summary_path,
                ablation_glob=args.ablation_glob,
                subset=args.ablation_subset,
            )
            rows.extend(ablation_rows)

    if str(args.seed_sweep_summary).strip():
        seed_summary_path = Path(args.seed_sweep_summary)
        if not seed_summary_path.exists():
            raise FileNotFoundError(seed_summary_path)
        seed_summary = load_json(seed_summary_path)
        rows.extend(build_seed_sweep_rows(seed_summary))

    if not rows:
        raise RuntimeError("No rows generated. Provide --experiment-summary and/or --seed-sweep-summary.")

    if summary_path is not None:
        default_base = summary_path.parent
    elif seed_summary_path is not None:
        default_base = seed_summary_path.parent
    else:
        default_base = ROOT_DIR / "outputs"

    default_csv = default_base / "paper_table.csv"
    default_md = default_base / "paper_table.md"
    out_csv = Path(args.output_csv) if args.output_csv else default_csv
    out_md = Path(args.output_md) if args.output_md else default_md

    write_csv(out_csv, rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(build_markdown(rows), encoding="utf-8")

    payload = {
        "experiment_summary": str(summary_path.resolve()) if summary_path is not None else "",
        "seed_sweep_summary": str(seed_summary_path.resolve()) if seed_summary_path is not None else "",
        "num_rows": int(len(rows)),
        "num_ablation_reports_detected": int(num_ablation_reports),
        "output_csv": str(out_csv.resolve()),
        "output_md": str(out_md.resolve()),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
