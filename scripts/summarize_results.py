import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = ROOT_DIR / "outputs" / "results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment records from results CSV.")
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument("--task", default="all")
    parser.add_argument("--split", default="all")
    parser.add_argument("--metric-key", default="", help="Optional metric key to focus on.")
    parser.add_argument("--output-json", default="", help="Optional output JSON summary path.")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def try_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except Exception:
        return None


def extract_numeric_metrics(rows: List[Dict[str, str]], metric_key: str) -> Dict[Tuple[str, str, str], List[float]]:
    grouped = defaultdict(list)
    for r in rows:
        if r.get("status") != "ok":
            continue
        task = r.get("task", "")
        model = r.get("model", "")
        split = r.get("split", "")
        metrics = {}
        try:
            metrics = json.loads(r.get("metrics_json", "{}"))
        except Exception:
            metrics = {}

        if metric_key:
            value = try_float(metrics.get(metric_key))
            if value is not None:
                grouped[(task, model, metric_key)].append(value)
            continue

        for k, v in metrics.items():
            value = try_float(v)
            if value is not None and math.isfinite(value):
                grouped[(task, model, k)].append(value)
    return grouped


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.results_csv))
    if args.task != "all":
        rows = [r for r in rows if r.get("task") == args.task]
    if args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]

    grouped = extract_numeric_metrics(rows, args.metric_key)
    summary = []
    for (task, model, metric), values in sorted(grouped.items()):
        if not values:
            continue
        summary.append(
            {
                "task": task,
                "model": model,
                "metric": metric,
                "count": len(values),
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "best": max(values),
                "worst": min(values),
            }
        )

    payload = {
        "num_records_filtered": len(rows),
        "num_summary_rows": len(summary),
        "summary": summary,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
