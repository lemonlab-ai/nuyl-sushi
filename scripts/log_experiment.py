import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_CSV = ROOT_DIR / "outputs" / "results.csv"
FIELDNAMES = [
    "timestamp_utc",
    "experiment_id",
    "task",
    "model",
    "repo",
    "config",
    "split",
    "seed",
    "commit",
    "status",
    "metrics_json",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one experiment record to results CSV.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--task", required=True, help="task_a or task_b")
    parser.add_argument("--model", required=True, help="e.g., videomaev2 / actionformer")
    parser.add_argument("--repo", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--status", choices=["ok", "fail", "skip"], default="ok")
    parser.add_argument("--notes", default="")
    parser.add_argument("--results-csv", default=str(DEFAULT_RESULTS_CSV))
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        help='Metric entry in "key=value" format. Can be passed multiple times.',
    )
    parser.add_argument("--metrics-json-file", default="", help="Optional JSON file containing metrics object.")
    return parser.parse_args()


def parse_metrics(metric_args: List[str], json_file: str) -> Dict:
    metrics = {}
    if json_file:
        metrics = json.loads(Path(json_file).read_text(encoding="utf-8"))
    for item in metric_args:
        if "=" not in item:
            raise ValueError(f"Invalid metric '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            if "." in value:
                value_cast = float(value)
            else:
                value_cast = int(value)
        except ValueError:
            value_cast = value
        metrics[key] = value_cast
    return metrics


def ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.results_csv)
    ensure_csv(csv_path)

    metrics = parse_metrics(args.metric, args.metrics_json_file)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "task": args.task,
        "model": args.model,
        "repo": args.repo,
        "config": args.config,
        "split": args.split,
        "seed": args.seed,
        "commit": args.commit,
        "status": args.status,
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
        "notes": args.notes,
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
    print(f"[DONE] Appended experiment record to {csv_path}")


if __name__ == "__main__":
    main()
