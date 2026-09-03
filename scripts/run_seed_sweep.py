import argparse
import csv
import datetime as dt
import json
import math
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "seed_sweeps"


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run k-fold experiments across multiple seeds and aggregate results."
    )
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seed list.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-name", default="multimodal_baseline")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--summary-name", default="seed_sweep_summary.json")
    parser.add_argument("--csv-name", default="seed_sweep_summary.csv")
    parser.add_argument("--aggregate-csv-name", default="seed_sweep_aggregate.csv")
    parser.add_argument("--export-paper-table", action="store_true")
    parser.add_argument("--paper-table-csv", default="")
    parser.add_argument("--paper-table-md", default="")
    parser.add_argument(
        "--per-run-timeout-sec",
        type=int,
        default=0,
        help="Timeout in seconds for each per-seed k-fold run (0 disables timeout).",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def parse_seed_list(text: str) -> List[int]:
    out: List[int] = []
    for raw in str(text).split(","):
        token = raw.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No valid seeds found in --seeds.")
    return out


def remove_conflicting_args(argv: List[str]) -> List[str]:
    strip_with_value = {"--seed", "--output-root", "--model-name"}
    out: List[str] = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in strip_with_value:
            skip_next = True
            continue
        if any(tok.startswith(flag + "=") for flag in strip_with_value):
            continue
        out.append(tok)
    return out


def terminate_process_tree(proc: subprocess.Popen, grace_sec: float = 3.0) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
        try:
            proc.wait(timeout=max(float(grace_sec), 0.1))
        except Exception:
            pass
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=max(float(grace_sec), 0.1))
        return
    except Exception:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=max(float(grace_sec), 0.1))
    except Exception:
        pass


def run_cmd(cmd: List[str], cwd: Path, timeout_sec: int = 0) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    popen_kwargs: Dict[str, object] = {"cwd": cwd}
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP"))
    elif os.name != "nt":
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        if int(timeout_sec) > 0:
            proc.wait(timeout=float(timeout_sec))
        else:
            proc.wait()
    except subprocess.TimeoutExpired as exc:
        terminate_process_tree(proc)
        raise TimeoutError(f"Command timed out after {int(timeout_sec)}s: {' '.join(cmd)}") from exc

    rc = int(proc.returncode or 0)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_seed_metrics(summary: Dict) -> Dict[str, float]:
    def get_mean(task: str, metric: str) -> float:
        d = summary.get(task, {}).get(metric, {})
        v = d.get("mean", 0.0) if isinstance(d, dict) else 0.0
        return float(v) if isinstance(v, (int, float)) else 0.0

    return {
        "task_a_top1_mean": get_mean("task_a", "top1_accuracy"),
        "task_a_macro_f1_mean": get_mean("task_a", "macro_f1"),
        "task_a_balanced_accuracy_mean": get_mean("task_a", "balanced_accuracy"),
        "task_b_average_mAP_mean": get_mean("task_b", "average_mAP"),
        "task_b_average_Recall_mean": get_mean("task_b", "average_Recall"),
    }


def summarize_with_ci(seed_values: List[Tuple[int, float]]) -> Dict[str, float]:
    if not seed_values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "best_seed": -1.0,
            "worst_seed": -1.0,
        }
    values = [float(v) for _, v in seed_values]
    n = len(values)
    avg = float(mean(values))
    std = float(pstdev(values)) if n > 1 else 0.0
    half_width = 1.96 * (std / math.sqrt(float(n))) if n > 1 else 0.0
    ordered = sorted(seed_values, key=lambda x: (float(x[1]), -int(x[0])))
    worst_seed = int(ordered[0][0])
    best_seed = int(ordered[-1][0])
    return {
        "count": float(n),
        "mean": avg,
        "std": std,
        "min": float(min(values)),
        "max": float(max(values)),
        "ci95_low": float(avg - half_width),
        "ci95_high": float(avg + half_width),
        "best_seed": float(best_seed),
        "worst_seed": float(worst_seed),
    }


def choose_primary_metric(aggregate: Dict[str, Dict[str, float]]) -> str:
    priority = ["task_a_top1_mean", "task_b_average_mAP_mean", "task_a_macro_f1_mean"]
    for key in priority:
        if key in aggregate and float(aggregate[key].get("count", 0.0)) > 0:
            return key
    for key, stats in aggregate.items():
        if float(stats.get("count", 0.0)) > 0:
            return key
    return ""


def write_per_seed_csv(path: Path, per_seed_rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "seed",
        "status",
        "summary_json",
        "task_a_top1_mean",
        "task_a_macro_f1_mean",
        "task_a_balanced_accuracy_mean",
        "task_b_average_mAP_mean",
        "task_b_average_Recall_mean",
        "error_type",
        "error_message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_aggregate_csv(path: Path, aggregate: Dict[str, Dict[str, float]]) -> None:
    fieldnames = [
        "metric",
        "count",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
        "best_seed",
        "worst_seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric, stats in sorted(aggregate.items()):
            row = {"metric": metric}
            for k in fieldnames[1:]:
                row[k] = stats.get(k, "")
            writer.writerow(row)


def main() -> None:
    args, passthrough = parse_args()
    seeds = parse_seed_list(args.seeds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_passthrough = remove_conflicting_args(passthrough)

    per_seed_rows: List[Dict[str, object]] = []
    aggregate_pool: Dict[str, List[Tuple[int, float]]] = {}
    failed_rows: List[Dict[str, object]] = []

    for seed in seeds:
        seed_output_root = output_root / f"seed_{seed}"
        cmd = [
            args.python,
            "scripts/run_kfold_experiments.py",
            "--model-name",
            str(args.model_name),
            "--seed",
            str(seed),
            "--output-root",
            str(seed_output_root),
        ] + clean_passthrough

        row: Dict[str, object] = {
            "seed": int(seed),
            "status": "ok",
            "summary_json": "",
            "error_type": "",
            "error_message": "",
        }
        try:
            run_cmd(cmd, ROOT_DIR, timeout_sec=int(args.per_run_timeout_sec))
            summary_json = seed_output_root / args.model_name / "experiment_summary.json"
            if not summary_json.exists():
                raise FileNotFoundError(summary_json)
            row["summary_json"] = str(summary_json.resolve())
            metrics = extract_seed_metrics(load_json(summary_json))
            for k, v in metrics.items():
                val = float(v)
                row[k] = val
                aggregate_pool.setdefault(k, []).append((int(seed), val))
        except Exception as exc:
            row["status"] = f"error:{exc.__class__.__name__}"
            row["error_type"] = str(exc.__class__.__name__)
            row["error_message"] = str(exc)
            failed_rows.append(
                {
                    "seed": int(seed),
                    "error_type": str(exc.__class__.__name__),
                    "error_message": str(exc),
                }
            )
            if not args.continue_on_error:
                per_seed_rows.append(row)
                break
        per_seed_rows.append(row)

    aggregate = {
        metric: summarize_with_ci(seed_vals)
        for metric, seed_vals in sorted(aggregate_pool.items())
    }
    primary_metric = choose_primary_metric(aggregate)
    primary_best_seed = int(aggregate.get(primary_metric, {}).get("best_seed", -1.0)) if primary_metric else -1
    primary_worst_seed = int(aggregate.get(primary_metric, {}).get("worst_seed", -1.0)) if primary_metric else -1

    payload = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": str(args.model_name),
        "seeds": [int(s) for s in seeds],
        "output_root": str(output_root.resolve()),
        "passthrough_args": clean_passthrough,
        "num_runs_ok": int(sum(1 for r in per_seed_rows if str(r.get("status", "")).startswith("ok"))),
        "num_runs_total": int(len(per_seed_rows)),
        "num_runs_failed": int(len(failed_rows)),
        "failed_runs": failed_rows,
        "primary_metric": primary_metric,
        "primary_metric_best_seed": int(primary_best_seed),
        "primary_metric_worst_seed": int(primary_worst_seed),
        "per_seed": per_seed_rows,
        "aggregate": aggregate,
    }

    out_dir = output_root / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / args.summary_name
    out_csv = out_dir / args.csv_name
    out_agg_csv = out_dir / args.aggregate_csv_name
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_per_seed_csv(out_csv, per_seed_rows)
    write_aggregate_csv(out_agg_csv, aggregate)
    if args.export_paper_table:
        cmd = [
            args.python,
            "scripts/export_paper_tables.py",
            "--seed-sweep-summary",
            str(out_json),
            "--no-ablation",
        ]
        if str(args.paper_table_csv).strip():
            cmd.extend(["--output-csv", str(args.paper_table_csv).strip()])
        if str(args.paper_table_md).strip():
            cmd.extend(["--output-md", str(args.paper_table_md).strip()])
        run_cmd(cmd, ROOT_DIR, timeout_sec=0)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if payload["num_runs_ok"] <= 0:
        raise RuntimeError("No successful seed runs were completed.")
    if payload["num_runs_failed"] > 0 and not args.continue_on_error:
        raise RuntimeError("Seed sweep aborted due to run failure.")


if __name__ == "__main__":
    main()
