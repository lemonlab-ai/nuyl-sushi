import argparse
import datetime as dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_KFOLD_ROOT = ROOT_DIR / "outputs" / "kfold_experiments"
DEFAULT_SWEEP_ROOT = ROOT_DIR / "outputs" / "seed_sweeps"
DEFAULT_HANDOFF_ROOT = ROOT_DIR / "outputs" / "outer_handoff"
ROOT_DIR_RESOLVED = ROOT_DIR.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a machine-readable handoff manifest for outer-loop orchestration."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--kfold-root", default=str(DEFAULT_KFOLD_ROOT))
    parser.add_argument("--seed-sweep-root", default=str(DEFAULT_SWEEP_ROOT))
    parser.add_argument("--handoff-root", default=str(DEFAULT_HANDOFF_ROOT))
    parser.add_argument("--strict", action="store_true", help="Fail if required artifacts are missing.")
    parser.add_argument(
        "--require-skill-level-metrics",
        action="store_true",
        help="Treat skill-level benchmark outputs as required artifacts.",
    )
    return parser.parse_args()


def get_cmd_output(cmd: List[str], cwd: Path) -> str:
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
        return (res.stdout or "").strip()
    except Exception:
        return ""


def normalize_path(path: Path) -> Path:
    p = path if path.is_absolute() else (ROOT_DIR / path)
    return p.resolve()


def to_project_relative(abs_path: Path) -> str:
    try:
        rel = abs_path.relative_to(ROOT_DIR_RESOLVED)
        return str(rel).replace("\\", "/")
    except Exception:
        return ""


def file_entry(path: Path) -> Dict[str, object]:
    abs_path = normalize_path(path)
    exists = abs_path.exists()
    return {
        "path": str(abs_path),
        "path_rel_project": to_project_relative(abs_path),
        "exists": bool(exists),
    }


def main() -> None:
    args = parse_args()
    model_name = str(args.model_name).strip()
    if not model_name:
        raise ValueError("--model-name is required")

    kfold_root = Path(args.kfold_root) / model_name
    sweep_root = Path(args.seed_sweep_root) / model_name
    handoff_root = Path(args.handoff_root) / model_name
    handoff_root.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "kfold_experiment_summary": file_entry(kfold_root / "experiment_summary.json"),
        "kfold_paper_table_csv": file_entry(kfold_root / "paper_table.csv"),
        "kfold_paper_table_md": file_entry(kfold_root / "paper_table.md"),
        "seed_sweep_summary": file_entry(sweep_root / "seed_sweep_summary.json"),
        "seed_sweep_per_seed_csv": file_entry(sweep_root / "seed_sweep_summary.csv"),
        "seed_sweep_aggregate_csv": file_entry(sweep_root / "seed_sweep_aggregate.csv"),
        "seed_sweep_paper_table_csv": file_entry(sweep_root / "paper_table.csv"),
        "seed_sweep_paper_table_md": file_entry(sweep_root / "paper_table.md"),
        "skill_level_benchmark_json": file_entry(
            ROOT_DIR / "outputs" / "skill_level" / "skill_level_keypoint_benchmark.json"
        ),
        "skill_level_benchmark_csv": file_entry(
            ROOT_DIR / "outputs" / "skill_level" / "skill_level_keypoint_benchmark.csv"
        ),
    }

    required_keys = [
        "kfold_experiment_summary",
        "seed_sweep_summary",
        "seed_sweep_aggregate_csv",
    ]
    if bool(args.require_skill_level_metrics):
        required_keys.extend(
            [
                "skill_level_benchmark_json",
                "skill_level_benchmark_csv",
            ]
        )
    missing_required = [k for k in required_keys if not bool(artifacts.get(k, {}).get("exists", False))]

    git_commit = get_cmd_output(["git", "rev-parse", "HEAD"], ROOT_DIR) or "unknown"
    git_branch = get_cmd_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], ROOT_DIR) or "unknown"
    python_executable = str(Path(sys.executable).resolve())

    def artifact_abs(key: str) -> str:
        return str(artifacts.get(key, {}).get("path", ""))

    def artifact_rel(key: str) -> str:
        rel = str(artifacts.get(key, {}).get("path_rel_project", ""))
        return rel if rel else artifact_abs(key)

    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(ROOT_DIR_RESOLVED),
        "model_name": model_name,
        "runtime": {
            "python_executable": python_executable,
            "platform": platform.platform(),
        },
        "git": {
            "commit": git_commit,
            "branch": git_branch,
        },
        "artifacts": artifacts,
        "required_artifact_keys": required_keys,
        "require_skill_level_metrics": bool(args.require_skill_level_metrics),
        "missing_required_artifacts": missing_required,
        "ready_for_outer_loop": len(missing_required) == 0,
        "outer_loop_inputs": {
            "primary_metrics_json": artifact_abs("kfold_experiment_summary"),
            "stability_metrics_json": artifact_abs("seed_sweep_summary"),
            "stability_metrics_table_csv": artifact_abs("seed_sweep_aggregate_csv"),
            "skill_level_metrics_json": artifact_abs("skill_level_benchmark_json"),
            "skill_level_metrics_table_csv": artifact_abs("skill_level_benchmark_csv"),
        },
        "outer_loop_inputs_relative": {
            "primary_metrics_json": artifact_rel("kfold_experiment_summary"),
            "stability_metrics_json": artifact_rel("seed_sweep_summary"),
            "stability_metrics_table_csv": artifact_rel("seed_sweep_aggregate_csv"),
            "skill_level_metrics_json": artifact_rel("skill_level_benchmark_json"),
            "skill_level_metrics_table_csv": artifact_rel("skill_level_benchmark_csv"),
        },
        "repro_commands": {
            "ci_smoke": "python scripts/ci_smoke.py",
            "kfold_resume_template": (
                "python scripts/run_kfold_experiments.py "
                f"--model-name {model_name} --run-task-a --resume --export-paper-table"
            ),
            "seed_sweep_template": (
                "python scripts/run_seed_sweep.py "
                f"--model-name {model_name} --seeds 41,42,43 --export-paper-table"
            ),
        },
        "repro_commands_python_explicit": {
            "ci_smoke": f'"{python_executable}" scripts/ci_smoke.py',
            "kfold_resume_template": (
                f'"{python_executable}" scripts/run_kfold_experiments.py '
                f"--model-name {model_name} --run-task-a --resume --export-paper-table"
            ),
            "seed_sweep_template": (
                f'"{python_executable}" scripts/run_seed_sweep.py '
                f"--model-name {model_name} --seeds 41,42,43 --export-paper-table"
            ),
        },
    }

    out_json = handoff_root / "handoff_manifest.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.strict and missing_required:
        raise RuntimeError(f"Missing required artifacts: {missing_required}")


if __name__ == "__main__":
    main()
