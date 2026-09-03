import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SlowFast training runner")
    parser.add_argument("--repo", required=True, help="Path to SlowFast repo")
    parser.add_argument("--cfg", required=True, help="Path to SlowFast YAML config")
    parser.add_argument("--data-dir", required=True, help="Path to Task A data root")
    parser.add_argument("--output-dir", required=True, help="Checkpoint/output dir")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", default="", help="Extra CLI args appended to command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    cfg = Path(args.cfg).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not repo.exists():
        raise FileNotFoundError(repo)
    if not cfg.exists():
        raise FileNotFoundError(cfg)

    cmd = [
        args.python,
        "tools/run_net.py",
        "--cfg",
        str(cfg),
        "NUM_GPUS",
        str(args.num_gpus),
        "DATA.PATH_TO_DATA_DIR",
        str(data_dir),
        "OUTPUT_DIR",
        str(output_dir),
    ]
    if args.extra.strip():
        cmd.extend(shlex.split(args.extra))

    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) if not current_pythonpath else f"{repo}{os.pathsep}{current_pythonpath}"
    subprocess.run(cmd, cwd=repo, check=True, env=env)


if __name__ == "__main__":
    main()
