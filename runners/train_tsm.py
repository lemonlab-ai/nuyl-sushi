import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSM training runner (official temporal-shift-module repo)")
    parser.add_argument("--repo", required=True, help="Path to temporal-shift-module repo")
    parser.add_argument("--dataset", default="ntnu_sushi")
    parser.add_argument("--root-path", default="", help="TSM frame root (folder with per-clip frame subfolders)")
    parser.add_argument("--train-list", default="", help="TSM train_videofolder.txt path")
    parser.add_argument("--val-list", default="", help="TSM val_videofolder.txt path")
    parser.add_argument("--category-file", default="", help="TSM category.txt path")
    parser.add_argument("--modality", default="RGB")
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--num-segments", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--auto-patch", action="store_true", help="Patch TSM dataset_config.py for ntnu_sushi")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", default="", help="Extra CLI args appended to command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        raise FileNotFoundError(repo)
    if args.auto_patch:
        patch_script = Path(__file__).resolve().parents[1] / "scripts" / "patch_tsm_ntnu_dataset.py"
        patch_cmd = [args.python, str(patch_script), "--repo", str(repo)]
        print("[PATCH CMD]", " ".join(shlex.quote(x) for x in patch_cmd))
        if not args.dry_run:
            subprocess.run(patch_cmd, check=True)

    cmd = [
        args.python,
        "main.py",
        args.dataset,
        args.modality,
        "--arch",
        args.arch,
        "--num_segments",
        str(args.num_segments),
        "--epochs",
        str(args.epochs),
        "-b",
        str(args.batch_size),
        "-j",
        str(args.workers),
        "--lr",
        str(args.lr),
    ]

    if args.extra.strip():
        cmd.extend(shlex.split(args.extra))

    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return
    env = os.environ.copy()
    if args.root_path:
        env["NTNU_SUSHI_TSM_ROOT"] = str(Path(args.root_path).resolve())
    if args.train_list:
        env["NTNU_SUSHI_TSM_TRAIN_LIST"] = str(Path(args.train_list).resolve())
    if args.val_list:
        env["NTNU_SUSHI_TSM_VAL_LIST"] = str(Path(args.val_list).resolve())
    if args.category_file:
        env["NTNU_SUSHI_TSM_CATEGORIES"] = str(Path(args.category_file).resolve())
    subprocess.run(cmd, cwd=repo, check=True, env=env)


if __name__ == "__main__":
    main()
