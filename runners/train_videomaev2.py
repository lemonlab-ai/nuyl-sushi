import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoMAE V2 fine-tuning runner")
    parser.add_argument("--repo", required=True, help="Path to VideoMAEv2 repo")
    parser.add_argument("--data-path", required=True, help="Folder containing train.csv/val.csv/test.csv")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root folder for video paths referenced from CSV (usually artifacts/task_a)",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nb-classes", type=int, required=True)
    parser.add_argument("--data-set", default="NTNU_SUSHI")
    parser.add_argument("--model", default="vit_small_patch16_224")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sampling-rate", type=int, default=4)
    parser.add_argument("--finetune", default="", help="Optional pretrained checkpoint path")
    parser.add_argument("--auto-patch", action="store_true", help="Patch VideoMAEv2 for NTNU_SUSHI support")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", default="", help="Extra CLI args appended to command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    data_path = Path(args.data_path).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    finetune = Path(args.finetune).resolve() if args.finetune else None
    if not repo.exists():
        raise FileNotFoundError(repo)
    if args.auto_patch:
        patch_script = Path(__file__).resolve().parents[1] / "scripts" / "patch_videomaev2_ntnu.py"
        patch_cmd = [args.python, str(patch_script), "--repo", str(repo)]
        print("[PATCH CMD]", " ".join(shlex.quote(x) for x in patch_cmd))
        if not args.dry_run:
            subprocess.run(patch_cmd, check=True)

    cmd = [
        args.python,
        "run_class_finetuning.py",
        "--model",
        args.model,
        "--data_set",
        args.data_set,
        "--data_path",
        str(data_path),
        "--data_root",
        str(data_root),
        "--nb_classes",
        str(args.nb_classes),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_frames",
        str(args.num_frames),
        "--sampling_rate",
        str(args.sampling_rate),
        "--output_dir",
        str(output_dir),
    ]
    if finetune:
        cmd.extend(["--finetune", str(finetune)])
    if args.extra.strip():
        cmd.extend(shlex.split(args.extra))

    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()
