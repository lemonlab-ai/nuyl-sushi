import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ActionFormer training/evaluation runner")
    parser.add_argument("--repo", required=True, help="Path to actionformer_release repo")
    parser.add_argument("--config", required=True, help="Path to ActionFormer config yaml")
    parser.add_argument("--output-name", default="ntnu_sushi_exp")
    parser.add_argument("--eval", action="store_true", help="Run eval after training")
    parser.add_argument(
        "--eval-ckpt",
        default="",
        help="Checkpoint folder/file for eval.py (default: infer from config output_folder + output-name).",
    )
    parser.add_argument("--eval-saveonly", action="store_true", help="Use eval.py --saveonly to export eval_results.pkl.")
    parser.add_argument("--eval-topk", type=int, default=-1, help="eval.py -t/--topk")
    parser.add_argument("--eval-print-freq", type=int, default=10, help="eval.py -p/--print-freq")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra", default="", help="Extra args for train.py")
    return parser.parse_args()


def resolve_ckpt_target(repo: Path, cfg: Path, output_name: str, eval_ckpt: str) -> Path:
    if eval_ckpt.strip():
        p = Path(eval_ckpt)
        if not p.is_absolute():
            p = repo / p
        return p
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to parse ActionFormer config.") from exc
    payload = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    output_folder = str(payload.get("output_folder", "./ckpt")).strip() if isinstance(payload, dict) else "./ckpt"
    out_root = Path(output_folder or "./ckpt")
    if not out_root.is_absolute():
        out_root = repo / out_root
    return out_root / f"{cfg.stem}_{output_name}"


def main() -> None:
    args = parse_args()
    repo = Path(args.repo).resolve()
    cfg = Path(args.config).resolve()
    if not repo.exists():
        raise FileNotFoundError(repo)
    if not cfg.exists():
        raise FileNotFoundError(cfg)

    train_cmd = [args.python, "train.py", str(cfg), "--output", args.output_name]
    if args.extra.strip():
        train_cmd.extend(shlex.split(args.extra))

    print("[TRAIN CMD]", " ".join(shlex.quote(x) for x in train_cmd))
    if not args.dry_run:
        subprocess.run(train_cmd, cwd=repo, check=True)

    if args.eval:
        ckpt_target = resolve_ckpt_target(repo, cfg, args.output_name, args.eval_ckpt)
        eval_cmd = [args.python, "eval.py", str(cfg), str(ckpt_target), "-p", str(args.eval_print_freq)]
        if args.eval_topk > 0:
            eval_cmd.extend(["-t", str(args.eval_topk)])
        if args.eval_saveonly:
            eval_cmd.append("--saveonly")
        print("[EVAL CMD ]", " ".join(shlex.quote(x) for x in eval_cmd))
        if not args.dry_run:
            subprocess.run(eval_cmd, cwd=repo, check=True)


if __name__ == "__main__":
    main()
