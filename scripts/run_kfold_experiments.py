import argparse
import atexit
import datetime as dt
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

try:
    from wandb_utils import finish_wandb_run, init_wandb_run, parse_wandb_tags, wandb_log  # type: ignore
except Exception:
    from scripts.wandb_utils import finish_wandb_run, init_wandb_run, parse_wandb_tags, wandb_log  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = ROOT_DIR / "artifacts" / "splits_kfold"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "kfold_experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run k-fold train/predict/eval orchestration (Task A + optional Task B)."
    )
    parser.add_argument("--split-root", default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--fold-pattern", default="fold_*")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-name", default="multimodal_baseline")
    parser.add_argument("--run-task-a", action="store_true")
    parser.add_argument("--run-task-b", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail on missing fold inputs (default: skip fold).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-fold metric json files when present (skip rerun for those tasks).",
    )

    # Task A paths
    parser.add_argument("--task-a-manifest-rel", default="task_a/task_a_fine_manifest.csv")
    parser.add_argument("--task-a-class-map-rel", default="task_a/class_map_fine.json")
    parser.add_argument("--task-a-subset", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--task-a-raw-pred-rel", default="", help="For custom mode: raw prediction file relative to fold.")
    parser.add_argument("--task-a-raw-format", default="auto", choices=["auto", "csv", "json"])

    # Task B paths
    parser.add_argument("--task-b-gt-rel", default="task_b/activitynet_fine.json")
    parser.add_argument(
        "--task-b-subset",
        default="all",
        choices=["training", "validation", "testing", "all"],
    )
    parser.add_argument("--task-b-raw-pred-rel", default="", help="Raw Task B prediction file relative to fold.")
    parser.add_argument("--task-b-raw-format", default="auto", choices=["auto", "csv", "json"])
    parser.add_argument("--task-b-iou-thresholds", default="0.3,0.5,0.75")

    # ActionFormer auto-run options (Task B)
    parser.add_argument("--actionformer-repo", default=str(ROOT_DIR / "external" / "actionformer_release"))
    parser.add_argument("--actionformer-config-rel", default="task_b/actionformer_fine_config.yaml")
    parser.add_argument("--actionformer-label-map-rel", default="task_b/actionformer_label_map_fine.json")
    parser.add_argument(
        "--actionformer-output-tag",
        default="actionformer",
        help="Per-fold output tag -> actual output name is <tag>_<fold_name>.",
    )
    parser.add_argument("--actionformer-skip-train", action="store_true")
    parser.add_argument(
        "--actionformer-ckpt-dir-rel",
        default="",
        help="Optional checkpoint directory path relative to fold dir (or absolute) when --actionformer-skip-train.",
    )
    parser.add_argument("--actionformer-train-extra", default="", help="Extra args for actionformer train.py.")
    parser.add_argument("--actionformer-eval-topk", type=int, default=-1)
    parser.add_argument("--actionformer-eval-print-freq", type=int, default=10)

    # Multimodal baseline options
    parser.add_argument("--multimodal-epochs", type=int, default=100)
    parser.add_argument("--multimodal-lr", type=float, default=0.1)
    parser.add_argument("--multimodal-weight-decay", type=float, default=1e-4)
    parser.add_argument("--multimodal-target-fps", type=float, default=3.0)
    parser.add_argument("--multimodal-max-frames", type=int, default=64)
    parser.add_argument("--multimodal-fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--multimodal-fusion-mod-drop-video", type=float, default=0.05)
    parser.add_argument("--multimodal-fusion-mod-drop-weight", type=float, default=0.20)
    parser.add_argument("--multimodal-fusion-mod-drop-gyro", type=float, default=0.20)
    parser.add_argument("--multimodal-use-reliability-gating", action="store_true")
    parser.add_argument("--multimodal-gating-weight-decay", type=float, default=1e-4)
    parser.add_argument("--multimodal-disable-weight", action="store_true")
    parser.add_argument("--multimodal-disable-gyro", action="store_true")
    parser.add_argument("--multimodal-deepvideo-backbone", choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument("--multimodal-deepvideo-weights", choices=["kinetics400", "none"], default="kinetics400")
    parser.add_argument("--multimodal-deepvideo-device", default="auto")
    parser.add_argument("--multimodal-deepvideo-clip-len", type=int, default=16)
    parser.add_argument("--multimodal-deepvideo-clip-hop", type=int, default=8)
    parser.add_argument("--multimodal-deepvideo-resize-short", type=int, default=128)
    parser.add_argument("--multimodal-deepvideo-crop-size", type=int, default=112)
    parser.add_argument("--multimodal-deepvideo-batch-size", type=int, default=8)
    parser.add_argument("--multimodal-deepvideo-max-sampled-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="ntnu-sushi")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-group", default="kfold")
    parser.add_argument("--wandb-tags", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-log-trainer-runs", action="store_true", help="Forward wandb args to trainer scripts.")
    parser.add_argument("--wandb-log-summary-run", action="store_true", help="Log fold/summary metrics in one orchestration run.")
    parser.add_argument("--export-paper-table", action="store_true")
    parser.add_argument("--paper-table-csv", default="")
    parser.add_argument("--paper-table-md", default="")
    parser.add_argument("--paper-table-ablation-subset", choices=["train", "val", "test"], default="test")

    # Custom command templates
    parser.add_argument(
        "--train-cmd-template",
        default="",
        help=(
            "Optional command template executed per fold. "
            "Available placeholders: {python},{fold_dir},{master_json},{task_a_dir},{task_b_dir},{run_dir}"
        ),
    )
    parser.add_argument(
        "--predict-cmd-template",
        default="",
        help=(
            "Optional prediction command template executed per fold. "
            "Same placeholders as train template."
        ),
    )
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_template(template: str, values: Dict[str, str], cwd: Path) -> None:
    if not template.strip():
        return
    rendered = template.format(**values)
    print("[RUN-TEMPLATE]", rendered)
    subprocess.run(rendered, cwd=cwd, shell=True, check=True)


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def try_load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def get_cmd_output(cmd: List[str], cwd: Path) -> str:
    try:
        res = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return (res.stdout or "").strip()
    except Exception:
        return ""


def get_package_version(name: str) -> str:
    try:
        return str(importlib.metadata.version(name))
    except Exception:
        return "not_installed"


def collect_reproducibility_metadata(args: argparse.Namespace, fold_dirs: List[Path]) -> Dict:
    git_commit = get_cmd_output(["git", "rev-parse", "HEAD"], ROOT_DIR)
    git_branch = get_cmd_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], ROOT_DIR)
    git_status = get_cmd_output(["git", "status", "--porcelain"], ROOT_DIR)
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "platform": platform.platform(),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": str(sys.version.replace("\n", " ")),
        },
        "packages": {
            "torch": get_package_version("torch"),
            "numpy": get_package_version("numpy"),
            "wandb": get_package_version("wandb"),
        },
        "git": {
            "commit": git_commit or "unknown",
            "branch": git_branch or "unknown",
            "dirty": bool(git_status.strip()),
        },
        "folds_detected": [p.name for p in fold_dirs],
        "run_config": {k: v for k, v in vars(args).items()},
    }


def build_trainer_wandb_args(args: argparse.Namespace, fold_name: str) -> List[str]:
    if not args.wandb or not args.wandb_log_trainer_runs:
        return []
    tags = parse_wandb_tags(args.wandb_tags)
    tags.extend([args.model_name, fold_name, "trainer"])
    dedup_tags = []
    seen = set()
    for t in tags:
        if t not in seen:
            seen.add(t)
            dedup_tags.append(t)
    out = [
        "--wandb",
        "--wandb-project",
        str(args.wandb_project),
        "--wandb-run-name",
        f"{args.model_name}_{fold_name}",
        "--wandb-group",
        str(args.wandb_group),
        "--wandb-mode",
        str(args.wandb_mode),
        "--wandb-tags",
        ",".join(dedup_tags),
    ]
    if str(args.wandb_entity).strip():
        out.extend(["--wandb-entity", str(args.wandb_entity).strip()])
    return out


def summarize_metric(records: List[Dict], metric_key: str) -> Dict[str, float]:
    vals = []
    for r in records:
        v = r.get(metric_key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(vals)),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def load_actionformer_output_folder(config_path: Path) -> str:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to parse ActionFormer config.") from exc
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return "./ckpt"
    out = str(payload.get("output_folder", "./ckpt")).strip()
    return out or "./ckpt"


def resolve_actionformer_ckpt_dir(
    actionformer_repo: Path,
    config_path: Path,
    output_name: str,
    fold_dir: Path,
    ckpt_override_rel: str,
) -> Path:
    if ckpt_override_rel.strip():
        override = Path(ckpt_override_rel)
        if not override.is_absolute():
            override = fold_dir / override
        return override

    output_folder = Path(load_actionformer_output_folder(config_path))
    if not output_folder.is_absolute():
        output_folder = actionformer_repo / output_folder
    return output_folder / f"{config_path.stem}_{output_name}"


def main() -> None:
    args = parse_args()
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted([p for p in split_root.glob(args.fold_pattern) if p.is_dir()])
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found: {split_root} pattern={args.fold_pattern}")
    if not args.run_task_a and not args.run_task_b:
        args.run_task_a = True

    wandb_run = init_wandb_run(
        enabled=bool(args.wandb and args.wandb_log_summary_run),
        project=str(args.wandb_project),
        entity=str(args.wandb_entity),
        run_name=str(args.wandb_run_name),
        group=str(args.wandb_group),
        tags=parse_wandb_tags(args.wandb_tags),
        mode=str(args.wandb_mode),
        job_type="kfold_orchestration",
        config={
            "model_name": str(args.model_name),
            "run_task_a": bool(args.run_task_a),
            "run_task_b": bool(args.run_task_b),
            "num_folds_detected": int(len(fold_dirs)),
            "seed": int(args.seed),
            "multimodal_epochs": int(args.multimodal_epochs),
            "multimodal_use_reliability_gating": bool(args.multimodal_use_reliability_gating),
            "wandb_log_trainer_runs": bool(args.wandb_log_trainer_runs),
        },
    )
    if wandb_run is not None:
        atexit.register(lambda: finish_wandb_run(wandb_run))

    actionformer_repo = Path(args.actionformer_repo)
    if not actionformer_repo.is_absolute():
        actionformer_repo = ROOT_DIR / actionformer_repo

    task_a_records = []
    task_b_records = []
    skipped_folds: List[str] = []

    fold_idx = -1
    for fold_dir in fold_dirs:
        fold_idx += 1
        fold_name = fold_dir.name
        trainer_wandb_args = build_trainer_wandb_args(args, fold_name)
        master_json = fold_dir / "master_annotations.json"
        task_a_dir = fold_dir / "task_a"
        task_b_dir = fold_dir / "task_b"
        run_dir = output_root / args.model_name / fold_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if not master_json.exists():
            msg = f"[SKIP] {fold_name}: missing {master_json}"
            if args.strict:
                raise FileNotFoundError(master_json)
            print(msg)
            skipped_folds.append(fold_name)
            continue

        values = {
            "python": args.python,
            "fold_dir": str(fold_dir.resolve()),
            "master_json": str(master_json.resolve()),
            "task_a_dir": str(task_a_dir.resolve()),
            "task_b_dir": str(task_b_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
        }

        run_template(args.train_cmd_template, values, ROOT_DIR)
        run_template(args.predict_cmd_template, values, ROOT_DIR)

        if args.run_task_a:
            task_a_metrics_path = run_dir / "task_a_metrics.json"
            if args.resume and task_a_metrics_path.exists():
                cached = try_load_json(task_a_metrics_path)
                if cached:
                    print(f"[RESUME] {fold_name} task_a: reuse {task_a_metrics_path}")
                    task_a_records.append({"fold": fold_name, "resumed": True, **cached})
                    wandb_log(
                        wandb_run,
                        {
                            "fold/index": int(fold_idx),
                            "fold/name": str(fold_name),
                            "task_a/top1_accuracy": float(cached.get("top1_accuracy", 0.0)),
                            "task_a/macro_f1": float(cached.get("macro_f1", 0.0)),
                            "task_a/balanced_accuracy": float(cached.get("balanced_accuracy", 0.0)),
                            "task_a/resumed": 1.0,
                        },
                    )
                    skip_task_a_execution = True
                else:
                    skip_task_a_execution = False
            else:
                skip_task_a_execution = False

            if skip_task_a_execution:
                pass
            else:
                manifest = fold_dir / args.task_a_manifest_rel
                class_map = fold_dir / args.task_a_class_map_rel
                pred_eval_csv = run_dir / "task_a_pred_eval.csv"
                should_eval_task_a = True
                if not manifest.exists() or not class_map.exists():
                    msg = f"[SKIP] {fold_name} task_a: missing manifest/class_map."
                    if args.strict:
                        raise FileNotFoundError(f"{manifest} or {class_map}")
                    print(msg)
                    skipped_folds.append(fold_name)
                    should_eval_task_a = False

                if should_eval_task_a:
                    if args.model_name == "multimodal_baseline":
                        model_out = run_dir / "task_a_multimodal"
                        run_cmd(
                            [
                                args.python,
                                "scripts/train_multimodal_baseline.py",
                                "--manifest-csv",
                                str(manifest),
                                "--master-json",
                                str(master_json),
                                "--output-dir",
                                str(model_out),
                                "--epochs",
                                str(args.multimodal_epochs),
                                "--lr",
                                str(args.multimodal_lr),
                                "--weight-decay",
                                str(args.multimodal_weight_decay),
                                "--target-fps",
                                str(args.multimodal_target_fps),
                                "--max-frames",
                                str(args.multimodal_max_frames),
                                "--seed",
                                str(args.seed),
                            ]
                            + trainer_wandb_args
                            + (["--disable-weight"] if args.multimodal_disable_weight else [])
                            + (["--disable-gyro"] if args.multimodal_disable_gyro else []),
                            ROOT_DIR,
                        )
                        pred_eval_csv = model_out / "predictions.csv"
                    elif args.model_name == "multimodal_fusion":
                        model_out = run_dir / "task_a_multimodal_fusion"
                        run_cmd(
                            [
                                args.python,
                                "scripts/train_multimodal_fusion.py",
                                "--manifest-csv",
                                str(manifest),
                                "--master-json",
                                str(master_json),
                                "--output-dir",
                                str(model_out),
                                "--epochs",
                                str(args.multimodal_epochs),
                                "--lr",
                                str(args.multimodal_lr),
                                "--weight-decay",
                                str(args.multimodal_weight_decay),
                                "--target-fps",
                                str(args.multimodal_target_fps),
                                "--max-frames",
                                str(args.multimodal_max_frames),
                                "--hidden-dim",
                                str(args.multimodal_fusion_hidden_dim),
                                "--mod-drop-video",
                                str(args.multimodal_fusion_mod_drop_video),
                                "--mod-drop-weight",
                                str(args.multimodal_fusion_mod_drop_weight),
                                "--mod-drop-gyro",
                                str(args.multimodal_fusion_mod_drop_gyro),
                                "--gating-weight-decay",
                                str(args.multimodal_gating_weight_decay),
                                "--seed",
                                str(args.seed),
                            ]
                            + trainer_wandb_args
                            + (["--use-reliability-gating"] if args.multimodal_use_reliability_gating else [])
                            + (["--disable-weight"] if args.multimodal_disable_weight else [])
                            + (["--disable-gyro"] if args.multimodal_disable_gyro else []),
                            ROOT_DIR,
                        )
                        pred_eval_csv = model_out / "predictions.csv"
                    elif args.model_name == "multimodal_fusion_deepvideo":
                        model_out = run_dir / "task_a_multimodal_fusion_deepvideo"
                        run_cmd(
                            [
                                args.python,
                                "scripts/train_multimodal_fusion_deepvideo.py",
                                "--manifest-csv",
                                str(manifest),
                                "--master-json",
                                str(master_json),
                                "--output-dir",
                                str(model_out),
                                "--epochs",
                                str(args.multimodal_epochs),
                                "--lr",
                                str(args.multimodal_lr),
                                "--weight-decay",
                                str(args.multimodal_weight_decay),
                                "--hidden-dim",
                                str(args.multimodal_fusion_hidden_dim),
                                "--video-backbone",
                                args.multimodal_deepvideo_backbone,
                                "--video-weights",
                                args.multimodal_deepvideo_weights,
                                "--video-device",
                                args.multimodal_deepvideo_device,
                                "--target-fps",
                                str(args.multimodal_target_fps),
                                "--clip-len",
                                str(args.multimodal_deepvideo_clip_len),
                                "--clip-hop",
                                str(args.multimodal_deepvideo_clip_hop),
                                "--resize-short",
                                str(args.multimodal_deepvideo_resize_short),
                                "--crop-size",
                                str(args.multimodal_deepvideo_crop_size),
                                "--batch-size",
                                str(args.multimodal_deepvideo_batch_size),
                                "--max-sampled-frames",
                                str(args.multimodal_deepvideo_max_sampled_frames),
                                "--mod-drop-video",
                                str(args.multimodal_fusion_mod_drop_video),
                                "--mod-drop-weight",
                                str(args.multimodal_fusion_mod_drop_weight),
                                "--mod-drop-gyro",
                                str(args.multimodal_fusion_mod_drop_gyro),
                                "--gating-weight-decay",
                                str(args.multimodal_gating_weight_decay),
                                "--seed",
                                str(args.seed),
                            ]
                            + trainer_wandb_args
                            + (["--use-reliability-gating"] if args.multimodal_use_reliability_gating else [])
                            + (["--disable-weight"] if args.multimodal_disable_weight else [])
                            + (["--disable-gyro"] if args.multimodal_disable_gyro else []),
                            ROOT_DIR,
                        )
                        pred_eval_csv = model_out / "predictions.csv"
                    else:
                        if not args.task_a_raw_pred_rel:
                            print(
                                f"[SKIP] {fold_name} task_a skipped for model={args.model_name}: "
                                "--task-a-raw-pred-rel not provided."
                            )
                            should_eval_task_a = False
                        else:
                            raw_pred = fold_dir / args.task_a_raw_pred_rel
                            run_cmd(
                                [
                                    args.python,
                                    "scripts/convert_task_a_predictions.py",
                                    "--input",
                                    str(raw_pred),
                                    "--output-csv",
                                    str(pred_eval_csv),
                                    "--class-map",
                                    str(class_map),
                                    "--format",
                                    args.task_a_raw_format,
                                ],
                                ROOT_DIR,
                            )

                if should_eval_task_a:
                    run_cmd(
                        [
                            args.python,
                            "scripts/eval_task_a.py",
                            "--manifest-csv",
                            str(manifest),
                            "--class-map",
                            str(class_map),
                            "--pred-csv",
                            str(pred_eval_csv),
                            "--subset",
                            args.task_a_subset,
                            "--output-json",
                            str(task_a_metrics_path),
                        ],
                        ROOT_DIR,
                    )
                    metrics = load_json(task_a_metrics_path)
                    task_a_records.append({"fold": fold_name, **metrics})
                    wandb_log(
                        wandb_run,
                        {
                            "fold/index": int(fold_idx),
                            "fold/name": str(fold_name),
                            "task_a/top1_accuracy": float(metrics.get("top1_accuracy", 0.0)),
                            "task_a/macro_f1": float(metrics.get("macro_f1", 0.0)),
                            "task_a/balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                            "task_a/resumed": 0.0,
                        },
                    )

        if args.run_task_b:
            task_b_metrics_path = run_dir / "task_b_metrics.json"
            if args.resume and task_b_metrics_path.exists():
                cached = try_load_json(task_b_metrics_path)
                if cached:
                    print(f"[RESUME] {fold_name} task_b: reuse {task_b_metrics_path}")
                    task_b_records.append({"fold": fold_name, "resumed": True, **cached})
                    wandb_log(
                        wandb_run,
                        {
                            "fold/index": int(fold_idx),
                            "fold/name": str(fold_name),
                            "task_b/average_mAP": float(cached.get("average_mAP", 0.0)),
                            "task_b/average_Recall": float(cached.get("average_Recall", 0.0)),
                            "task_b/resumed": 1.0,
                        },
                    )
                    continue
            if args.model_name == "actionformer":
                gt_json = fold_dir / args.task_b_gt_rel
                af_config = fold_dir / args.actionformer_config_rel
                af_label_map = fold_dir / args.actionformer_label_map_rel
                if not gt_json.exists() or not af_config.exists():
                    msg = f"[SKIP] {fold_name} task_b(actionformer): missing gt/config input."
                    if args.strict:
                        raise FileNotFoundError(f"{gt_json} or {af_config}")
                    print(msg)
                    skipped_folds.append(fold_name)
                    continue
                if not actionformer_repo.exists():
                    raise FileNotFoundError(actionformer_repo)

                af_output_name = f"{args.actionformer_output_tag}_{fold_name}"
                af_ckpt_dir = resolve_actionformer_ckpt_dir(
                    actionformer_repo=actionformer_repo,
                    config_path=af_config,
                    output_name=af_output_name,
                    fold_dir=fold_dir,
                    ckpt_override_rel=args.actionformer_ckpt_dir_rel,
                )

                if not args.actionformer_skip_train:
                    af_train_cmd = [args.python, "train.py", str(af_config.resolve()), "--output", af_output_name]
                    if args.actionformer_train_extra.strip():
                        af_train_cmd.extend(shlex.split(args.actionformer_train_extra))
                    run_cmd(af_train_cmd, actionformer_repo)

                if not af_ckpt_dir.exists():
                    msg = f"[SKIP] {fold_name} task_b(actionformer): missing ckpt dir {af_ckpt_dir}"
                    if args.strict:
                        raise FileNotFoundError(af_ckpt_dir)
                    print(msg)
                    skipped_folds.append(fold_name)
                    continue

                raw_pred = af_ckpt_dir / "eval_results.pkl"
                if (not args.actionformer_skip_train) or (not raw_pred.exists()):
                    af_eval_cmd = [
                        args.python,
                        "eval.py",
                        str(af_config.resolve()),
                        str(af_ckpt_dir.resolve()),
                        "--saveonly",
                        "-p",
                        str(args.actionformer_eval_print_freq),
                    ]
                    if args.actionformer_eval_topk > 0:
                        af_eval_cmd.extend(["-t", str(args.actionformer_eval_topk)])
                    run_cmd(af_eval_cmd, actionformer_repo)

                if not raw_pred.exists():
                    msg = f"[SKIP] {fold_name} task_b(actionformer): missing {raw_pred}"
                    if args.strict:
                        raise FileNotFoundError(raw_pred)
                    print(msg)
                    skipped_folds.append(fold_name)
                    continue

                pred_eval_json = run_dir / "task_b_pred_eval.json"
                convert_cmd = [
                    args.python,
                    "scripts/convert_task_b_predictions.py",
                    "--input",
                    str(raw_pred),
                    "--output-json",
                    str(pred_eval_json),
                    "--format",
                    "pkl",
                ]
                if af_label_map.exists():
                    convert_cmd.extend(["--label-map", str(af_label_map)])
                run_cmd(convert_cmd, ROOT_DIR)

                run_cmd(
                    [
                        args.python,
                        "scripts/eval_task_b.py",
                        "--gt-json",
                        str(gt_json),
                        "--pred-json",
                        str(pred_eval_json),
                        "--subset",
                        args.task_b_subset,
                        "--iou-thresholds",
                        args.task_b_iou_thresholds,
                        "--output-json",
                        str(task_b_metrics_path),
                    ],
                    ROOT_DIR,
                )
                metrics = load_json(task_b_metrics_path)
                task_b_records.append(
                    {
                        "fold": fold_name,
                        "actionformer_ckpt_dir": str(af_ckpt_dir.resolve()),
                        **metrics,
                    }
                )
                wandb_log(
                    wandb_run,
                    {
                        "fold/index": int(fold_idx),
                        "fold/name": str(fold_name),
                        "task_b/average_mAP": float(metrics.get("average_mAP", 0.0)),
                        "task_b/average_Recall": float(metrics.get("average_Recall", 0.0)),
                        "task_b/resumed": 0.0,
                    },
                )
            elif not args.task_b_raw_pred_rel:
                print(f"[SKIP] {fold_name} task_b skipped: --task-b-raw-pred-rel not provided.")
            else:
                gt_json = fold_dir / args.task_b_gt_rel
                raw_pred = fold_dir / args.task_b_raw_pred_rel
                if not gt_json.exists() or not raw_pred.exists():
                    msg = f"[SKIP] {fold_name} task_b: missing gt/pred input."
                    if args.strict:
                        raise FileNotFoundError(f"{gt_json} or {raw_pred}")
                    print(msg)
                    skipped_folds.append(fold_name)
                    continue
                pred_eval_json = run_dir / "task_b_pred_eval.json"
                run_cmd(
                    [
                        args.python,
                        "scripts/convert_task_b_predictions.py",
                        "--input",
                        str(raw_pred),
                        "--output-json",
                        str(pred_eval_json),
                        "--format",
                        args.task_b_raw_format,
                    ],
                    ROOT_DIR,
                )
                run_cmd(
                    [
                        args.python,
                        "scripts/eval_task_b.py",
                        "--gt-json",
                        str(gt_json),
                        "--pred-json",
                        str(pred_eval_json),
                        "--subset",
                        args.task_b_subset,
                        "--iou-thresholds",
                        args.task_b_iou_thresholds,
                        "--output-json",
                        str(task_b_metrics_path),
                    ],
                    ROOT_DIR,
                )
                metrics = load_json(task_b_metrics_path)
                task_b_records.append({"fold": fold_name, **metrics})
                wandb_log(
                    wandb_run,
                    {
                        "fold/index": int(fold_idx),
                        "fold/name": str(fold_name),
                        "task_b/average_mAP": float(metrics.get("average_mAP", 0.0)),
                        "task_b/average_Recall": float(metrics.get("average_Recall", 0.0)),
                        "task_b/resumed": 0.0,
                    },
                )

    reproducibility = collect_reproducibility_metadata(args, fold_dirs)
    num_task_a_resumed = int(sum(1 for r in task_a_records if bool(r.get("resumed", False))))
    num_task_b_resumed = int(sum(1 for r in task_b_records if bool(r.get("resumed", False))))
    summary = {
        "model_name": args.model_name,
        "num_folds": len(fold_dirs),
        "num_skipped_folds": len(sorted(set(skipped_folds))),
        "skipped_folds": sorted(set(skipped_folds)),
        "task_a": {
            "num_records": len(task_a_records),
            "num_resumed_records": num_task_a_resumed,
            "top1_accuracy": summarize_metric(task_a_records, "top1_accuracy"),
            "macro_f1": summarize_metric(task_a_records, "macro_f1"),
            "balanced_accuracy": summarize_metric(task_a_records, "balanced_accuracy"),
            "records": task_a_records,
        },
        "task_b": {
            "num_records": len(task_b_records),
            "num_resumed_records": num_task_b_resumed,
            "average_mAP": summarize_metric(task_b_records, "average_mAP"),
            "average_Recall": summarize_metric(task_b_records, "average_Recall"),
            "records": task_b_records,
        },
        "reproducibility": reproducibility,
    }
    out_json = output_root / args.model_name / "experiment_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.export_paper_table:
        export_cmd = [
            args.python,
            "scripts/export_paper_tables.py",
            "--experiment-summary",
            str(out_json),
            "--ablation-subset",
            str(args.paper_table_ablation_subset),
        ]
        if str(args.paper_table_csv).strip():
            export_cmd.extend(["--output-csv", str(args.paper_table_csv).strip()])
        if str(args.paper_table_md).strip():
            export_cmd.extend(["--output-md", str(args.paper_table_md).strip()])
        run_cmd(export_cmd, ROOT_DIR)
    wandb_log(
        wandb_run,
        {
            "summary/task_a/top1_mean": float(summary["task_a"]["top1_accuracy"]["mean"]),
            "summary/task_a/macro_f1_mean": float(summary["task_a"]["macro_f1"]["mean"]),
            "summary/task_a/balanced_accuracy_mean": float(summary["task_a"]["balanced_accuracy"]["mean"]),
            "summary/task_b/average_mAP_mean": float(summary["task_b"]["average_mAP"]["mean"]),
            "summary/task_b/average_Recall_mean": float(summary["task_b"]["average_Recall"]["mean"]),
            "summary/num_records_task_a": int(summary["task_a"]["num_records"]),
            "summary/num_records_task_b": int(summary["task_b"]["num_records"]),
            "summary/num_task_a_resumed_records": int(summary["task_a"]["num_resumed_records"]),
            "summary/num_task_b_resumed_records": int(summary["task_b"]["num_resumed_records"]),
            "summary/num_skipped_folds": int(summary["num_skipped_folds"]),
        },
    )
    print("[DONE] k-fold experiment orchestration completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
