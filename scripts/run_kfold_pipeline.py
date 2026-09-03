import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
DEFAULT_OUTPUT = ROOT_DIR / "artifacts" / "splits_kfold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-click k-fold data pipeline (export Task A/B + adapters) for all folds."
    )
    parser.add_argument("--master-json", default=str(DEFAULT_MASTER))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kfold-stratify-level", choices=["none", "coarse", "fine", "both"], default="fine")
    parser.add_argument("--kfold-subject-leakage-policy", choices=["warn", "fail"], default="fail")
    parser.add_argument("--kfold-imbalance-policy", choices=["off", "warn", "fail"], default="warn")
    parser.add_argument("--kfold-max-label-relative-deviation", type=float, default=0.8)
    parser.add_argument("--kfold-imbalance-min-total", type=int, default=5)
    parser.add_argument("--kfold-min-test-label-count-per-fold", type=int, default=0)
    parser.add_argument("--kfold-min-count-policy", choices=["off", "warn", "fail"], default="off")
    parser.add_argument("--trim-clips", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--task-a-level", choices=["coarse", "fine", "both"], default="both")
    parser.add_argument("--task-b-level", choices=["coarse", "fine", "both"], default="both")
    parser.add_argument(
        "--repo-format-level",
        choices=["coarse", "fine"],
        default="fine",
        help="Label level used to build repo formats from Task A manifest.",
    )
    parser.add_argument("--prepare-repo-formats", action="store_true", help="Export VideoMAEv2/SlowFast formats.")
    parser.add_argument("--prepare-tsm", action="store_true", help="When preparing repo formats, also export TSM.")
    parser.add_argument("--prepare-actionformer", action="store_true", help="Export ActionFormer JSON/config.")
    parser.add_argument("--extract-actionformer-features", action="store_true")
    parser.add_argument(
        "--actionformer-feature-extractor",
        choices=["basic", "deep", "ensemble"],
        default="basic",
        help="Feature extractor used when --extract-actionformer-features is enabled.",
    )
    parser.add_argument("--actionformer-input-dim", type=int, default=13)
    parser.add_argument("--actionformer-infer-input-dim", action="store_true")
    parser.add_argument("--actionformer-feature-dir-name", default="")

    # deep feature extractor options
    parser.add_argument("--deep-feature-model", choices=["r3d_18", "mc3_18", "r2plus1d_18"], default="r3d_18")
    parser.add_argument("--deep-feature-weights", choices=["kinetics400", "none"], default="kinetics400")
    parser.add_argument("--deep-feature-device", default="auto")
    parser.add_argument("--deep-feature-target-fps", type=float, default=8.0)
    parser.add_argument("--deep-feature-clip-len", type=int, default=16)
    parser.add_argument("--deep-feature-clip-hop", type=int, default=8)
    parser.add_argument("--deep-feature-resize-short", type=int, default=128)
    parser.add_argument("--deep-feature-crop-size", type=int, default=112)
    parser.add_argument("--deep-feature-batch-size", type=int, default=8)
    parser.add_argument("--deep-feature-max-sampled-frames", type=int, default=0)
    parser.add_argument(
        "--ensemble-feature-models",
        default="r3d_18,mc3_18,r2plus1d_18",
        help="Comma-separated model list for --actionformer-feature-extractor ensemble.",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    args = parse_args()
    py = args.python
    root = ROOT_DIR

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Generate k-fold masters
    run_cmd(
        [
            py,
            "scripts/generate_subject_kfold_masters.py",
            "--master-json",
            str(Path(args.master_json)),
            "--output-dir",
            str(output_dir),
            "--num-folds",
            str(args.num_folds),
            "--val-ratio",
            str(args.val_ratio),
            "--seed",
            str(args.seed),
            "--stratify-level",
            str(args.kfold_stratify_level),
            "--subject-leakage-policy",
            str(args.kfold_subject_leakage_policy),
            "--imbalance-policy",
            str(args.kfold_imbalance_policy),
            "--max-label-relative-deviation",
            str(args.kfold_max_label_relative_deviation),
            "--imbalance-min-total",
            str(args.kfold_imbalance_min_total),
            "--min-test-label-count-per-fold",
            str(args.kfold_min_test_label_count_per_fold),
            "--min-count-policy",
            str(args.kfold_min_count_policy),
        ],
        cwd=root,
    )

    run_summary: Dict[str, Dict] = {}

    # 2) Per-fold exports
    for fold_idx in range(args.num_folds):
        fold_name = f"fold_{fold_idx:02d}"
        fold_dir = output_dir / fold_name
        fold_master = fold_dir / "master_annotations.json"
        if not fold_master.exists():
            raise FileNotFoundError(fold_master)

        task_a_dir = fold_dir / "task_a"
        task_b_dir = fold_dir / "task_b"
        repo_dir = fold_dir / "repo_formats"

        run_cmd(
            [
                py,
                "scripts/export_task_a_clips.py",
                "--master-json",
                str(fold_master),
                "--output-dir",
                str(task_a_dir),
                "--level",
                args.task_a_level,
            ]
            + (["--trim-clips", "--ffmpeg", args.ffmpeg] if args.trim_clips else []),
            cwd=root,
        )

        run_cmd(
            [
                py,
                "scripts/export_task_b_activitynet.py",
                "--master-json",
                str(fold_master),
                "--output-dir",
                str(task_b_dir),
                "--level",
                args.task_b_level,
            ],
            cwd=root,
        )

        if args.prepare_repo_formats:
            manifest = task_a_dir / f"task_a_{args.repo_format_level}_manifest.csv"
            class_map = task_a_dir / f"class_map_{args.repo_format_level}.json"
            cmd = [
                py,
                "scripts/prepare_task_a_repo_formats.py",
                "--manifest-csv",
                str(manifest),
                "--class-map",
                str(class_map),
                "--output-root",
                str(repo_dir),
                "--task-a-root",
                str(task_a_dir),
                "--path-mode",
                "relative",
            ]
            if not args.trim_clips:
                cmd.append("--allow-video-fallback")
            if args.prepare_tsm:
                cmd.append("--prepare-tsm")
                cmd.extend(["--ffmpeg", args.ffmpeg])
            run_cmd(cmd, cwd=root)

        if args.prepare_actionformer:
            if args.repo_format_level not in {"coarse", "fine"}:
                raise ValueError("--repo-format-level must be coarse/fine")
            activitynet_src = task_b_dir / f"activitynet_{args.repo_format_level}.json"
            actionformer_json = task_b_dir / f"actionformer_{args.repo_format_level}.json"
            label_map_out = task_b_dir / f"actionformer_label_map_{args.repo_format_level}.json"
            config_out = task_b_dir / f"actionformer_{args.repo_format_level}_config.yaml"
            if args.actionformer_feature_dir_name.strip():
                feature_dir = task_b_dir / args.actionformer_feature_dir_name.strip()
            elif args.actionformer_feature_extractor == "deep":
                feature_dir = task_b_dir / f"features_deep_{args.deep_feature_model}"
            elif args.actionformer_feature_extractor == "ensemble":
                feature_dir = task_b_dir / "features_deep_ensemble"
            else:
                feature_dir = task_b_dir / "features_basic"

            if args.extract_actionformer_features:
                # prepare ActionFormer json first so extractor can read source_path from it
                run_cmd(
                    [
                        py,
                        "scripts/prepare_task_b_actionformer.py",
                        "--activitynet-json",
                        str(activitynet_src),
                        "--output-json",
                        str(actionformer_json),
                        "--label-map-out",
                        str(label_map_out),
                        "--config-out",
                        str(config_out),
                        "--feature-dir",
                        str(feature_dir),
                        "--input-dim",
                        str(args.actionformer_input_dim),
                    ],
                    cwd=root,
                )
                if args.actionformer_feature_extractor == "deep":
                    run_cmd(
                        [
                            py,
                            "scripts/extract_task_b_features_deep.py",
                            "--gt-json",
                            str(actionformer_json),
                            "--output-dir",
                            str(feature_dir),
                            "--model",
                            args.deep_feature_model,
                            "--weights",
                            args.deep_feature_weights,
                            "--device",
                            args.deep_feature_device,
                            "--target-fps",
                            str(args.deep_feature_target_fps),
                            "--clip-len",
                            str(args.deep_feature_clip_len),
                            "--clip-hop",
                            str(args.deep_feature_clip_hop),
                            "--resize-short",
                            str(args.deep_feature_resize_short),
                            "--crop-size",
                            str(args.deep_feature_crop_size),
                            "--batch-size",
                            str(args.deep_feature_batch_size),
                            "--max-sampled-frames",
                            str(args.deep_feature_max_sampled_frames),
                        ],
                        cwd=root,
                    )
                elif args.actionformer_feature_extractor == "ensemble":
                    run_cmd(
                        [
                            py,
                            "scripts/extract_task_b_features_ensemble.py",
                            "--gt-json",
                            str(actionformer_json),
                            "--output-dir",
                            str(feature_dir),
                            "--models",
                            args.ensemble_feature_models,
                            "--weights",
                            args.deep_feature_weights,
                            "--device",
                            args.deep_feature_device,
                            "--target-fps",
                            str(args.deep_feature_target_fps),
                            "--clip-len",
                            str(args.deep_feature_clip_len),
                            "--clip-hop",
                            str(args.deep_feature_clip_hop),
                            "--resize-short",
                            str(args.deep_feature_resize_short),
                            "--crop-size",
                            str(args.deep_feature_crop_size),
                            "--batch-size",
                            str(args.deep_feature_batch_size),
                            "--max-sampled-frames",
                            str(args.deep_feature_max_sampled_frames),
                        ],
                        cwd=root,
                    )
                else:
                    run_cmd(
                        [
                            py,
                            "scripts/extract_task_b_features_basic.py",
                            "--gt-json",
                            str(actionformer_json),
                            "--output-dir",
                            str(feature_dir),
                        ],
                        cwd=root,
                    )

                # regenerate config with inferred input dim from extracted features
                run_cmd(
                    [
                        py,
                        "scripts/prepare_task_b_actionformer.py",
                        "--activitynet-json",
                        str(activitynet_src),
                        "--output-json",
                        str(actionformer_json),
                        "--label-map-out",
                        str(label_map_out),
                        "--config-out",
                        str(config_out),
                        "--feature-dir",
                        str(feature_dir),
                        "--input-dim",
                        str(args.actionformer_input_dim),
                        "--infer-input-dim",
                    ],
                    cwd=root,
                )
            else:
                prepare_cmd = [
                    py,
                    "scripts/prepare_task_b_actionformer.py",
                    "--activitynet-json",
                    str(activitynet_src),
                    "--output-json",
                    str(actionformer_json),
                    "--label-map-out",
                    str(label_map_out),
                    "--config-out",
                    str(config_out),
                    "--feature-dir",
                    str(feature_dir),
                    "--input-dim",
                    str(args.actionformer_input_dim),
                ]
                if args.actionformer_infer_input_dim:
                    prepare_cmd.append("--infer-input-dim")
                run_cmd(prepare_cmd, cwd=root)

        run_summary[fold_name] = {
            "master_json": str(fold_master.resolve()),
            "task_a_dir": str(task_a_dir.resolve()),
            "task_b_dir": str(task_b_dir.resolve()),
            "repo_formats_dir": str(repo_dir.resolve()) if args.prepare_repo_formats else "",
            "prepared_actionformer": bool(args.prepare_actionformer),
            "actionformer_feature_extractor": args.actionformer_feature_extractor if args.prepare_actionformer else "",
        }

    payload = {
        "num_folds": args.num_folds,
        "output_dir": str(output_dir.resolve()),
        "kfold_stratify_level": args.kfold_stratify_level,
        "kfold_subject_leakage_policy": args.kfold_subject_leakage_policy,
        "kfold_imbalance_policy": args.kfold_imbalance_policy,
        "kfold_max_label_relative_deviation": args.kfold_max_label_relative_deviation,
        "kfold_imbalance_min_total": args.kfold_imbalance_min_total,
        "kfold_min_test_label_count_per_fold": args.kfold_min_test_label_count_per_fold,
        "kfold_min_count_policy": args.kfold_min_count_policy,
        "trim_clips": args.trim_clips,
        "prepare_repo_formats": args.prepare_repo_formats,
        "prepare_tsm": args.prepare_tsm,
        "prepare_actionformer": args.prepare_actionformer,
        "extract_actionformer_features": args.extract_actionformer_features,
        "actionformer_feature_extractor": args.actionformer_feature_extractor,
        "folds": run_summary,
    }
    (output_dir / "pipeline_run_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[DONE] k-fold pipeline completed.")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
