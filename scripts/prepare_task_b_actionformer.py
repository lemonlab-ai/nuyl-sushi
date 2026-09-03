import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ACTIVITYNET_JSON = ROOT_DIR / "artifacts" / "task_b" / "activitynet_fine.json"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "artifacts" / "task_b" / "actionformer_fine.json"
DEFAULT_LABEL_MAP = ROOT_DIR / "artifacts" / "task_b" / "actionformer_label_map_fine.json"
DEFAULT_CONFIG_OUT = ROOT_DIR / "artifacts" / "task_b" / "actionformer_fine_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ActivityNet-style Task B JSON into ActionFormer-compatible JSON (with label_id)."
    )
    parser.add_argument("--activitynet-json", default=str(DEFAULT_ACTIVITYNET_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--label-map-out", default=str(DEFAULT_LABEL_MAP))
    parser.add_argument(
        "--config-out",
        default=str(DEFAULT_CONFIG_OUT),
        help="Optional output config template for actionformer_release.",
    )
    parser.add_argument("--feature-dir", default="", help="Feature folder path for ActionFormer config template.")
    parser.add_argument("--input-dim", type=int, default=2304, help="Feature dimension for ActionFormer config.")
    parser.add_argument(
        "--infer-input-dim",
        action="store_true",
        help="Infer input_dim from the first .npy in --feature-dir (or use feature_meta.json if present).",
    )
    parser.add_argument("--feat-stride", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--default-fps", type=float, default=30.0)
    parser.add_argument("--max-seq-len", type=int, default=2304)
    parser.add_argument(
        "--output-folder",
        default="",
        help="ActionFormer checkpoint root folder (defaults to <config_dir>/ckpt).",
    )
    return parser.parse_args()


def build_label_map(database: Dict[str, Dict]) -> Dict[str, int]:
    labels = set()
    for item in database.values():
        for ann in item.get("annotations", []):
            label = ann.get("label")
            if label is not None:
                labels.add(str(label))
    ordered = sorted(labels)
    return {name: idx for idx, name in enumerate(ordered)}


def convert_database(database: Dict[str, Dict], label_map: Dict[str, int]) -> Tuple[Dict[str, Dict], int]:
    converted = {}
    ann_count = 0
    for vid, item in database.items():
        anns = []
        for ann in item.get("annotations", []):
            label = str(ann["label"])
            anns.append(
                {
                    "segment": [float(ann["segment"][0]), float(ann["segment"][1])],
                    "label": label,
                    "label_id": int(label_map[label]),
                }
            )
            ann_count += 1
        converted[vid] = {
            "duration": float(item.get("duration", 0.0)),
            "subset": str(item.get("subset", "training")).lower(),
            "fps": float(item.get("fps", 0.0)),
            "source_path": item.get("source_path", ""),
            "annotations": anns,
        }
    return converted, ann_count


def write_config_template(
    config_out: Path,
    output_json: Path,
    feature_dir: Path,
    output_folder: Path,
    num_classes: int,
    input_dim: int,
    feat_stride: int,
    num_frames: int,
    default_fps: float,
    max_seq_len: int,
) -> None:
    config_out.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "dataset_name: anet\n"
        "train_split: [training]\n"
        "val_split: [validation]\n"
        "dataset:\n"
        f"  json_file: {output_json.resolve().as_posix()}\n"
        f"  feat_stride: {feat_stride}\n"
        f"  num_frames: {num_frames}\n"
        f"  default_fps: {default_fps}\n"
        "  downsample_rate: 1\n"
        f"  max_seq_len: {max_seq_len}\n"
        "  trunc_thresh: 0.5\n"
        "  crop_ratio: null\n"
        f"  input_dim: {input_dim}\n"
        f"  num_classes: {num_classes}\n"
        f"  feat_folder: {feature_dir.resolve().as_posix()}\n"
        "  file_prefix: null\n"
        "  file_ext: .npy\n"
        "  force_upsampling: false\n"
        "\n"
        "model_name: LocPointTransformer\n"
        "opt:\n"
        "  learning_rate: 0.001\n"
        "  epochs: 20\n"
        "  weight_decay: 0.05\n"
        "  warmup_epochs: 5\n"
        "loader:\n"
        "  batch_size: 4\n"
        "  num_workers: 4\n"
        f"output_folder: {output_folder.resolve().as_posix()}\n"
    )
    config_out.write_text(text, encoding="utf-8")


def infer_input_dim_from_features(feature_dir: Path, fallback: int) -> int:
    meta_json = feature_dir / "feature_meta.json"
    if meta_json.exists():
        try:
            payload = json.loads(meta_json.read_text(encoding="utf-8"))
            dim = int(payload.get("input_dim", 0))
            if dim > 0:
                return dim
        except Exception:
            pass

    npy_files = sorted(feature_dir.glob("*.npy"))
    if not npy_files:
        return fallback
    try:
        import numpy as np  # type: ignore
    except Exception:
        return fallback
    arr = np.load(npy_files[0])
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim >= 2:
        return int(arr.shape[-1])
    return fallback


def main() -> None:
    args = parse_args()
    src = Path(args.activitynet_json)
    out_json = Path(args.output_json)
    label_map_out = Path(args.label_map_out)
    config_out = Path(args.config_out)
    feature_dir = Path(args.feature_dir) if args.feature_dir else (ROOT_DIR / "artifacts" / "task_b" / "features")
    output_folder = Path(args.output_folder) if args.output_folder else (config_out.parent / "ckpt")

    if not src.exists():
        raise FileNotFoundError(src)
    payload = json.loads(src.read_text(encoding="utf-8"))
    database = payload.get("database", {})
    if not isinstance(database, dict):
        raise RuntimeError("Invalid ActivityNet JSON: missing database dict")

    label_map = build_label_map(database)
    converted_db, ann_count = convert_database(database, label_map)

    output_payload = {
        "version": payload.get("version", "1.0"),
        "task": "temporal_localization_actionformer",
        "label_map": label_map,
        "database": converted_db,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    label_map_out.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")
    input_dim = int(args.input_dim)
    if args.infer_input_dim:
        input_dim = infer_input_dim_from_features(feature_dir, fallback=input_dim)

    write_config_template(
        config_out=config_out,
        output_json=out_json,
        feature_dir=feature_dir,
        output_folder=output_folder,
        num_classes=len(label_map),
        input_dim=input_dim,
        feat_stride=args.feat_stride,
        num_frames=args.num_frames,
        default_fps=args.default_fps,
        max_seq_len=args.max_seq_len,
    )

    subset_counter: Dict[str, int] = {}
    for item in converted_db.values():
        subset = str(item["subset"])
        subset_counter[subset] = subset_counter.get(subset, 0) + 1
    stats = {
        "num_videos": len(converted_db),
        "num_annotations": ann_count,
        "num_classes": len(label_map),
        "input_dim": input_dim,
        "subset_counts": subset_counter,
        "output_json": str(out_json.resolve()),
        "label_map_out": str(label_map_out.resolve()),
        "config_out": str(config_out.resolve()),
    }
    print("[DONE] ActionFormer annotations prepared.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
