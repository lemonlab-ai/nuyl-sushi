import json
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from scripts.train_multimodal_baseline import build_feature, softmax  # type: ignore # noqa: E402
except Exception:
    _module_path = ROOT_DIR / "scripts" / "train_multimodal_baseline.py"
    _spec = importlib.util.spec_from_file_location("train_multimodal_baseline", _module_path)
    if _spec is None or _spec.loader is None:
        raise RuntimeError(f"Cannot load module from {_module_path}")
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_feature = _mod.build_feature
    softmax = _mod.softmax


MODEL_CANDIDATES = [
    ROOT_DIR / "outputs" / "multimodal_fusion" / "model.npz",
    ROOT_DIR / "outputs" / "multimodal_baseline" / "model.npz",
    ROOT_DIR / "outputs" / "multimodal_smoke" / "model.npz",
]
DEFAULT_CLASS_MAP = ROOT_DIR / "artifacts" / "task_a" / "class_map_fine.json"
DEFAULT_GT_JSON = ROOT_DIR / "artifacts" / "task_b" / "activitynet_fine.json"
DEFAULT_LABEL_MAP = ROOT_DIR / "artifacts" / "task_b" / "actionformer_label_map_fine.json"
DEFAULT_ACTIONFORMER_REPO = ROOT_DIR / "external" / "actionformer_release"
DEFAULT_ACTIONFORMER_CONFIG = ROOT_DIR / "artifacts" / "task_b" / "actionformer_fine_config.yaml"


def choose_default_model_path() -> Path:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return MODEL_CANDIDATES[0]


def save_uploaded_file(uploaded_file, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix or ".bin"
    safe_name = Path(uploaded_file.name).stem.replace(" ", "_")
    out_path = output_dir / f"{safe_name}_{int(time.time() * 1000)}{suffix}"
    out_path.write_bytes(uploaded_file.getbuffer())
    return out_path


@st.cache_data(show_spinner=False)
def load_class_map(path_str: str) -> Dict[int, str]:
    path = Path(path_str)
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, str] = {}
    for label, idx in payload.items():
        out[int(idx)] = str(label)
    return out


@st.cache_resource(show_spinner=False)
def load_model_npz(path_str: str) -> Dict[str, np.ndarray]:
    path = Path(path_str)
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


_DEEPVIDEO_RUNTIME = None
_GESTURE_COACH_RUNTIME = None


def load_deepvideo_runtime():
    global _DEEPVIDEO_RUNTIME
    if _DEEPVIDEO_RUNTIME is not None:
        return _DEEPVIDEO_RUNTIME

    try:
        from scripts.extract_task_b_features_deep import load_backbone, resolve_device  # type: ignore
    except Exception:
        _mod_path = ROOT_DIR / "scripts" / "extract_task_b_features_deep.py"
        _spec = importlib.util.spec_from_file_location("extract_task_b_features_deep", _mod_path)
        if _spec is None or _spec.loader is None:
            raise RuntimeError(f"Cannot load deep feature module from {_mod_path}")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        load_backbone = _mod.load_backbone  # type: ignore
        resolve_device = _mod.resolve_device  # type: ignore

    try:
        from scripts.train_multimodal_fusion_deepvideo import build_deep_video_feature, build_sensor_feature  # type: ignore
    except Exception:
        _mod_path = ROOT_DIR / "scripts" / "train_multimodal_fusion_deepvideo.py"
        _spec = importlib.util.spec_from_file_location("train_multimodal_fusion_deepvideo", _mod_path)
        if _spec is None or _spec.loader is None:
            raise RuntimeError(f"Cannot load deepvideo fusion module from {_mod_path}")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        build_deep_video_feature = _mod.build_deep_video_feature  # type: ignore
        build_sensor_feature = _mod.build_sensor_feature  # type: ignore

    _DEEPVIDEO_RUNTIME = (load_backbone, resolve_device, build_deep_video_feature, build_sensor_feature)
    return _DEEPVIDEO_RUNTIME


def load_gesture_coach_runtime():
    global _GESTURE_COACH_RUNTIME
    if _GESTURE_COACH_RUNTIME is not None:
        return _GESTURE_COACH_RUNTIME

    try:
        from apps.gesture_coach.coach import (  # type: ignore
            compare_sequences,
            extract_keypoints_mediapipe,
            load_sequence_json,
            save_sequence_json,
        )
        from apps.gesture_coach.mano_adapter import ensure_mano_sequence  # type: ignore
    except Exception:
        _mod_path = ROOT_DIR / "apps" / "gesture_coach" / "coach.py"
        _spec = importlib.util.spec_from_file_location("gesture_coach", _mod_path)
        if _spec is None or _spec.loader is None:
            raise RuntimeError(f"Cannot load gesture coach module from {_mod_path}")
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        compare_sequences = _mod.compare_sequences  # type: ignore
        extract_keypoints_mediapipe = _mod.extract_keypoints_mediapipe  # type: ignore
        load_sequence_json = _mod.load_sequence_json  # type: ignore
        save_sequence_json = _mod.save_sequence_json  # type: ignore
        _mano_mod_path = ROOT_DIR / "apps" / "gesture_coach" / "mano_adapter.py"
        _mano_spec = importlib.util.spec_from_file_location("gesture_mano_adapter", _mano_mod_path)
        if _mano_spec is None or _mano_spec.loader is None:
            raise RuntimeError(f"Cannot load MANO adapter module from {_mano_mod_path}")
        _mano_mod = importlib.util.module_from_spec(_mano_spec)
        _mano_spec.loader.exec_module(_mano_mod)
        ensure_mano_sequence = _mano_mod.ensure_mano_sequence  # type: ignore

    _GESTURE_COACH_RUNTIME = (
        compare_sequences,
        extract_keypoints_mediapipe,
        load_sequence_json,
        save_sequence_json,
        ensure_mano_sequence,
    )
    return _GESTURE_COACH_RUNTIME


@st.cache_resource(show_spinner=False)
def load_deepvideo_backbone_cached(backbone: str, weights: str, device_pref: str):
    load_backbone, resolve_device, _, _ = load_deepvideo_runtime()
    device = str(resolve_device(device_pref))
    model = load_backbone(backbone, weights)
    model.eval()
    model.to(device)
    return model, device


def predict_task_a(
    model: Dict[str, np.ndarray],
    id_to_label: Dict[int, str],
    video_path: Path,
    gyro_csv_path: Optional[Path],
    start_sec: float,
    end_sec: float,
    weight_g: Optional[float],
    target_fps: float,
    max_frames: int,
    top_k: int,
) -> Dict:
    def read_npz_bool(name: str, default: bool = True) -> bool:
        arr = model.get(name)
        if arr is None:
            return default
        try:
            return bool(int(np.asarray(arr).reshape(-1)[0]))
        except Exception:
            return default

    def read_npz_int(name: str, default: int) -> int:
        arr = model.get(name)
        if arr is None:
            return int(default)
        try:
            return int(np.asarray(arr).reshape(-1)[0])
        except Exception:
            return int(default)

    def read_npz_float(name: str, default: float) -> float:
        arr = model.get(name)
        if arr is None:
            return float(default)
        try:
            return float(np.asarray(arr).reshape(-1)[0])
        except Exception:
            return float(default)

    def read_npz_str(name: str, default: str) -> str:
        arr = model.get(name)
        if arr is None:
            return str(default)
        try:
            return str(np.asarray(arr).reshape(-1)[0])
        except Exception:
            return str(default)

    use_weight = read_npz_bool("use_weight", True)
    use_gyro = read_npz_bool("use_gyro", True)
    model_type_hint = read_npz_str("model_type", "")

    video_id = "streamlit_upload"
    row = {
        "video_id": video_id,
        "video_path": str(video_path),
        "start_sec": str(float(start_sec)),
        "end_sec": str(float(end_sec)),
        "weight_g": "" if (weight_g is None or not use_weight) else str(float(weight_g)),
        "has_gyro": "true" if (gyro_csv_path and use_gyro) else "false",
    }
    master_index = {}
    if gyro_csv_path is not None and use_gyro:
        master_index[video_id] = {"has_gyro": True, "gyro_path": str(gyro_csv_path)}
    mean = model["mean"].astype(np.float32)
    std = model["std"].astype(np.float32)

    if mean.ndim == 2:
        mean = mean[0]
    if std.ndim == 2:
        std = std[0]
    if model_type_hint == "fusion_mlp_deepvideo":
        expected_dim = int(model["w1"].shape[0])
        model_type = "fusion_mlp_deepvideo"
    elif "w1" in model:
        expected_dim = int(model["w1"].shape[0])
        model_type = "fusion_mlp"
    elif "w" in model:
        expected_dim = int(model["w"].shape[0])
        model_type = "linear_softmax"
    else:
        raise RuntimeError("Unsupported model.npz format: expected keys w/b or w1/b1/w2/b2.")

    if model_type == "fusion_mlp_deepvideo":
        _, _, build_deep_video_feature, build_sensor_feature = load_deepvideo_runtime()
        video_backbone = read_npz_str("video_backbone", "r3d_18")
        video_weights = read_npz_str("video_weights", "kinetics400")
        video_device = read_npz_str("video_device", "auto")
        video_model, resolved_device = load_deepvideo_backbone_cached(video_backbone, video_weights, video_device)
        video_dim = read_npz_int("video_feature_dim", max(1, expected_dim - 9))

        deep_target_fps = read_npz_float("target_fps", target_fps)
        deep_clip_len = read_npz_int("clip_len", 16)
        deep_clip_hop = read_npz_int("clip_hop", 8)
        deep_resize_short = read_npz_int("resize_short", 128)
        deep_crop_size = read_npz_int("crop_size", 112)
        deep_batch_size = read_npz_int("batch_size", 8)
        deep_max_sampled_frames = read_npz_int("max_sampled_frames", max_frames)

        video_feat = build_deep_video_feature(
            row=row,
            model=video_model,
            device=resolved_device,
            target_fps=deep_target_fps,
            clip_len=deep_clip_len,
            clip_hop=deep_clip_hop,
            resize_short=deep_resize_short,
            crop_size=deep_crop_size,
            batch_size=deep_batch_size,
            max_sampled_frames=deep_max_sampled_frames,
            video_dim=video_dim,
        ).astype(np.float32)
        sensor_feat = build_sensor_feature(
            row=row,
            master_index=master_index,
            use_weight=use_weight,
            use_gyro=use_gyro,
        ).astype(np.float32)
        feat = np.concatenate([video_feat, sensor_feat], axis=0).astype(np.float32)
    else:
        feat = build_feature(
            row,
            master_index,
            target_fps=target_fps,
            max_frames=max_frames,
            use_weight=use_weight,
            use_gyro=use_gyro,
        ).astype(np.float32)

    if feat.shape[0] != expected_dim:
        raise RuntimeError(
            f"Feature dim mismatch: extracted={feat.shape[0]}, model_expected={expected_dim}. "
            "Please use a matching model.npz."
        )

    x_norm = (feat[None, :] - mean[None, :]) / np.where(std[None, :] < 1e-6, 1.0, std[None, :])
    use_reliability_gating = read_npz_bool("use_reliability_gating", False)
    if use_reliability_gating and ("wg" in model) and ("bg" in model):
        def sigmoid(arr: np.ndarray) -> np.ndarray:
            arr_clip = np.clip(arr, -30.0, 30.0)
            return 1.0 / (1.0 + np.exp(-arr_clip))

        if model_type == "fusion_mlp_deepvideo":
            video_dim = read_npz_int("video_feature_dim", max(1, expected_dim - 9))
            video_slice = slice(0, video_dim)
            weight_slice = slice(video_dim, video_dim + 2)
            gyro_slice = slice(video_dim + 2, expected_dim)
        else:
            video_slice = slice(0, 13)
            weight_slice = slice(13, 15)
            gyro_slice = slice(15, 22)

        q_video = 1.0 if float(np.linalg.norm(feat[video_slice])) > 1e-8 else 0.0
        q_weight = 1.0 if float(feat[weight_slice.start + 1]) > 0.5 else 0.0
        q_gyro = 1.0 if float(feat[gyro_slice.stop - 1]) > 0.5 else 0.0
        q = np.array([[q_video, q_weight, q_gyro]], dtype=np.float32)
        wg = model["wg"].astype(np.float32)
        bg = model["bg"].astype(np.float32).reshape(-1)
        gates = sigmoid(q @ wg + bg[None, :]).astype(np.float32)
        x_norm[:, video_slice] *= gates[:, 0:1]
        x_norm[:, weight_slice] *= gates[:, 1:2]
        x_norm[:, gyro_slice] *= gates[:, 2:3]

    if "w1" in model:
        w1 = model["w1"].astype(np.float32)
        b1 = model["b1"].astype(np.float32)
        w2 = model["w2"].astype(np.float32)
        b2 = model["b2"].astype(np.float32)
        h = np.maximum(x_norm @ w1 + b1[None, :], 0.0)
        probs = softmax(h @ w2 + b2[None, :])[0]
    else:
        w = model["w"].astype(np.float32)
        b = model["b"].astype(np.float32)
        probs = softmax(x_norm @ w + b[None, :])[0]
    sorted_idx = np.argsort(probs)[::-1][: max(1, top_k)]

    preds: List[Dict] = []
    for idx in sorted_idx:
        label_id = int(idx)
        preds.append(
            {
                "rank": len(preds) + 1,
                "label_id": label_id,
                "label": id_to_label.get(label_id, str(label_id)),
                "probability": float(probs[label_id]),
            }
        )
    return {
        "predictions": preds,
        "feature_dim": int(feat.shape[0]),
        "model_type": model_type,
        "use_weight": bool(use_weight),
        "use_gyro": bool(use_gyro),
    }


def run_task_b_eval(
    pred_file: Path,
    gt_json: Path,
    label_map: Optional[Path],
    subset: str,
    iou_thresholds: str,
    work_dir: Path,
) -> Dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    pred_eval_json = work_dir / "pred_eval.json"
    metrics_json = work_dir / "metrics.json"

    convert_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "convert_task_b_predictions.py"),
        "--input",
        str(pred_file),
        "--output-json",
        str(pred_eval_json),
    ]
    if label_map is not None and label_map.exists():
        convert_cmd.extend(["--label-map", str(label_map)])

    convert_run = subprocess.run(convert_cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if convert_run.returncode != 0:
        raise RuntimeError(convert_run.stderr.strip() or convert_run.stdout.strip() or "convert_task_b_predictions failed")

    eval_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "eval_task_b.py"),
        "--gt-json",
        str(gt_json),
        "--pred-json",
        str(pred_eval_json),
        "--subset",
        subset,
        "--iou-thresholds",
        iou_thresholds,
        "--output-json",
        str(metrics_json),
    ]
    eval_run = subprocess.run(eval_cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    if eval_run.returncode != 0:
        raise RuntimeError(eval_run.stderr.strip() or eval_run.stdout.strip() or "eval_task_b failed")

    return json.loads(metrics_json.read_text(encoding="utf-8"))


def run_cmd_or_raise(cmd: List[str], cwd: Path, err_prefix: str) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{err_prefix}\n{result.stdout}\n{result.stderr}".strip())


def probe_video_metadata(video_path: Path) -> Dict[str, float]:
    try:
        import cv2  # type: ignore
    except Exception:
        return {"fps": 30.0, "duration": 10.0}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": 30.0, "duration": 10.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    nframes = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    duration = nframes / fps if (fps > 0 and nframes > 0) else 10.0
    return {"fps": float(fps), "duration": float(max(duration, 0.2))}


def load_label_map_raw(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, int] = {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid label_map JSON: {path}")
    for label, idx in payload.items():
        out[str(label)] = int(idx)
    if not out:
        raise RuntimeError(f"Empty label_map JSON: {path}")
    return out


def safe_video_id(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
    return cleaned or f"video_{int(time.time())}"


def run_actionformer_single_video(
    video_path: Path,
    actionformer_repo: Path,
    base_config: Path,
    ckpt_path: Path,
    label_map_path: Path,
    work_dir: Path,
    target_fps: float,
    window_size: int,
    window_hop: int,
    max_sampled_frames: int,
    eval_topk: int,
    eval_print_freq: int,
) -> Dict:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to run ActionFormer inference.") from exc

    if not actionformer_repo.exists():
        raise FileNotFoundError(f"ActionFormer repo not found: {actionformer_repo}")
    if not base_config.exists():
        raise FileNotFoundError(f"ActionFormer config not found: {base_config}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video_metadata(video_path)
    fps = float(metadata["fps"])
    duration = float(metadata["duration"])
    vid = safe_video_id(video_path.stem)

    label_map = load_label_map_raw(label_map_path)
    dummy_end = max(0.05, min(0.2, duration))
    dummy_annotations = [
        {"segment": [0.0, dummy_end], "label": label, "label_id": int(label_id)}
        for label, label_id in sorted(label_map.items(), key=lambda kv: kv[1])
    ]
    actionformer_json = work_dir / "actionformer_single.json"
    payload = {
        "version": "1.0",
        "task": "temporal_localization_actionformer",
        "label_map": label_map,
        "database": {
            vid: {
                "duration": duration,
                "subset": "validation",
                "fps": fps,
                "source_path": str(video_path.resolve()),
                "annotations": dummy_annotations,
            }
        },
    }
    actionformer_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    feature_dir = work_dir / "features_basic"
    extract_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "extract_task_b_features_basic.py"),
        "--gt-json",
        str(actionformer_json),
        "--output-dir",
        str(feature_dir),
        "--target-fps",
        str(float(target_fps)),
        "--window-size",
        str(int(window_size)),
        "--window-hop",
        str(int(window_hop)),
        "--max-sampled-frames",
        str(int(max_sampled_frames)),
    ]
    run_cmd_or_raise(extract_cmd, ROOT_DIR, "Feature extraction failed")

    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid ActionFormer config YAML.")
    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    dataset_cfg["json_file"] = str(actionformer_json.resolve().as_posix())
    dataset_cfg["feat_folder"] = str(feature_dir.resolve().as_posix())
    dataset_cfg["file_prefix"] = None
    dataset_cfg["file_ext"] = ".npy"
    dataset_cfg["input_dim"] = 13
    dataset_cfg["num_classes"] = len(label_map)
    dataset_cfg["default_fps"] = fps
    cfg["dataset"] = dataset_cfg
    cfg["dataset_name"] = "anet"
    cfg["train_split"] = ["training"]
    cfg["val_split"] = ["validation"]
    cfg["output_folder"] = str((work_dir / "ckpt_tmp").resolve().as_posix())
    eval_cfg_path = work_dir / "actionformer_eval_config.yaml"
    eval_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    eval_cmd = [
        sys.executable,
        str(actionformer_repo / "eval.py"),
        str(eval_cfg_path),
        str(ckpt_path.resolve()),
        "--saveonly",
        "-p",
        str(int(eval_print_freq)),
    ]
    if int(eval_topk) > 0:
        eval_cmd.extend(["-t", str(int(eval_topk))])
    run_cmd_or_raise(eval_cmd, actionformer_repo, "ActionFormer eval failed")

    if ckpt_path.is_file():
        raw_pred = ckpt_path.parent / "eval_results.pkl"
    else:
        raw_pred = ckpt_path / "eval_results.pkl"
    if not raw_pred.exists():
        raise FileNotFoundError(f"eval_results.pkl not found after eval: {raw_pred}")

    pred_eval_json = work_dir / "pred_eval.json"
    convert_cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "convert_task_b_predictions.py"),
        "--input",
        str(raw_pred),
        "--format",
        "pkl",
        "--label-map",
        str(label_map_path.resolve()),
        "--output-json",
        str(pred_eval_json),
    ]
    run_cmd_or_raise(convert_cmd, ROOT_DIR, "Prediction conversion failed")

    pred_payload = json.loads(pred_eval_json.read_text(encoding="utf-8"))
    per_video = pred_payload.get("results", {}).get(vid, [])
    return {
        "video_id": vid,
        "predictions": per_video,
        "pred_json_path": str(pred_eval_json.resolve()),
        "raw_pkl_path": str(raw_pred.resolve()),
        "payload": pred_payload,
    }


def main() -> None:
    st.set_page_config(page_title="NTNU Sushi Demo", layout="wide")
    st.title("NTNU Sushi: Streamlit Frontend")
    st.caption("Task A inference + Task B evaluation UI (single-machine integrated demo).")

    with st.sidebar:
        st.subheader("Task A Model")
        model_path = st.text_input("model.npz path", str(choose_default_model_path()))
        class_map_path = st.text_input("class_map path", str(DEFAULT_CLASS_MAP))
        target_fps = st.number_input("target_fps", min_value=0.5, max_value=30.0, value=3.0, step=0.5)
        max_frames = st.number_input("max_frames", min_value=8, max_value=512, value=64, step=8)
        top_k = st.slider("top_k", min_value=1, max_value=10, value=3)

    tab_a, tab_b, tab_c = st.tabs(["Task A Inference", "Task B Eval", "Gesture Coach (MVP)"])

    with tab_a:
        st.subheader("Task A: Multimodal Inference (Fusion/Linear)")
        video_upload = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="taska_video")
        gyro_upload = st.file_uploader("Upload gyro CSV (optional)", type=["csv", "txt"], key="taska_gyro")
        col1, col2, col3 = st.columns(3)
        with col1:
            start_sec = st.number_input("start_sec", min_value=0.0, value=0.0, step=0.1)
        with col2:
            end_sec = st.number_input("end_sec (0 means full video)", min_value=0.0, value=0.0, step=0.1)
        with col3:
            use_weight = st.checkbox("Use weight_g", value=False)
            weight_g = st.number_input("weight_g", value=0.0, step=1.0, disabled=not use_weight)

        if st.button("Run Task A Prediction", type="primary"):
            if video_upload is None:
                st.error("Please upload a video.")
            else:
                try:
                    model = load_model_npz(model_path)
                    id_to_label = load_class_map(class_map_path)
                    upload_dir = ROOT_DIR / "artifacts" / "demo_streamlit" / "uploads"
                    video_path = save_uploaded_file(video_upload, upload_dir)
                    gyro_path = save_uploaded_file(gyro_upload, upload_dir) if gyro_upload is not None else None
                    with st.spinner("Running feature extraction + prediction..."):
                        result = predict_task_a(
                            model=model,
                            id_to_label=id_to_label,
                            video_path=video_path,
                            gyro_csv_path=gyro_path,
                            start_sec=float(start_sec),
                            end_sec=float(end_sec),
                            weight_g=float(weight_g) if use_weight else None,
                            target_fps=float(target_fps),
                            max_frames=int(max_frames),
                            top_k=int(top_k),
                        )
                    preds = result["predictions"]
                    st.success(f"Top-1: {preds[0]['label']} (p={preds[0]['probability']:.4f})")
                    st.dataframe(preds, use_container_width=True)
                    st.caption(
                        "Model: "
                        f"{result['model_type']} | Feature dim: {result['feature_dim']} | "
                        f"use_weight={result['use_weight']} | use_gyro={result['use_gyro']} | "
                        f"saved input: {video_path}"
                    )
                except Exception as exc:
                    st.error(str(exc))

    with tab_b:
        st.subheader("Task B: Prediction Convert + Eval")
        pred_upload = st.file_uploader("Upload Task B prediction (.json/.csv/.pkl)", type=["json", "csv", "pkl"])
        gt_json_path = st.text_input("GT json path", str(DEFAULT_GT_JSON))
        label_map_path = st.text_input("label_map path (optional)", str(DEFAULT_LABEL_MAP))
        col1, col2 = st.columns(2)
        with col1:
            subset = st.selectbox("subset", ["all", "validation", "testing", "training"], index=0)
        with col2:
            iou_text = st.text_input("IoU thresholds", "0.3,0.5,0.75")

        if st.button("Run Task B Eval", type="primary"):
            if pred_upload is None:
                st.error("Please upload a prediction file.")
            else:
                gt_json = Path(gt_json_path)
                if not gt_json.exists():
                    st.error(f"GT json not found: {gt_json}")
                else:
                    try:
                        eval_dir = ROOT_DIR / "artifacts" / "demo_streamlit" / "task_b_eval"
                        pred_path = save_uploaded_file(pred_upload, eval_dir)
                        lm = Path(label_map_path) if label_map_path.strip() else None
                        with st.spinner("Converting + evaluating..."):
                            metrics = run_task_b_eval(
                                pred_file=pred_path,
                                gt_json=gt_json,
                                label_map=lm,
                                subset=subset,
                                iou_thresholds=iou_text,
                                work_dir=eval_dir,
                            )
                        st.success(f"average_mAP: {metrics.get('average_mAP', 0.0):.6f}")
                        st.json(metrics)
                    except Exception as exc:
                        st.error(str(exc))

        st.divider()
        st.subheader("Task B: Run ActionFormer On Uploaded Video")
        st.caption("Requires a trained ActionFormer checkpoint and CUDA-capable torch environment.")
        actionformer_video = st.file_uploader(
            "Upload video for ActionFormer inference", type=["mp4", "mov", "avi", "mkv"], key="taskb_video_infer"
        )
        af_repo_path = st.text_input("ActionFormer repo path", str(DEFAULT_ACTIONFORMER_REPO))
        af_cfg_path = st.text_input("ActionFormer base config path", str(DEFAULT_ACTIONFORMER_CONFIG))
        af_ckpt_path = st.text_input("ActionFormer ckpt dir/file path", "")
        af_label_map_path = st.text_input("ActionFormer label_map path", str(DEFAULT_LABEL_MAP))
        col3, col4, col5 = st.columns(3)
        with col3:
            af_target_fps = st.number_input("AF feature target_fps", min_value=0.5, max_value=30.0, value=2.0, step=0.5)
            af_window_size = st.number_input("AF feature window_size", min_value=2, max_value=128, value=16, step=2)
        with col4:
            af_window_hop = st.number_input("AF feature window_hop", min_value=1, max_value=128, value=8, step=1)
            af_max_sampled = st.number_input("AF feature max_sampled_frames (0=no cap)", min_value=0, max_value=10000, value=0, step=10)
        with col5:
            af_eval_topk = st.number_input("AF eval topk (-1=default)", min_value=-1, max_value=10000, value=-1, step=1)
            af_eval_print_freq = st.number_input("AF eval print_freq", min_value=1, max_value=10000, value=10, step=1)

        if st.button("Run ActionFormer Inference", type="primary"):
            if actionformer_video is None:
                st.error("Please upload a video for ActionFormer inference.")
            elif not af_ckpt_path.strip():
                st.error("Please provide ActionFormer checkpoint path.")
            else:
                try:
                    ts = int(time.time() * 1000)
                    work_dir = ROOT_DIR / "artifacts" / "demo_streamlit" / "task_b_infer" / f"run_{ts}"
                    upload_dir = work_dir / "uploads"
                    video_path = save_uploaded_file(actionformer_video, upload_dir)

                    with st.spinner("Extracting features + running ActionFormer eval..."):
                        out = run_actionformer_single_video(
                            video_path=video_path,
                            actionformer_repo=Path(af_repo_path).expanduser(),
                            base_config=Path(af_cfg_path).expanduser(),
                            ckpt_path=Path(af_ckpt_path).expanduser(),
                            label_map_path=Path(af_label_map_path).expanduser(),
                            work_dir=work_dir,
                            target_fps=float(af_target_fps),
                            window_size=int(af_window_size),
                            window_hop=int(af_window_hop),
                            max_sampled_frames=int(af_max_sampled),
                            eval_topk=int(af_eval_topk),
                            eval_print_freq=int(af_eval_print_freq),
                        )

                    preds = out["predictions"]
                    st.success(f"Inference completed. video_id={out['video_id']} preds={len(preds)}")
                    if preds:
                        rows = []
                        for i, p in enumerate(sorted(preds, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:100], start=1):
                            seg = p.get("segment", [0.0, 0.0])
                            rows.append(
                                {
                                    "rank": i,
                                    "label": p.get("label", ""),
                                    "score": float(p.get("score", 0.0)),
                                    "start": float(seg[0]) if isinstance(seg, list) and len(seg) > 1 else 0.0,
                                    "end": float(seg[1]) if isinstance(seg, list) and len(seg) > 1 else 0.0,
                                }
                            )
                        st.dataframe(rows, use_container_width=True)
                    st.caption(f"pred_json: {out['pred_json_path']}")
                    st.caption(f"raw_pkl: {out['raw_pkl_path']}")
                    payload_text = json.dumps(out["payload"], indent=2, ensure_ascii=False)
                    st.download_button(
                        "Download pred_eval.json",
                        data=payload_text,
                        file_name="pred_eval.json",
                        mime="application/json",
                    )
                except Exception as exc:
                    st.error(str(exc))

    with tab_c:
        st.subheader("Gesture Coach (MVP)")
        st.caption(
            "MVP flow: (1) compare two keypoint JSON files, or (2) extract MediaPipe 2D/3D hand keypoints and compare."
        )
        compare_sequences, extract_keypoints_mediapipe, load_sequence_json, save_sequence_json, ensure_mano_sequence = load_gesture_coach_runtime()
        mode = st.radio(
            "Input mode",
            ["Upload keypoint JSON", "Extract from videos (MediaPipe)"],
            horizontal=True,
        )
        col_mode_a, col_mode_b = st.columns(2)
        with col_mode_a:
            hand_mode = st.selectbox("Hand mode", ["both", "left", "right"], index=0)
        with col_mode_b:
            coord_mode_label = st.selectbox(
                "Coordinate mode",
                ["2D (x,y)", "3D image (x,y,z)", "3D world (x,y,z)", "3D MANO-compatible joints (x,y,z)"],
                index=0,
            )
        coord_mode_map = {
            "2D (x,y)": "2d",
            "3D image (x,y,z)": "3d_image",
            "3D world (x,y,z)": "3d_world",
            "3D MANO-compatible joints (x,y,z)": "3d_mano",
        }
        coord_mode = coord_mode_map.get(coord_mode_label, "2d")
        st.caption("3D JSON points suggest format `[x, y, z, conf]`. Legacy 2D format `[x, y, conf]` is still supported.")
        allow_mano_proxy = False
        if coord_mode == "3d_mano":
            allow_mano_proxy = st.checkbox(
                "Allow MANO proxy from world_3d if MANO joints missing",
                value=True,
                help=(
                    "If checked, `left_world_3d/right_world_3d` are copied to "
                    "`left_mano/right_mano` as a MANO-compatible fallback proxy."
                ),
            )
        coach_max_frames = st.number_input(
            "max_frames_for_compare",
            min_value=30,
            max_value=2000,
            value=300,
            step=30,
        )

        coach_dir = ROOT_DIR / "artifacts" / "demo_streamlit" / "gesture_coach"
        coach_dir.mkdir(parents=True, exist_ok=True)

        if mode == "Upload keypoint JSON":
            ref_json_upload = st.file_uploader(
                "Reference keypoint JSON",
                type=["json"],
                key="coach_ref_json",
            )
            test_json_upload = st.file_uploader(
                "Test keypoint JSON",
                type=["json"],
                key="coach_test_json",
            )
            if st.button("Run Gesture Compare", type="primary", key="coach_compare_json_btn"):
                if ref_json_upload is None or test_json_upload is None:
                    st.error("Please upload both reference and test keypoint JSON files.")
                else:
                    try:
                        ref_path = save_uploaded_file(ref_json_upload, coach_dir / "uploads")
                        test_path = save_uploaded_file(test_json_upload, coach_dir / "uploads")
                        ref_seq = load_sequence_json(ref_path)
                        test_seq = load_sequence_json(test_path)
                        if coord_mode == "3d_mano":
                            ref_seq = ensure_mano_sequence(ref_seq, allow_world_fallback=bool(allow_mano_proxy))
                            test_seq = ensure_mano_sequence(test_seq, allow_world_fallback=bool(allow_mano_proxy))
                        result = compare_sequences(
                            ref_seq=ref_seq,
                            test_seq=test_seq,
                            hand_mode=hand_mode,
                            coord_mode=coord_mode,
                            max_frames=int(coach_max_frames),
                        )
                        st.metric("Gesture similarity (0-100)", f"{result['score_0_100']:.2f}")
                        st.caption(
                            f"mode={result['coord_mode']} | "
                            f"DTW distance={result['dtw_distance']:.6f} | "
                            f"mean pair dist={result['mean_pair_distance']:.6f} | "
                            f"p90 pair dist={result['p90_pair_distance']:.6f}"
                        )
                        if result.get("pair_distances"):
                            st.line_chart(result["pair_distances"])
                        st.json(
                            {
                                "num_ref_frames": result["num_ref_frames"],
                                "num_test_frames": result["num_test_frames"],
                                "hand_mode": result["hand_mode"],
                                "coord_mode": result["coord_mode"],
                                "alignment_pairs": len(result.get("alignment_path", [])),
                            }
                        )
                    except Exception as exc:
                        st.error(str(exc))
        else:
            ref_video_upload = st.file_uploader(
                "Reference video",
                type=["mp4", "mov", "avi", "mkv"],
                key="coach_ref_video",
            )
            test_video_upload = st.file_uploader(
                "Test video",
                type=["mp4", "mov", "avi", "mkv"],
                key="coach_test_video",
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                extract_max_frames = st.number_input(
                    "extract_max_frames",
                    min_value=30,
                    max_value=5000,
                    value=300,
                    step=30,
                )
            with col2:
                det_conf = st.number_input(
                    "min_detection_confidence",
                    min_value=0.1,
                    max_value=0.99,
                    value=0.5,
                    step=0.05,
                )
            with col3:
                trk_conf = st.number_input(
                    "min_tracking_confidence",
                    min_value=0.1,
                    max_value=0.99,
                    value=0.5,
                    step=0.05,
                )

            if st.button("Extract + Compare", type="primary", key="coach_extract_compare_btn"):
                if ref_video_upload is None or test_video_upload is None:
                    st.error("Please upload both reference and test videos.")
                else:
                    try:
                        ts = int(time.time() * 1000)
                        run_dir = coach_dir / f"run_{ts}"
                        upload_dir = run_dir / "uploads"
                        ref_video_path = save_uploaded_file(ref_video_upload, upload_dir)
                        test_video_path = save_uploaded_file(test_video_upload, upload_dir)
                        with st.spinner("Extracting keypoints (MediaPipe 2D/3D) and comparing..."):
                            ref_seq = extract_keypoints_mediapipe(
                                video_path=ref_video_path,
                                max_frames=int(extract_max_frames),
                                min_detection_confidence=float(det_conf),
                                min_tracking_confidence=float(trk_conf),
                            )
                            test_seq = extract_keypoints_mediapipe(
                                video_path=test_video_path,
                                max_frames=int(extract_max_frames),
                                min_detection_confidence=float(det_conf),
                                min_tracking_confidence=float(trk_conf),
                            )
                            if coord_mode == "3d_mano":
                                ref_seq = ensure_mano_sequence(ref_seq, allow_world_fallback=bool(allow_mano_proxy))
                                test_seq = ensure_mano_sequence(test_seq, allow_world_fallback=bool(allow_mano_proxy))
                            ref_json_path = run_dir / "ref_keypoints.json"
                            test_json_path = run_dir / "test_keypoints.json"
                            save_sequence_json(ref_json_path, ref_seq)
                            save_sequence_json(test_json_path, test_seq)
                            result = compare_sequences(
                                ref_seq=ref_seq,
                                test_seq=test_seq,
                                hand_mode=hand_mode,
                                coord_mode=coord_mode,
                                max_frames=int(coach_max_frames),
                            )

                        st.metric("Gesture similarity (0-100)", f"{result['score_0_100']:.2f}")
                        st.caption(
                            f"mode={result['coord_mode']} | "
                            f"DTW distance={result['dtw_distance']:.6f} | "
                            f"ref_frames={result['num_ref_frames']} | test_frames={result['num_test_frames']}"
                        )
                        if result.get("pair_distances"):
                            st.line_chart(result["pair_distances"])

                        ref_json_text = json.dumps(ref_seq, indent=2, ensure_ascii=False)
                        test_json_text = json.dumps(test_seq, indent=2, ensure_ascii=False)
                        st.download_button(
                            "Download ref_keypoints.json",
                            data=ref_json_text,
                            file_name="ref_keypoints.json",
                            mime="application/json",
                        )
                        st.download_button(
                            "Download test_keypoints.json",
                            data=test_json_text,
                            file_name="test_keypoints.json",
                            mime="application/json",
                        )
                    except Exception as exc:
                        st.error(str(exc))
                        st.info(
                            "If MediaPipe is missing, install it in your env: "
                            "`pip install mediapipe` (or conda equivalent)."
                        )


if __name__ == "__main__":
    main()
