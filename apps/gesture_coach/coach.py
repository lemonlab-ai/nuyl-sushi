import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


NUM_HAND_KPTS = 21


def _normalize_hand_points(hand: List[List[float]], dims: int = 2) -> np.ndarray:
    # Internal layout is [x, y, z, conf], then projected to 2D/3D output.
    arr = np.zeros((NUM_HAND_KPTS, 4), dtype=np.float32)
    if not isinstance(hand, list):
        if int(dims) == 3:
            return np.zeros((NUM_HAND_KPTS, 4), dtype=np.float32)
        return np.zeros((NUM_HAND_KPTS, 3), dtype=np.float32)
    for i in range(min(NUM_HAND_KPTS, len(hand))):
        p = hand[i]
        if not isinstance(p, list) or len(p) < 2:
            continue
        x = float(p[0])
        y = float(p[1])
        z = 0.0
        c = 1.0
        if len(p) >= 4:
            z = float(p[2])
            c = float(p[3])
        elif len(p) == 3:
            # In 3D mode, [x, y, z] is allowed (conf defaults to 1.0).
            # In 2D mode, keep backward-compatible [x, y, conf].
            if int(dims) == 3:
                z = float(p[2])
                c = 1.0
            else:
                c = float(p[2])
        arr[i, 0] = x
        arr[i, 1] = y
        arr[i, 2] = z
        arr[i, 3] = c

    use_dims = 3 if int(dims) == 3 else 2
    wrist = arr[0, :use_dims]
    conf_idx = 3
    ref = arr[9, :use_dims] if float(arr[9, conf_idx]) > 0 else arr[5, :use_dims]
    scale = float(np.linalg.norm(ref - wrist))
    if scale < 1e-5:
        scale = 1.0
    centered = (arr[:, :use_dims] - wrist[None, :]) / scale
    centered *= arr[:, conf_idx : conf_idx + 1]
    return np.concatenate([centered, arr[:, conf_idx : conf_idx + 1]], axis=1)


def _pick_hand_key(frame: Dict, hand_name: str, coord_mode: str) -> List[List[float]]:
    mode = str(coord_mode).strip().lower()
    if mode == "3d_mano":
        return frame.get(f"{hand_name}_mano", frame.get(f"{hand_name}_world_3d", frame.get(f"{hand_name}_3d", [])))
    if mode == "3d_world":
        return frame.get(f"{hand_name}_world_3d", frame.get(f"{hand_name}_3d", frame.get(hand_name, [])))
    if mode == "3d_image":
        return frame.get(f"{hand_name}_3d", frame.get(hand_name, []))
    return frame.get(hand_name, [])


def _frame_to_vec(frame: Dict, use_left: bool, use_right: bool, coord_mode: str) -> np.ndarray:
    mode = str(coord_mode).strip().lower()
    use_3d = mode in {"3d_image", "3d_world", "3d_mano"}
    dims = 3 if use_3d else 2
    per_hand_dim = NUM_HAND_KPTS * (dims + 1)
    num_hands = int(use_left) + int(use_right)
    chunks: List[np.ndarray] = []
    if use_left:
        chunks.append(_normalize_hand_points(_pick_hand_key(frame, "left", mode), dims=dims))
    if use_right:
        chunks.append(_normalize_hand_points(_pick_hand_key(frame, "right", mode), dims=dims))
    if not chunks:
        return np.zeros((max(1, num_hands) * per_hand_dim,), dtype=np.float32)
    return np.concatenate([c.reshape(-1) for c in chunks], axis=0).astype(np.float32)


def _frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    valid_a = np.abs(a) > 1e-8
    valid_b = np.abs(b) > 1e-8
    valid = np.logical_or(valid_a, valid_b)
    if not np.any(valid):
        return 0.0
    diff = a[valid] - b[valid]
    return float(np.sqrt(np.mean(diff * diff)))


def _dtw(a: List[np.ndarray], b: List[np.ndarray]) -> Tuple[float, List[Tuple[int, int]]]:
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return float("inf"), []

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0
    back = np.zeros((n + 1, m + 1, 2), dtype=np.int32)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = _frame_distance(a[i - 1], b[j - 1])
            cands = [
                (cost[i - 1, j], (i - 1, j)),
                (cost[i, j - 1], (i, j - 1)),
                (cost[i - 1, j - 1], (i - 1, j - 1)),
            ]
            prev_val, prev_idx = min(cands, key=lambda x: float(x[0]))
            cost[i, j] = d + prev_val
            back[i, j, 0] = int(prev_idx[0])
            back[i, j, 1] = int(prev_idx[1])

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        ni = int(back[i, j, 0])
        nj = int(back[i, j, 1])
        if ni == i and nj == j:
            break
        i, j = ni, nj
    path.reverse()
    norm = float(max(len(path), 1))
    score = float(cost[n, m] / norm)
    return score, path


def compare_sequences(
    ref_seq: Dict,
    test_seq: Dict,
    hand_mode: str = "both",
    coord_mode: str = "2d",
    max_frames: int = 180,
) -> Dict:
    mode = str(hand_mode).strip().lower()
    use_left = mode in {"left", "both"}
    use_right = mode in {"right", "both"}
    if not use_left and not use_right:
        raise ValueError("hand_mode must be one of: left, right, both")
    c_mode = str(coord_mode).strip().lower()
    if c_mode not in {"2d", "3d_image", "3d_world", "3d_mano"}:
        raise ValueError("coord_mode must be one of: 2d, 3d_image, 3d_world, 3d_mano")

    ref_frames = ref_seq.get("frames", [])
    test_frames = test_seq.get("frames", [])
    if not isinstance(ref_frames, list) or not isinstance(test_frames, list):
        raise RuntimeError("Invalid sequence format: expected list in 'frames'.")

    if max_frames > 0:
        ref_frames = ref_frames[: int(max_frames)]
        test_frames = test_frames[: int(max_frames)]

    ref_vecs = [_frame_to_vec(f, use_left, use_right, c_mode) for f in ref_frames]
    test_vecs = [_frame_to_vec(f, use_left, use_right, c_mode) for f in test_frames]
    dtw_dist, path = _dtw(ref_vecs, test_vecs)
    if not math.isfinite(dtw_dist):
        raise RuntimeError("Empty sequence. Please provide non-empty keypoint sequence.")

    pair_distances: List[float] = []
    for i, j in path:
        pair_distances.append(_frame_distance(ref_vecs[i], test_vecs[j]))
    mean_dist = float(np.mean(pair_distances)) if pair_distances else 0.0
    p90_dist = float(np.quantile(np.asarray(pair_distances, dtype=np.float32), 0.9)) if pair_distances else 0.0

    # The scale is heuristic for normalized keypoint vectors.
    decay = 3.0 if c_mode == "2d" else 2.2
    score_0_100 = float(max(0.0, min(100.0, 100.0 * math.exp(-decay * dtw_dist))))

    return {
        "hand_mode": mode,
        "coord_mode": c_mode,
        "num_ref_frames": int(len(ref_vecs)),
        "num_test_frames": int(len(test_vecs)),
        "dtw_distance": float(dtw_dist),
        "mean_pair_distance": mean_dist,
        "p90_pair_distance": p90_dist,
        "score_0_100": score_0_100,
        "alignment_path": [{"ref_idx": int(i), "test_idx": int(j)} for i, j in path],
        "pair_distances": [float(x) for x in pair_distances],
    }


def load_sequence_json(path: Path) -> Dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid sequence JSON at {path}")
    if "frames" not in payload or not isinstance(payload["frames"], list):
        raise RuntimeError(f"Sequence JSON must contain a list field 'frames': {path}")
    return payload


def save_sequence_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def extract_keypoints_mediapipe(
    video_path: Path,
    max_frames: int = 300,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Dict:
    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe extraction requires 'mediapipe' and 'opencv-python'. "
            "Install mediapipe in your environment first."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=float(min_detection_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )

    frames: List[Dict] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        left = [[0.0, 0.0, 0.0] for _ in range(NUM_HAND_KPTS)]
        right = [[0.0, 0.0, 0.0] for _ in range(NUM_HAND_KPTS)]
        left_3d = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_HAND_KPTS)]
        right_3d = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_HAND_KPTS)]
        left_world_3d = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_HAND_KPTS)]
        right_world_3d = [[0.0, 0.0, 0.0, 0.0] for _ in range(NUM_HAND_KPTS)]

        if res.multi_hand_landmarks and res.multi_handedness:
            world_sets = []
            if res.multi_hand_world_landmarks:
                world_sets = list(res.multi_hand_world_landmarks)
            for hand_idx, (lm_set, handed) in enumerate(zip(res.multi_hand_landmarks, res.multi_handedness)):
                label = str(handed.classification[0].label).strip().lower()
                target_2d = left if label == "left" else right
                target_3d = left_3d if label == "left" else right_3d
                target_world_3d = left_world_3d if label == "left" else right_world_3d
                for k, lm in enumerate(lm_set.landmark[:NUM_HAND_KPTS]):
                    target_2d[k] = [float(lm.x), float(lm.y), 1.0]
                    target_3d[k] = [float(lm.x), float(lm.y), float(lm.z), 1.0]
                if hand_idx < len(world_sets):
                    world_lm_set = world_sets[hand_idx]
                    for k, lm_w in enumerate(world_lm_set.landmark[:NUM_HAND_KPTS]):
                        target_world_3d[k] = [float(lm_w.x), float(lm_w.y), float(lm_w.z), 1.0]

        frames.append(
            {
                "frame_idx": int(idx),
                "t_sec": float(idx / fps),
                "left": left,
                "right": right,
                "left_3d": left_3d,
                "right_3d": right_3d,
                "left_world_3d": left_world_3d,
                "right_world_3d": right_world_3d,
            }
        )
        idx += 1
        if max_frames > 0 and idx >= int(max_frames):
            break

    cap.release()
    hands.close()
    return {
        "source_video": str(video_path.resolve()),
        "fps": float(fps),
        "frames": frames,
        "num_frames": int(len(frames)),
    }
