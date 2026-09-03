import copy
from typing import Dict, List


NUM_HAND_KPTS = 21


def _valid_hand_points(points: List[List[float]]) -> bool:
    if not isinstance(points, list) or len(points) < NUM_HAND_KPTS:
        return False
    for p in points[:NUM_HAND_KPTS]:
        if not isinstance(p, list) or len(p) < 3:
            return False
    return True


def _as_mano_points(points: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for p in points[:NUM_HAND_KPTS]:
        x = float(p[0]) if len(p) > 0 else 0.0
        y = float(p[1]) if len(p) > 1 else 0.0
        z = float(p[2]) if len(p) > 2 else 0.0
        c = float(p[3]) if len(p) > 3 else 1.0
        out.append([x, y, z, c])
    while len(out) < NUM_HAND_KPTS:
        out.append([0.0, 0.0, 0.0, 0.0])
    return out


def ensure_mano_sequence(sequence: Dict, allow_world_fallback: bool = False) -> Dict:
    """
    Ensure each frame has `left_mano` / `right_mano` fields.

    Priority:
    1. Use existing `left_mano` / `right_mano` if provided.
    2. If allow_world_fallback=True, copy from `*_world_3d` (or `*_3d`) as proxy.
    3. Otherwise raise.
    """
    frames = sequence.get("frames", [])
    if not isinstance(frames, list):
        raise RuntimeError("Invalid sequence: expected list in 'frames'.")

    out = copy.deepcopy(sequence)
    out_frames = out.get("frames", [])
    used_fallback = False

    for i, frame in enumerate(out_frames):
        if not isinstance(frame, dict):
            raise RuntimeError(f"Invalid frame at index {i}: expected object.")

        for hand in ("left", "right"):
            mano_key = f"{hand}_mano"
            if _valid_hand_points(frame.get(mano_key, [])):
                frame[mano_key] = _as_mano_points(frame.get(mano_key, []))
                continue

            if allow_world_fallback:
                src = frame.get(f"{hand}_world_3d", frame.get(f"{hand}_3d", []))
                if _valid_hand_points(src):
                    frame[mano_key] = _as_mano_points(src)
                    used_fallback = True
                    continue

            raise RuntimeError(
                f"MANO mode requires '{mano_key}' in every frame. "
                "Provide precomputed MANO joints, or enable world_3d fallback proxy."
            )

    meta = out.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    if used_fallback:
        meta["mano_source"] = "world_3d_proxy"
    else:
        meta["mano_source"] = "precomputed"
    out["meta"] = meta
    return out
