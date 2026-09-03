import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def infer_view(
    filename: str, manifest_view: str, source_locations: str
) -> Tuple[str, str, str]:
    """Return (view, evidence, confidence) using known source lineage only."""
    direct = manifest_view.strip().lower()
    if direct in {"view1", "view2"}:
        return direct, f"direct source path: {source_locations}", "high"

    if re.fullmatch(r"221210_\d{2}\.mp4", filename, flags=re.IGNORECASE):
        return (
            "view2",
            "edited sequence derives from the only 20221210 raw camera source: 20221210/view2/221210.mp4",
            "medium",
        )
    if re.fullmatch(r"IMG_642(?:1|3)_\d{2}\.mp4", filename, flags=re.IGNORECASE):
        return (
            "view2",
            "filename family IMG_6421/IMG_6423 is stored under 20230208/view2",
            "high",
        )
    if filename.lower() == "vid_20221215_192928.mp4":
        return (
            "view1",
            "raw counterpart is 20221215/view1/VID_20221215_192928_1.mp4",
            "high",
        )
    if re.fullmatch(r"VID_20230208_122207_1_\d{2}\.mp4", filename, flags=re.IGNORECASE):
        return (
            "view1",
            "segmented filename derives from 20230208/view1/VID_20230208_122207_1.mp4",
            "high",
        )
    return "unknown", "no deterministic source-lineage rule", "unresolved"


def _clock_seconds(hour: int, minute: int, second: int) -> int:
    return hour * 3600 + minute * 60 + second


def extract_video_clock_seconds(filename: str) -> Optional[int]:
    match = re.search(r"(?:VID_)?\d{8}_(\d{2})(\d{2})(\d{2})", filename, re.IGNORECASE)
    if not match:
        return None
    hour, minute, second = (int(value) for value in match.groups())
    if hour > 23 or minute > 59 or second > 59:
        return None
    return _clock_seconds(hour, minute, second)


def extract_sensor_clock_seconds(capture_id: str) -> Optional[int]:
    match = re.search(r"\](\d{2})-(\d{2})-(\d{2})$", capture_id)
    if not match:
        return None
    hour, minute, second = (int(value) for value in match.groups())
    if hour > 23 or minute > 59 or second > 59:
        return None
    return _clock_seconds(hour, minute, second)


def sensor_candidate_confidence(filename: str, delta_seconds: int) -> str:
    if re.search(r"_1_\d{2}\.mp4$", filename, re.IGNORECASE):
        return "manual_only_segmented_parent"
    if delta_seconds <= 10:
        return "high_candidate"
    if delta_seconds <= 60:
        return "medium_candidate"
    if delta_seconds <= 180:
        return "low_candidate"
    return "weak_candidate"


def build_sensor_candidates(
    video_rows: List[Dict[str, str]], sensor_rows: List[Dict[str, str]], top_k: int = 3
) -> List[Dict[str, Any]]:
    sensors_by_session: Dict[str, List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for sensor in sensor_rows:
        seconds = extract_sensor_clock_seconds(sensor.get("capture_id", ""))
        if seconds is not None:
            sensors_by_session[sensor.get("session", "")].append((seconds, sensor))

    candidates: List[Dict[str, Any]] = []
    for video in video_rows:
        video_seconds = extract_video_clock_seconds(video["filename"])
        if video_seconds is None:
            continue
        session = video["session_guess"]
        ranked = sorted(
            sensors_by_session.get(session, []),
            key=lambda item: (abs(item[0] - video_seconds), item[1]["capture_id"]),
        )[:top_k]
        for rank, (sensor_seconds, sensor) in enumerate(ranked, start=1):
            delta = abs(sensor_seconds - video_seconds)
            candidates.append(
                {
                    "video_id": video["video_id"],
                    "filename": video["filename"],
                    "session_id": session,
                    "candidate_rank": rank,
                    "sensor_capture_id": sensor["capture_id"],
                    "sensor_relative_path": sensor["relative_path"],
                    "absolute_clock_delta_seconds": delta,
                    "confidence": sensor_candidate_confidence(video["filename"], delta),
                    "decision": "review_required",
                    "warning": "clock proximity is not proof of synchronization",
                }
            )
    return candidates


def build_label_distribution(master: Dict[str, Any]) -> List[Dict[str, Any]]:
    aggregates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for video_id, record in master.get("videos", {}).items():
        for annotation in record.get("annotations", []):
            duration = max(
                0.0,
                float(annotation.get("end", 0)) - float(annotation.get("start", 0)),
            )
            for level, key in (("coarse", "coarse_label"), ("fine", "fine_label")):
                label = str(annotation.get(key, "")).strip()
                if not label:
                    continue
                item = aggregates.setdefault(
                    (level, label),
                    {
                        "level": level,
                        "label": label,
                        "segment_count": 0,
                        "video_ids": set(),
                        "seconds": 0.0,
                    },
                )
                item["segment_count"] += 1
                item["video_ids"].add(str(video_id))
                item["seconds"] += duration

    rows = []
    for key in sorted(aggregates):
        item = aggregates[key]
        rows.append(
            {
                "level": item["level"],
                "label": item["label"],
                "segment_count": item["segment_count"],
                "video_count": len(item["video_ids"]),
                "total_annotated_seconds": round(item["seconds"], 3),
            }
        )
    return rows

