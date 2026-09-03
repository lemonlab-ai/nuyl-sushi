import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from nuyl_sushi.data.via import (
    extract_fine_label,
    extract_segment,
    infer_parent_label,
    parse_jsonish,
    parse_via_csv,
)
from nuyl_sushi.domain.models import MasterDataset, SegmentAnnotation, VideoRecord


SKILL_LEVEL_ALIASES = {
    "beginner": "beginner",
    "novice": "beginner",
    "newbie": "beginner",
    "entry": "beginner",
    "entry_level": "beginner",
    "junior": "beginner",
    "intermediate": "intermediate",
    "advanced": "intermediate",
    "mid": "intermediate",
    "mid_level": "intermediate",
    "expert": "expert",
    "senior": "expert",
    "master": "expert",
    "pro": "expert",
}


def load_optional_meta(path: Optional[Path | str]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and isinstance(payload.get("videos"), dict):
        return payload["videos"]
    if isinstance(payload, dict):
        return payload
    return {}


def normalize_skill_level(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "unknown"
    canonical = text.replace("-", "_").replace(" ", "_")
    return SKILL_LEVEL_ALIASES.get(canonical, canonical)


def probe_video(video_path: Path) -> Dict[str, Any]:
    try:
        import cv2  # type: ignore
    except Exception:
        return {}
    if not video_path.exists():
        return {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return {
        "fps": fps,
        "num_frames": frames,
        "duration": (frames / fps) if fps > 0 else 0.0,
    }


def assign_subsets(
    videos: Dict[str, VideoRecord],
    split_by: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    fixed_ids = [key for key, record in videos.items() if record.subset in {"train", "val", "test"}]
    dynamic_ids = [key for key in videos if key not in fixed_ids]
    groups: Dict[str, list[str]] = defaultdict(list)
    for video_id in dynamic_ids:
        record = videos[video_id]
        if split_by == "subject_id" and record.subject_id not in ("", "unknown", None):
            group = f"subject::{record.subject_id}"
        else:
            group = f"video::{video_id}"
        groups[group].append(video_id)

    group_keys = list(groups)
    random.Random(seed).shuffle(group_keys)
    total = len(group_keys)
    if total == 0:
        return
    num_test = int(round(total * test_ratio))
    num_val = int(round(total * val_ratio))
    if test_ratio > 0 and num_test == 0 and total >= 3:
        num_test = 1
    if val_ratio > 0 and num_val == 0 and total >= 2:
        num_val = 1
    while total - num_test - num_val < 1:
        if num_val > 0:
            num_val -= 1
        elif num_test > 0:
            num_test -= 1
        else:
            break

    test_groups = set(group_keys[:num_test])
    val_groups = set(group_keys[num_test : num_test + num_val])
    for group, ids in groups.items():
        subset = "test" if group in test_groups else "val" if group in val_groups else "train"
        for video_id in ids:
            videos[video_id].subset = subset


def convert_via_dataset(
    csv_dir: Path,
    video_dir: Path,
    *,
    video_meta: Optional[Mapping[str, Mapping[str, Any]]] = None,
    split_by: str = "subject_id",
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[MasterDataset, Dict[str, Any]]:
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    metadata = video_meta or {}
    videos: Dict[str, VideoRecord] = {}
    coarse_labels = set()
    fine_labels = set()
    id_counter: Dict[str, int] = defaultdict(int)

    for csv_file in csv_files:
        for row in parse_via_csv(csv_file):
            file_list = parse_jsonish(row.get("file_list", "[]"))
            if not isinstance(file_list, list) or not file_list:
                continue
            filename = Path(str(file_list[0])).name
            base_video_id = Path(filename).stem
            video_id = base_video_id
            if video_id in videos and videos[video_id].filename != filename:
                id_counter[base_video_id] += 1
                video_id = f"{base_video_id}_{id_counter[base_video_id]:02d}"

            segment = extract_segment(parse_jsonish(row.get("temporal_coordinates", "[]")))
            if segment is None:
                continue
            start, end = segment
            coarse_label = infer_parent_label(csv_file, filename)
            fine_label = extract_fine_label(parse_jsonish(row.get("metadata", "{}")))
            coarse_labels.add(coarse_label)
            fine_labels.add(fine_label)

            extra = metadata.get(filename) or metadata.get(base_video_id) or {}
            source_path = Path(video_dir) / filename
            if video_id not in videos:
                media = probe_video(source_path)
                weight = extra.get("weight_g")
                videos[video_id] = VideoRecord(
                    video_id=video_id,
                    filename=filename,
                    source_path=str(source_path.resolve()) if source_path.exists() else str(source_path),
                    duration=float(media.get("duration", 0.0)),
                    fps=float(media.get("fps", 0.0)),
                    num_frames=int(media.get("num_frames", 0)),
                    subset=str(extra.get("subset", "")).lower(),
                    subject_id=str(extra.get("subject_id", "unknown")),
                    session_id=str(extra.get("session_id", "unknown")),
                    view_type=str(extra.get("view_type", "unknown")),
                    skill_level=normalize_skill_level(extra.get("skill_level", "unknown")),
                    has_gyro=bool(extra.get("has_gyro", False)),
                    gyro_path=extra.get("gyro_path"),
                    weight_path=extra.get("weight_path"),
                    weight_g=float(weight) if weight not in (None, "") else None,
                )
            videos[video_id].annotations.append(
                SegmentAnnotation(start, end, coarse_label, fine_label)
            )

    assign_subsets(videos, split_by, val_ratio, test_ratio, seed)
    for record in videos.values():
        if record.duration <= 0 and record.annotations:
            record.duration = max(annotation.end for annotation in record.annotations)

    dataset = MasterDataset(
        version="1.0",
        meta={
            "source": "VIA CSV",
            "split_by": split_by,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
        },
        label_space={"coarse": sorted(coarse_labels), "fine": sorted(fine_labels)},
        videos=videos,
    )
    subset_counts: Dict[str, int] = defaultdict(int)
    skill_counts: Dict[str, int] = defaultdict(int)
    for record in videos.values():
        subset_counts[record.subset] += 1
        skill_counts[record.skill_level] += 1
    stats = {
        "num_videos": len(videos),
        "num_annotations": sum(len(record.annotations) for record in videos.values()),
        "subset_counts": dict(subset_counts),
        "skill_level_counts": dict(sorted(skill_counts.items())),
        "num_coarse_classes": len(coarse_labels),
        "num_fine_classes": len(fine_labels),
    }
    return dataset, stats
