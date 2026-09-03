from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class SegmentAnnotation:
    start: float
    end: float
    coarse_label: str
    fine_label: str
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SegmentAnnotation":
        known = {"start", "end", "coarse_label", "fine_label"}
        return cls(
            start=float(value.get("start", 0.0)),
            end=float(value.get("end", 0.0)),
            coarse_label=str(value.get("coarse_label", "")),
            fine_label=str(value.get("fine_label", "")),
            extras={key: item for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "start": self.start,
            "end": self.end,
            "coarse_label": self.coarse_label,
            "fine_label": self.fine_label,
        }
        payload.update(self.extras)
        return payload


@dataclass
class VideoRecord:
    video_id: str
    filename: str
    source_path: str
    duration: float = 0.0
    fps: float = 0.0
    num_frames: int = 0
    subset: str = ""
    subject_id: str = "unknown"
    session_id: str = "unknown"
    view_type: str = "unknown"
    skill_level: str = "unknown"
    has_gyro: bool = False
    gyro_path: Optional[str] = None
    weight_path: Optional[str] = None
    weight_g: Optional[float] = None
    annotations: List[SegmentAnnotation] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, key: str, value: Mapping[str, Any]) -> "VideoRecord":
        known = {
            "video_id", "filename", "source_path", "duration", "fps", "num_frames",
            "subset", "subject_id", "session_id", "view_type", "skill_level", "has_gyro",
            "gyro_path", "weight_path", "weight_g", "annotations",
        }
        weight = value.get("weight_g")
        raw_annotations = value.get("annotations", [])
        if not isinstance(raw_annotations, list):
            raise TypeError(f"video {key!r} field 'annotations' must be an array")
        if any(not isinstance(item, Mapping) for item in raw_annotations):
            raise TypeError(f"video {key!r} annotations must contain objects")
        return cls(
            video_id=str(value.get("video_id", key)),
            filename=str(value.get("filename", "")),
            source_path=str(value.get("source_path", "")),
            duration=float(value.get("duration", 0.0) or 0.0),
            fps=float(value.get("fps", 0.0) or 0.0),
            num_frames=int(value.get("num_frames", 0) or 0),
            subset=str(value.get("subset", "")),
            subject_id=str(value.get("subject_id", "unknown")),
            session_id=str(value.get("session_id", "unknown")),
            view_type=str(value.get("view_type", "unknown")),
            skill_level=str(value.get("skill_level", "unknown")),
            has_gyro=bool(value.get("has_gyro", False)),
            gyro_path=value.get("gyro_path"),
            weight_path=value.get("weight_path"),
            weight_g=float(weight) if weight not in (None, "") else None,
            annotations=[SegmentAnnotation.from_mapping(item) for item in raw_annotations],
            extras={item_key: item for item_key, item in value.items() if item_key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "video_id": self.video_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "duration": self.duration,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "subset": self.subset,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "view_type": self.view_type,
            "skill_level": self.skill_level,
            "has_gyro": self.has_gyro,
            "gyro_path": self.gyro_path,
            "weight_path": self.weight_path,
            "weight_g": self.weight_g,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }
        payload.update(self.extras)
        return payload


@dataclass
class MasterDataset:
    version: str
    meta: Dict[str, Any]
    label_space: Dict[str, List[str]]
    videos: Dict[str, VideoRecord]
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MasterDataset":
        raw_videos = value.get("videos", {})
        if not isinstance(raw_videos, Mapping):
            raise TypeError("master field 'videos' must be an object")
        raw_labels = value.get("label_space", {})
        if not isinstance(raw_labels, Mapping):
            raise TypeError("master field 'label_space' must be an object")
        known = {"version", "meta", "label_space", "videos"}
        if any(not isinstance(item, Mapping) for item in raw_videos.values()):
            raise TypeError("master field 'videos' must contain objects")
        return cls(
            version=str(value.get("version", "")),
            meta=dict(value.get("meta", {})),
            label_space={
                "coarse": [str(item) for item in raw_labels.get("coarse", [])],
                "fine": [str(item) for item in raw_labels.get("fine", [])],
            },
            videos={
                str(key): VideoRecord.from_mapping(str(key), item)
                for key, item in raw_videos.items()
            },
            extras={key: item for key, item in value.items() if key not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "version": self.version,
            "meta": self.meta,
            "label_space": self.label_space,
            "videos": {key: record.to_dict() for key, record in self.videos.items()},
        }
        payload.update(self.extras)
        return payload


@dataclass(frozen=True)
class DatasetSplit:
    train: List[str] = field(default_factory=list)
    val: List[str] = field(default_factory=list)
    test: List[str] = field(default_factory=list)

    def all_video_ids(self) -> List[str]:
        return [*self.train, *self.val, *self.test]
