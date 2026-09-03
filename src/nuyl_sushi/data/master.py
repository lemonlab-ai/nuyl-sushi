import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from nuyl_sushi.domain.models import MasterDataset


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    location: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def load_master(path: Path) -> MasterDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("master annotation root must be an object")
    return MasterDataset.from_mapping(payload)


def save_master(dataset: MasterDataset, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _issue(severity: str, code: str, location: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity, code, location, message)


def validate_master(
    dataset: MasterDataset,
    *,
    check_source_files: bool = True,
    duration_tolerance: float = 1e-3,
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    coarse_space = set(dataset.label_space.get("coarse", []))
    fine_space = set(dataset.label_space.get("fine", []))
    subject_subsets: Dict[str, set[str]] = {}

    if not dataset.version:
        issues.append(_issue("error", "missing_version", "version", "dataset version is empty"))

    for key, record in dataset.videos.items():
        location = f"videos.{key}"
        if record.video_id != key:
            issues.append(
                _issue("error", "video_id_mismatch", location, f"record video_id={record.video_id!r}")
            )
        if not record.filename:
            issues.append(_issue("error", "missing_filename", location, "filename is empty"))
        if check_source_files and (not record.source_path or not Path(record.source_path).is_file()):
            issues.append(
                _issue("warning", "missing_video_file", location, f"missing video file: {record.source_path}")
            )
        if record.skill_level.strip().lower() in {"", "unknown"}:
            issues.append(
                _issue(
                    "warning",
                    "unknown_skill_level",
                    location,
                    "skill_level is unknown (recommended: beginner/intermediate/expert)",
                )
            )
        if record.duration < 0 or not math.isfinite(record.duration):
            issues.append(_issue("error", "invalid_duration", location, f"duration={record.duration}"))
        if record.subset not in {"", "train", "val", "test"}:
            issues.append(_issue("warning", "unknown_subset", location, f"subset={record.subset!r}"))
        subject = record.subject_id.strip()
        if subject and subject.lower() != "unknown" and record.subset:
            subject_subsets.setdefault(subject, set()).add(record.subset)
        if not record.annotations:
            issues.append(_issue("warning", "no_annotations", location, "video has no annotations"))

        seen_segments = set()
        for index, annotation in enumerate(record.annotations):
            ann_location = f"{location}.annotations[{index}]"
            if not all(math.isfinite(value) for value in (annotation.start, annotation.end)):
                issues.append(_issue("error", "non_finite_segment", ann_location, "segment is non-finite"))
            elif annotation.start < 0 or annotation.end <= annotation.start:
                issues.append(
                    _issue(
                        "error",
                        "invalid_segment",
                        ann_location,
                        f"start={annotation.start}, end={annotation.end}",
                    )
                )
            elif record.duration > 0 and annotation.end > record.duration + duration_tolerance:
                issues.append(
                    _issue(
                        "warning",
                        "segment_beyond_duration",
                        ann_location,
                        f"end={annotation.end:.3f} beyond duration={record.duration:.3f}",
                    )
                )
            if not annotation.coarse_label:
                issues.append(_issue("error", "missing_coarse_label", ann_location, "coarse label is empty"))
            elif coarse_space and annotation.coarse_label not in coarse_space:
                issues.append(
                    _issue("error", "coarse_label_outside_space", ann_location, annotation.coarse_label)
                )
            if not annotation.fine_label:
                issues.append(_issue("error", "missing_fine_label", ann_location, "fine label is empty"))
            elif fine_space and annotation.fine_label not in fine_space:
                issues.append(_issue("error", "fine_label_outside_space", ann_location, annotation.fine_label))
            signature = (
                annotation.start,
                annotation.end,
                annotation.coarse_label,
                annotation.fine_label,
            )
            if signature in seen_segments:
                issues.append(_issue("warning", "duplicate_segment", ann_location, "exact duplicate annotation"))
            seen_segments.add(signature)

    for subject, subsets in sorted(subject_subsets.items()):
        if len(subsets) > 1:
            issues.append(
                _issue(
                    "error",
                    "subject_leakage",
                    f"subjects.{subject}",
                    f"subject appears in subsets: {sorted(subsets)}",
                )
            )
    return issues


def validation_summary(dataset: MasterDataset, issues: Iterable[ValidationIssue]) -> Dict[str, Any]:
    materialized = list(issues)
    return {
        "videos": len(dataset.videos),
        "annotations": sum(len(record.annotations) for record in dataset.videos.values()),
        "warnings": sum(issue.severity == "warning" for issue in materialized),
        "errors": sum(issue.severity == "error" for issue in materialized),
        "issues": [issue.to_dict() for issue in materialized],
    }

