import ast
import csv
import json
import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def parse_jsonish(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list, int, float, bool)):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    for decoder in (json.loads, ast.literal_eval):
        try:
            return decoder(text)
        except Exception:
            pass
    try:
        import jsonpickle  # type: ignore

        return jsonpickle.decode(text)
    except Exception:
        return text


def sanitize_label(label: str) -> str:
    value = re.sub(r"\s+", "-", str(label).strip())
    return value.replace("_", "-").upper()


def infer_parent_label(csv_path: Path, video_filename: str) -> str:
    csv_stem = csv_path.stem
    video_stem = Path(video_filename).stem
    inferred = (
        csv_stem[len(video_stem) + 1 :]
        if csv_stem.startswith(video_stem + "_")
        else csv_stem
    )
    inferred = re.sub(r"\(.*?\)", "", inferred).strip()
    return sanitize_label(inferred) if inferred else "UNKNOWN"


def extract_fine_label(metadata: Any, fallback: str = "UNKNOWN") -> str:
    if isinstance(metadata, dict):
        for key in ("1", "label", "fine_label", "action", "name"):
            if key in metadata and metadata[key]:
                return sanitize_label(str(metadata[key]))
    if isinstance(metadata, list) and metadata:
        return sanitize_label(str(metadata[0]))
    if isinstance(metadata, str) and metadata:
        return sanitize_label(metadata)
    return fallback


def extract_segment(temporal: Any) -> Optional[Tuple[float, float]]:
    if isinstance(temporal, dict):
        for start_key, end_key in (("start", "end"), ("from", "to")):
            if start_key in temporal and end_key in temporal:
                try:
                    start = float(temporal[start_key])
                    end = float(temporal[end_key])
                    return (start, end) if end > start else None
                except (TypeError, ValueError):
                    return None
        return None
    if isinstance(temporal, (list, tuple)) and len(temporal) >= 2:
        try:
            start = float(temporal[0])
            end = float(temporal[1])
            return (start, end) if end > start else None
        except (TypeError, ValueError):
            return None
    return None


def parse_via_csv(csv_path: Path) -> Iterable[Dict[str, str]]:
    lines = csv_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    header = None
    body_lines = []
    for line in lines:
        if line.startswith("# CSV_HEADER"):
            header = line.split("=", 1)[1].strip()
        elif not line.startswith("#"):
            body_lines.append(line)
    if not body_lines:
        return []
    raw_csv = (header + "\n" if header else "") + "\n".join(body_lines)
    return list(csv.DictReader(StringIO(raw_csv)))

