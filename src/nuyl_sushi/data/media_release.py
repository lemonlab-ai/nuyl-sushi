"""Read-only media probing and release-planning helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class MediaProbe:
    status: str
    error: str = ""
    codec: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    pixel_format: str = ""
    frame_rate: Optional[float] = None
    duration_seconds: Optional[float] = None
    audio_streams: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReleaseDecision:
    action: str
    reasons: tuple[str, ...]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_media_profiles(path: Path) -> tuple[int, dict[str, dict[str, Any]]]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    schema_version = payload.pop("schema_version", None)
    if schema_version != 1:
        raise ValueError(f"Unsupported media profile schema_version: {schema_version!r}")
    profiles = {name: dict(value) for name, value in payload.items() if isinstance(value, dict)}
    if not profiles:
        raise ValueError("No media profiles found")
    return schema_version, profiles


def find_executable(name: str, explicit: str = "") -> Optional[str]:
    if explicit:
        candidate = Path(explicit).expanduser()
        return str(candidate.resolve()) if candidate.is_file() else None
    return shutil.which(name)


def parse_frame_rate(value: Any) -> Optional[float]:
    if value in (None, "", "N/A", "0/0"):
        return None
    text = str(value)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            divisor = float(denominator)
            return float(numerator) / divisor if divisor else None
        return float(text)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "", "N/A") else None
    except (TypeError, ValueError):
        return None


def probe_from_manifest(row: Mapping[str, Any]) -> MediaProbe:
    """Use prior audit columns when ffprobe is unavailable.

    Old OpenCV inventories do not contain pixel format or audio information, so
    compliance remains blocked instead of being guessed.
    """

    status = str(row.get("probe_status", "not_checked"))
    return MediaProbe(
        status=status,
        error=str(row.get("probe_error", "")),
        codec=str(row.get("codec", "")),
        width=_optional_int(row.get("width")),
        height=_optional_int(row.get("height")),
        pixel_format=str(row.get("pixel_format", "")),
        frame_rate=parse_frame_rate(row.get("frame_rate")),
        duration_seconds=_optional_float(row.get("duration_probe")),
        audio_streams=_optional_int(row.get("audio_streams")),
    )


def probe_video(path: Path, ffprobe: str, timeout_seconds: int = 120) -> MediaProbe:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration:"
            "stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return MediaProbe(status="failed", error=str(exc))
    if result.returncode != 0:
        return MediaProbe(status="failed", error=result.stderr.strip()[:500])
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return MediaProbe(status="failed", error=f"invalid ffprobe JSON: {exc}")

    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not video:
        return MediaProbe(status="failed", error="no video stream")
    return MediaProbe(
        status="ok",
        codec=str(video.get("codec_name", "")),
        width=_optional_int(video.get("width")),
        height=_optional_int(video.get("height")),
        pixel_format=str(video.get("pix_fmt", "")),
        frame_rate=parse_frame_rate(video.get("avg_frame_rate")),
        duration_seconds=_optional_float(payload.get("format", {}).get("duration")),
        audio_streams=sum(item.get("codec_type") == "audio" for item in streams),
    )


def expected_probe_codec(profile: Mapping[str, Any]) -> str:
    return {
        "libx264": "h264",
        "libx265": "hevc",
    }.get(str(profile.get("video_codec", "")), str(profile.get("video_codec", "")))


def decide_release_action(
    profile: Mapping[str, Any],
    probe: MediaProbe,
    *,
    source_exists: bool = True,
    source_size_bytes: int = 1,
    checksum_matches: Optional[bool] = None,
) -> ReleaseDecision:
    if not source_exists:
        return ReleaseDecision("reject", ("source_missing",))
    if source_size_bytes <= 0:
        return ReleaseDecision("reject", ("source_empty",))
    if checksum_matches is False:
        return ReleaseDecision("reject", ("checksum_mismatch",))
    if profile.get("operation") == "copy":
        return ReleaseDecision("copy", ("immutable_archive",))
    if probe.status != "ok":
        return ReleaseDecision("blocked", (f"probe_{probe.status}", probe.error or "probe_failed"))

    required = {
        "codec": probe.codec,
        "width": probe.width,
        "height": probe.height,
        "pixel_format": probe.pixel_format,
        "audio_streams": probe.audio_streams,
    }
    missing = tuple(f"unknown_{name}" for name, value in required.items() if value in (None, ""))
    if missing:
        return ReleaseDecision("blocked", missing)

    audio_policy = str(profile.get("audio_policy", "preserve"))
    if audio_policy == "review_then_strip" and (probe.audio_streams or 0) > 0:
        return ReleaseDecision("blocked", ("audio_consent_review_required",))

    transcode_reasons: list[str] = []
    expected_codec = expected_probe_codec(profile)
    if expected_codec and probe.codec != expected_codec:
        transcode_reasons.append("video_codec")
    if probe.width and probe.width > int(profile.get("max_width", probe.width)):
        transcode_reasons.append("width")
    if probe.height and probe.height > int(profile.get("max_height", probe.height)):
        transcode_reasons.append("height")
    if profile.get("pixel_format") and probe.pixel_format != profile["pixel_format"]:
        transcode_reasons.append("pixel_format")
    if profile.get("max_fps") and probe.frame_rate is not None:
        if probe.frame_rate > float(profile["max_fps"]) + 0.001:
            transcode_reasons.append("frame_rate")
    if audio_policy == "strip" and (probe.audio_streams or 0) > 0:
        transcode_reasons.append("strip_audio")
    if not bool(profile.get("copy_if_compliant", False)):
        transcode_reasons.append("profile_requires_derivative")

    if transcode_reasons:
        return ReleaseDecision("transcode", tuple(dict.fromkeys(transcode_reasons)))
    return ReleaseDecision("copy", ("profile_compliant",))


def build_ffmpeg_command(
    source: str,
    output: str,
    profile: Mapping[str, Any],
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    width = int(profile["max_width"])
    height = int(profile["max_height"])
    filters = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    ]
    if profile.get("max_fps"):
        filters.append(f"fps={profile['max_fps']}")
    command = [
        ffmpeg,
        "-nostdin",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-vf",
        ",".join(filters),
        "-c:v",
        str(profile["video_codec"]),
        "-crf",
        str(profile["crf"]),
        "-preset",
        str(profile["preset"]),
        "-pix_fmt",
        str(profile["pixel_format"]),
    ]
    if profile.get("audio_policy") in {"strip", "review_then_strip"}:
        command.append("-an")
    if profile.get("faststart"):
        command.extend(["-movflags", "+faststart"])
    command.extend(["-n", output])
    return command


def resolve_manifest_source(source_dir: Path, filename: str) -> Optional[Path]:
    root = source_dir.resolve()
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate
