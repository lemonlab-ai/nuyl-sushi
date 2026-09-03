"""Create and verify an immutable no-audio research release."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from nuyl_sushi.config import ProjectPaths
from nuyl_sushi.data.media_release import (
    build_remux_command,
    decide_release_action,
    find_executable,
    load_media_profiles,
    probe_video,
    resolve_manifest_source,
    sha256_file,
)


MANIFEST_FIELDS = [
    "dataset_version",
    "video_id",
    "source_filename",
    "source_size_bytes",
    "source_sha256",
    "derivative_relative_path",
    "derivative_size_bytes",
    "derivative_sha256",
    "operation",
    "profile_name",
    "source_codec",
    "derivative_codec",
    "width",
    "height",
    "frame_rate",
    "source_audio_streams",
    "derivative_audio_streams",
    "duration_delta_seconds",
    "validation_status",
]


def build_parser(paths: Optional[ProjectPaths] = None) -> argparse.ArgumentParser:
    project = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description=(
            "Remove audio by copying the original video bitstream into new MP4 files. "
            "Sources are checksum-verified and never modified."
        )
    )
    parser.add_argument("--video-manifest", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument(
        "--config",
        default=str(project.root / "configs" / "datasets" / "media_profiles.toml"),
    )
    parser.add_argument("--profile", default="research-720p-no-audio")
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--ffprobe", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required acknowledgement that new derivative files will be written.",
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _same_video_stream(source: Any, derivative: Any) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for field in ("codec", "width", "height", "pixel_format"):
        if getattr(source, field) != getattr(derivative, field):
            errors.append(f"changed_{field}")
    if source.frame_rate is None or derivative.frame_rate is None:
        errors.append("unknown_frame_rate")
    elif abs(source.frame_rate - derivative.frame_rate) > 0.001:
        errors.append("changed_frame_rate")
    if derivative.audio_streams != 0:
        errors.append("audio_not_removed")
    if source.duration_seconds is None or derivative.duration_seconds is None:
        errors.append("unknown_duration")
    elif abs(source.duration_seconds - derivative.duration_seconds) > 0.1:
        errors.append("duration_mismatch")
    return not errors, errors


def _ffmpeg_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "-version"], capture_output=True, text=True, timeout=30, check=True
    )
    return result.stdout.splitlines()[0] if result.stdout else "unknown"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.execute:
        raise ValueError("Refusing to write derivatives without --execute")
    manifest_path = Path(args.video_manifest).resolve()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    for path in (manifest_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if (
        output_dir == source_dir
        or _inside(output_dir, source_dir)
        or _inside(source_dir, output_dir)
    ):
        raise ValueError(
            "Output directory must be separate from, not inside, and not an ancestor of "
            "the immutable source directory"
        )

    schema_version, profiles = load_media_profiles(config_path)
    if args.profile not in profiles:
        raise ValueError(f"Unknown profile: {args.profile}")
    profile = profiles[args.profile]
    if profile.get("audio_policy") != "strip":
        raise ValueError("The selected profile must explicitly set audio_policy=strip")
    ffmpeg = find_executable("ffmpeg", args.ffmpeg)
    ffprobe = find_executable("ffprobe", args.ffprobe)
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("Both ffmpeg and ffprobe are required")

    input_rows = [
        row
        for row in _read_csv(manifest_path)
        if row.get("is_referenced", "").lower() == "true"
    ]
    videos_dir = output_dir / "videos"
    manifests_dir = output_dir / "manifests"
    videos_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    release_rows: list[dict[str, Any]] = []

    for index, row in enumerate(sorted(input_rows, key=lambda item: item["video_id"]), start=1):
        filename = row["filename"]
        source = resolve_manifest_source(source_dir, filename)
        if source is None or not source.is_file():
            raise FileNotFoundError(f"Source missing or unsafe manifest path: {filename}")
        source_hash = sha256_file(source)
        expected_hash = row.get("sha256", "").lower()
        if not expected_hash or source_hash.lower() != expected_hash:
            raise ValueError(f"Source checksum mismatch: {filename}")
        source_probe = probe_video(source, ffprobe)
        decision = decide_release_action(
            profile,
            source_probe,
            source_exists=True,
            source_size_bytes=source.stat().st_size,
            checksum_matches=True,
        )
        if decision.action != "remux":
            raise ValueError(
                f"Expected safe remux for {filename}, got {decision.action}: {decision.reasons}"
            )

        destination = videos_dir / filename
        partial = videos_dir / f"{source.stem}.partial{source.suffix}"
        if destination.exists() and not args.resume:
            raise FileExistsError(f"Destination exists; use --resume to verify it: {destination}")
        if not destination.exists():
            if partial.exists():
                partial.unlink()
            result = subprocess.run(
                build_remux_command(str(source), str(partial), ffmpeg),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                if partial.exists():
                    partial.unlink()
                raise RuntimeError(f"FFmpeg failed for {filename}: {result.stderr[-1000:]}")
            partial.replace(destination)

        derivative_probe = probe_video(destination, ffprobe)
        valid, errors = _same_video_stream(source_probe, derivative_probe)
        if not valid:
            raise ValueError(f"Derivative validation failed for {filename}: {errors}")
        release_rows.append(
            {
                "dataset_version": args.dataset_version,
                "video_id": row["video_id"],
                "source_filename": filename,
                "source_size_bytes": source.stat().st_size,
                "source_sha256": source_hash,
                "derivative_relative_path": f"videos/{filename}",
                "derivative_size_bytes": destination.stat().st_size,
                "derivative_sha256": sha256_file(destination),
                "operation": "remux_strip_audio_video_copy",
                "profile_name": args.profile,
                "source_codec": source_probe.codec,
                "derivative_codec": derivative_probe.codec,
                "width": derivative_probe.width,
                "height": derivative_probe.height,
                "frame_rate": derivative_probe.frame_rate,
                "source_audio_streams": source_probe.audio_streams,
                "derivative_audio_streams": derivative_probe.audio_streams,
                "duration_delta_seconds": round(
                    abs((source_probe.duration_seconds or 0) - (derivative_probe.duration_seconds or 0)),
                    6,
                ),
                "validation_status": "ok",
            }
        )
        print(f"[REMUX] {index}/{len(input_rows)} {filename}")

    csv_path = manifests_dir / "release_manifest.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(release_rows)
    summary = {
        "schema_version": 1,
        "dataset_version": args.dataset_version,
        "profile_name": args.profile,
        "profile_schema_version": schema_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "remux_strip_audio_video_copy",
        "publication_consent": "unresolved",
        "videos": len(release_rows),
        "source_bytes": sum(int(row["source_size_bytes"]) for row in release_rows),
        "derivative_bytes": sum(int(row["derivative_size_bytes"]) for row in release_rows),
        "all_valid": all(row["validation_status"] == "ok" for row in release_rows),
        "ffmpeg_version": _ffmpeg_version(ffmpeg),
    }
    (manifests_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
