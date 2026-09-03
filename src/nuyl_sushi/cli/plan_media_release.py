"""Create a read-only media release plan; never transcode source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from nuyl_sushi.config import ProjectPaths
from nuyl_sushi.data.media_release import (
    build_ffmpeg_command,
    decide_release_action,
    find_executable,
    load_media_profiles,
    probe_from_manifest,
    probe_video,
    resolve_manifest_source,
    sha256_file,
)


PLAN_FIELDS = [
    "video_id",
    "filename",
    "is_referenced",
    "source_size_bytes",
    "manifest_sha256",
    "observed_sha256",
    "checksum_status",
    "probe_status",
    "codec",
    "width",
    "height",
    "pixel_format",
    "frame_rate",
    "duration_seconds",
    "audio_streams",
    "action",
    "reasons",
    "output_relative_path",
    "command_preview",
]


def build_parser(paths: Optional[ProjectPaths] = None) -> argparse.ArgumentParser:
    project = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description=(
            "Probe videos and create a copy/transcode/reject release plan. "
            "This command is read-only and never executes FFmpeg transcoding."
        )
    )
    parser.add_argument(
        "--video-manifest",
        default=str(project.artifacts / "data_audit" / "video_manifest.csv"),
    )
    parser.add_argument("--source-dir", required=True, help="Read-only folder containing videos.")
    parser.add_argument(
        "--config",
        default=str(project.root / "configs" / "datasets" / "media_profiles.toml"),
    )
    parser.add_argument("--profile", default="research-720p")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to artifacts/release_plans/<profile>.",
    )
    parser.add_argument("--ffprobe", default="", help="Optional explicit ffprobe executable.")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Name used only in command previews.")
    parser.add_argument("--include-unreferenced", action="store_true")
    parser.add_argument(
        "--verify-checksum",
        action="store_true",
        help="Recompute SHA-256 and reject mismatches. This reads every input byte.",
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.video_manifest).resolve()
    source_dir = Path(args.source_dir).resolve()
    config_path = Path(args.config).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else ProjectPaths.discover().artifacts / "release_plans" / args.profile
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    schema_version, profiles = load_media_profiles(config_path)
    if args.profile not in profiles:
        raise ValueError(f"Unknown profile {args.profile!r}; choose from {sorted(profiles)}")
    profile = profiles[args.profile]
    ffprobe = find_executable("ffprobe", args.ffprobe)
    if args.ffprobe and not ffprobe:
        raise FileNotFoundError(f"ffprobe executable not found: {args.ffprobe}")
    input_rows = _read_csv(manifest_path)
    if not args.include_unreferenced:
        input_rows = [row for row in input_rows if row.get("is_referenced", "").lower() == "true"]

    plan_rows: list[dict[str, Any]] = []
    for row in sorted(input_rows, key=lambda item: item.get("video_id", item.get("filename", ""))):
        filename = row.get("filename", "")
        source = resolve_manifest_source(source_dir, filename)
        source_exists = bool(source and source.is_file())
        source_size = source.stat().st_size if source_exists and source else 0
        manifest_hash = row.get("sha256", "")
        observed_hash = ""
        checksum_matches: Optional[bool] = None
        if args.verify_checksum and source_exists and source:
            observed_hash = sha256_file(source)
            checksum_matches = not manifest_hash or observed_hash.lower() == manifest_hash.lower()

        if profile.get("operation") == "copy":
            probe = probe_from_manifest(row)
        elif source_exists and source and ffprobe:
            probe = probe_video(source, ffprobe)
        else:
            probe = probe_from_manifest(row)
        decision = decide_release_action(
            profile,
            probe,
            source_exists=source_exists,
            source_size_bytes=source_size,
            checksum_matches=checksum_matches,
        )
        output_rel = f"{args.profile}/{filename}" if filename else ""
        command_preview = ""
        if decision.action == "transcode":
            command_preview = subprocess.list2cmdline(
                build_ffmpeg_command("<source>", f"<output>/{output_rel}", profile, args.ffmpeg)
            )
        plan_rows.append(
            {
                "video_id": row.get("video_id", Path(filename).stem),
                "filename": filename,
                "is_referenced": row.get("is_referenced", ""),
                "source_size_bytes": source_size,
                "manifest_sha256": manifest_hash,
                "observed_sha256": observed_hash,
                "checksum_status": (
                    "not_checked"
                    if not args.verify_checksum
                    else "computed_no_reference"
                    if not manifest_hash
                    else "match"
                    if checksum_matches
                    else "mismatch"
                ),
                "probe_status": probe.status,
                "codec": probe.codec,
                "width": probe.width if probe.width is not None else "",
                "height": probe.height if probe.height is not None else "",
                "pixel_format": probe.pixel_format,
                "frame_rate": probe.frame_rate if probe.frame_rate is not None else "",
                "duration_seconds": (
                    probe.duration_seconds if probe.duration_seconds is not None else ""
                ),
                "audio_streams": probe.audio_streams if probe.audio_streams is not None else "",
                "action": decision.action,
                "reasons": "|".join(decision.reasons),
                "output_relative_path": output_rel,
                "command_preview": command_preview,
            }
        )

    action_counts = Counter(row["action"] for row in plan_rows)
    reason_counts = Counter(
        reason for row in plan_rows for reason in str(row["reasons"]).split("|") if reason
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run",
        "transcoding_executed": False,
        "profile": args.profile,
        "profile_schema_version": schema_version,
        "profile_config_sha256": _config_sha256(config_path),
        "manifest_filename": manifest_path.name,
        "ffprobe_available": bool(ffprobe),
        "checksum_verification": bool(args.verify_checksum),
        "videos_planned": len(plan_rows),
        "action_counts": dict(sorted(action_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "release_ready": bool(plan_rows) and not action_counts.get("blocked") and not action_counts.get("reject"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "release_plan.csv", plan_rows)
    (output_dir / "release_plan.json").write_text(
        json.dumps({"summary": summary, "videos": plan_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    summary = run(build_parser().parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
