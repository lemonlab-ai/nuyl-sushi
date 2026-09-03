"""Resumable, read-only SHA-256 inventories for immutable source archives."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


INVENTORY_FIELDS = [
    "relative_path",
    "size_bytes",
    "mtime_ns",
    "sha256",
    "status",
    "error",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_regular_files(root: Path) -> Iterable[Path]:
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            candidate = current_path / filename
            if not candidate.is_symlink():
                yield candidate


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid checkpoint line {line_number}: {exc}") from exc
            rows[str(row["relative_path"])] = row
    return rows


def _checkpoint_matches(row: dict[str, Any], size: int, mtime_ns: int) -> bool:
    return (
        row.get("status") == "ok"
        and int(row.get("size_bytes", -1)) == size
        and int(row.get("mtime_ns", -1)) == mtime_ns
        and len(str(row.get("sha256", ""))) == 64
    )


def _tree_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["size_bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_archive_inventory(
    source_root: Path,
    output_dir: Path,
    archive_id: str,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict[str, Any]:
    root = source_root.resolve()
    destination = output_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        destination.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("Inventory output must be outside the immutable source root")

    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "inventory.checkpoint.jsonl"
    previous = _load_checkpoint(checkpoint_path)
    files = list(iter_regular_files(root))
    rows: list[dict[str, Any]] = []
    reused = 0
    hashed_bytes = 0

    with checkpoint_path.open("a", encoding="utf-8", newline="\n") as checkpoint:
        for index, path in enumerate(files, start=1):
            relative = path.relative_to(root).as_posix()
            try:
                before = path.stat()
                cached = previous.get(relative)
                if cached and _checkpoint_matches(cached, before.st_size, before.st_mtime_ns):
                    row = {field: cached.get(field, "") for field in INVENTORY_FIELDS}
                    reused += 1
                else:
                    digest = sha256_file(path)
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        row = {
                            "relative_path": relative,
                            "size_bytes": after.st_size,
                            "mtime_ns": after.st_mtime_ns,
                            "sha256": "",
                            "status": "error",
                            "error": "source_changed_during_hash",
                        }
                    else:
                        row = {
                            "relative_path": relative,
                            "size_bytes": before.st_size,
                            "mtime_ns": before.st_mtime_ns,
                            "sha256": digest,
                            "status": "ok",
                            "error": "",
                        }
                        hashed_bytes += before.st_size
                    checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
                    checkpoint.flush()
            except OSError as exc:
                row = {
                    "relative_path": relative,
                    "size_bytes": "",
                    "mtime_ns": "",
                    "sha256": "",
                    "status": "error",
                    "error": str(exc)[:500],
                }
                checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
                checkpoint.flush()
            rows.append(row)
            if progress:
                progress(index, len(files), relative)

    rows.sort(key=lambda item: str(item["relative_path"]))
    temporary_checkpoint = checkpoint_path.with_suffix(".canonical.tmp")
    with temporary_checkpoint.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary_checkpoint.replace(checkpoint_path)

    with (destination / "archive_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    status_counts = Counter(str(row["status"]) for row in rows)
    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("sha256"):
            hash_groups[str(row["sha256"])].append(row)
    duplicate_groups = [group for group in hash_groups.values() if len(group) > 1]
    duplicate_excess_bytes = sum(
        sum(int(row["size_bytes"]) for row in group)
        - max(int(row["size_bytes"]) for row in group)
        for group in duplicate_groups
    )
    extension_counts = Counter(
        Path(str(row["relative_path"])).suffix.lower() or "[no extension]" for row in rows
    )
    top_level_bytes: dict[str, int] = defaultdict(int)
    top_level_files: Counter[str] = Counter()
    for row in rows:
        top = str(row["relative_path"]).split("/", 1)[0]
        top_level_bytes[top] += int(row["size_bytes"] or 0)
        top_level_files[top] += 1
    summary = {
        "schema_version": 1,
        "archive_id": archive_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root_name": root.name,
        "files": len(rows),
        "bytes": sum(int(row["size_bytes"] or 0) for row in rows),
        "status_counts": dict(sorted(status_counts.items())),
        "hashed_bytes": sum(
            int(row["size_bytes"] or 0) for row in rows if row["status"] == "ok"
        ),
        "reused_files_current_run": reused,
        "newly_hashed_bytes_current_run": hashed_bytes,
        "tree_sha256": _tree_digest(rows) if not status_counts.get("error") else "",
        "duplicate_hash_groups": len(duplicate_groups),
        "duplicate_excess_bytes": duplicate_excess_bytes,
        "extension_counts": dict(sorted(extension_counts.items())),
        "top_level": [
            {
                "name": name,
                "files": top_level_files[name],
                "bytes": top_level_bytes[name],
            }
            for name in sorted(top_level_files)
        ],
        "complete": bool(rows) and not status_counts.get("error"),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
