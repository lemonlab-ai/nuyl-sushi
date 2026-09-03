"""CLI for the resumable immutable-source inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from nuyl_sushi.config import ProjectPaths
from nuyl_sushi.data.archive_inventory import build_archive_inventory


def build_parser(paths: Optional[ProjectPaths] = None) -> argparse.ArgumentParser:
    project = paths or ProjectPaths.discover()
    parser = argparse.ArgumentParser(
        description="Build or resume a read-only SHA-256 inventory for an immutable source tree."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--archive-id", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(project.artifacts / "archive_inventory"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    def progress(index: int, total: int, relative: str) -> None:
        if index % 25 == 0 or index == total:
            print(f"[HASH] {index}/{total} {relative}", flush=True)

    summary = build_archive_inventory(
        Path(args.source_root), Path(args.output_dir), args.archive_id, progress
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
