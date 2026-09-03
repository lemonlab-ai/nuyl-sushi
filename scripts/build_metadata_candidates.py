"""Compatibility entry point; implementation lives in the installable package."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nuyl_sushi.cli.metadata_candidates import main  # noqa: E402
from nuyl_sushi.data.metadata_candidates import (  # noqa: E402,F401
    build_label_distribution,
    build_sensor_candidates,
    extract_sensor_clock_seconds,
    extract_video_clock_seconds,
    infer_view,
    sensor_candidate_confidence,
)


if __name__ == "__main__":
    main()
