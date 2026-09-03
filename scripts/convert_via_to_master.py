"""Compatibility entry point; VIA conversion now lives in the package."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nuyl_sushi.cli.convert_via import main  # noqa: E402
from nuyl_sushi.data.conversion import (  # noqa: E402,F401
    SKILL_LEVEL_ALIASES,
    assign_subsets,
    convert_via_dataset,
    load_optional_meta,
    normalize_skill_level,
    probe_video as get_video_probe,
)
from nuyl_sushi.data.via import (  # noqa: E402,F401
    extract_fine_label,
    extract_segment,
    infer_parent_label,
    parse_jsonish,
    parse_via_csv,
    sanitize_label,
)


if __name__ == "__main__":
    main()
