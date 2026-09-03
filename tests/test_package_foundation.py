import os
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nuyl_sushi import __version__
from nuyl_sushi.cli.metadata_candidates import build_parser
from nuyl_sushi.config import PROJECT_ROOT_ENV, ProjectPaths, find_project_root


class PackageFoundationTests(unittest.TestCase):
    def test_version_is_exposed(self):
        self.assertEqual(__version__, "0.1.0")

    def test_pyproject_matches_package_version(self):
        with (ROOT_DIR / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle)["project"]
        self.assertEqual(project["version"], __version__)
        self.assertGreaterEqual(tuple(map(int, project["requires-python"].removeprefix(">=").split("."))), (3, 11))

    def test_discovers_checkout_from_nested_path(self):
        self.assertEqual(find_project_root(ROOT_DIR / "tests"), ROOT_DIR.resolve())

    def test_environment_override_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {PROJECT_ROOT_ENV: temp_dir}):
                self.assertEqual(find_project_root(), Path(temp_dir).resolve())

    def test_cli_defaults_derive_from_project_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = ProjectPaths(Path(temp_dir))
            args = build_parser(paths).parse_args([])
            self.assertEqual(
                Path(args.video_manifest),
                Path(temp_dir) / "artifacts" / "data_audit" / "video_manifest.csv",
            )


if __name__ == "__main__":
    unittest.main()
