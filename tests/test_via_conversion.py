import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nuyl_sushi.data.conversion import convert_via_dataset, load_optional_meta
from nuyl_sushi.data.via import extract_segment, parse_jsonish, sanitize_label
from nuyl_sushi.cli.convert_via import main as convert_via_main


FIXTURE_DIR = ROOT_DIR / "tests" / "fixtures" / "via"


def projection(dataset) -> dict:
    return {
        "version": dataset.version,
        "label_space": dataset.label_space,
        "videos": {
            video_id: {
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "view_type": record.view_type,
                "skill_level": record.skill_level,
                "subset": record.subset,
                "annotations": [
                    [ann.start, ann.end, ann.coarse_label, ann.fine_label]
                    for ann in record.annotations
                ],
            }
            for video_id, record in dataset.videos.items()
        },
    }


class ViaParsingTests(unittest.TestCase):
    def test_jsonish_and_segment_variants(self):
        self.assertEqual(parse_jsonish("{'start': 1, 'end': 2}"), {"start": 1, "end": 2})
        self.assertEqual(extract_segment({"from": "1", "to": "2.5"}), (1.0, 2.5))
        self.assertIsNone(extract_segment([2, 1]))

    def test_label_normalization(self):
        self.assertEqual(sanitize_label(" cutting_slices "), "CUTTING-SLICES")

    def test_metadata_loader_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "meta.json"
            path.write_text('{"videos":{"v.mp4":{"subject_id":"S1"}}}', encoding="utf-8-sig")
            self.assertEqual(load_optional_meta(path)["v.mp4"]["subject_id"], "S1")

    def test_golden_conversion(self):
        dataset, stats = convert_via_dataset(
            FIXTURE_DIR / "annotations",
            FIXTURE_DIR / "videos",
            video_meta=load_optional_meta(FIXTURE_DIR / "video_meta.json"),
            val_ratio=0,
            test_ratio=0,
            seed=42,
        )
        expected = json.loads((FIXTURE_DIR / "expected_projection.json").read_text(encoding="utf-8"))
        self.assertEqual(projection(dataset), expected)
        self.assertEqual(stats["num_videos"], 2)
        self.assertEqual(stats["num_annotations"], 3)

    def test_compatibility_cli_writes_all_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            convert_via_main(
                [
                    "--csv-dir", str(FIXTURE_DIR / "annotations"),
                    "--video-dir", str(FIXTURE_DIR / "videos"),
                    "--video-meta-json", str(FIXTURE_DIR / "video_meta.json"),
                    "--output-dir", temp_dir,
                    "--val-ratio", "0",
                    "--test-ratio", "0",
                ]
            )
            outputs = {path.name for path in Path(temp_dir).iterdir()}
            self.assertEqual(
                outputs,
                {"master_annotations.json", "label_map_coarse.txt", "label_map_fine.txt", "stats.json"},
            )


if __name__ == "__main__":
    unittest.main()
