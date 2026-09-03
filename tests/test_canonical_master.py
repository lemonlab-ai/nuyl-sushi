import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nuyl_sushi.data.master import load_master, save_master, validate_master, validation_summary
from nuyl_sushi.domain import MasterDataset


def valid_payload() -> dict:
    return {
        "version": "1.0",
        "meta": {"source": "test"},
        "label_space": {"coarse": ["CUT"], "fine": ["SLICE"]},
        "videos": {
            "v1": {
                "video_id": "v1",
                "filename": "v1.mp4",
                "source_path": "v1.mp4",
                "duration": 10.0,
                "subset": "train",
                "subject_id": "S1",
                "skill_level": "expert",
                "annotations": [
                    {
                        "start": 1.0,
                        "end": 2.0,
                        "coarse_label": "CUT",
                        "fine_label": "SLICE",
                    }
                ],
                "future_field": {"preserved": True},
            }
        },
        "future_root": "preserved",
    }


class CanonicalMasterTests(unittest.TestCase):
    def test_round_trip_preserves_unknown_fields(self):
        dataset = MasterDataset.from_mapping(valid_payload())
        self.assertEqual(dataset.to_dict()["future_root"], "preserved")
        self.assertEqual(dataset.to_dict()["videos"]["v1"]["future_field"], {"preserved": True})

    def test_valid_dataset_has_no_errors_without_file_check(self):
        dataset = MasterDataset.from_mapping(valid_payload())
        issues = validate_master(dataset, check_source_files=False)
        self.assertEqual(validation_summary(dataset, issues)["errors"], 0)

    def test_detects_invalid_segment_label_and_subject_leakage(self):
        payload = valid_payload()
        payload["videos"]["v1"]["annotations"][0]["end"] = 0.5
        payload["videos"]["v1"]["annotations"][0]["fine_label"] = "UNKNOWN-LABEL"
        second = dict(payload["videos"]["v1"])
        second.update({"video_id": "v2", "filename": "v2.mp4", "subset": "test"})
        second["annotations"] = []
        payload["videos"]["v2"] = second
        codes = {
            issue.code
            for issue in validate_master(
                MasterDataset.from_mapping(payload), check_source_files=False
            )
        }
        self.assertTrue({"invalid_segment", "fine_label_outside_space", "subject_leakage"} <= codes)

    def test_load_and_save(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "master.json"
            destination = Path(temp_dir) / "roundtrip.json"
            source.write_text(json.dumps(valid_payload()), encoding="utf-8")
            dataset = load_master(source)
            save_master(dataset, destination)
            self.assertEqual(load_master(destination).videos["v1"].video_id, "v1")


if __name__ == "__main__":
    unittest.main()
