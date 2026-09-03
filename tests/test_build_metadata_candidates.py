import unittest

from scripts.build_metadata_candidates import (
    build_label_distribution,
    extract_sensor_clock_seconds,
    extract_video_clock_seconds,
    infer_view,
    sensor_candidate_confidence,
)


class ViewInferenceTests(unittest.TestCase):
    def test_preserves_direct_view(self):
        self.assertEqual(infer_view("x.mp4", "view1", "20221215/view1/x.mp4")[0], "view1")

    def test_infers_all_known_edited_families(self):
        cases = {
            "221210_15.mp4": "view2",
            "IMG_6421_01.mp4": "view2",
            "IMG_6423_02.mp4": "view2",
            "VID_20221215_192928.mp4": "view1",
            "VID_20230208_122207_1_12.mp4": "view1",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(infer_view(filename, "unknown", "edited/path")[0], expected)

    def test_unknown_remains_unknown(self):
        self.assertEqual(infer_view("mystery.mp4", "unknown", "edited/path")[0], "unknown")


class TimeCandidateTests(unittest.TestCase):
    def test_extracts_video_and_sensor_clock(self):
        expected = 19 * 3600 + 29 * 60 + 28
        self.assertEqual(extract_video_clock_seconds("VID_20221215_192928.mp4"), expected)
        self.assertEqual(extract_sensor_clock_seconds("[12_15]19-29-28"), expected)

    def test_segmented_parent_requires_manual_review(self):
        self.assertEqual(
            sensor_candidate_confidence("VID_20230208_122207_1_03.mp4", 3),
            "manual_only_segmented_parent",
        )


class LabelDistributionTests(unittest.TestCase):
    def test_aggregates_segments_videos_and_duration(self):
        master = {
            "videos": {
                "v1": {"annotations": [{"start": 1, "end": 4, "coarse_label": "C", "fine_label": "F"}]},
                "v2": {"annotations": [{"start": 0, "end": 2, "coarse_label": "C", "fine_label": "F"}]},
            }
        }
        rows = build_label_distribution(master)
        coarse = next(row for row in rows if row["level"] == "coarse")
        self.assertEqual(coarse["segment_count"], 2)
        self.assertEqual(coarse["video_count"], 2)
        self.assertEqual(coarse["total_annotated_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()
