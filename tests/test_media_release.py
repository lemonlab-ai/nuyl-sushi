import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nuyl_sushi.data.media_release import (
    MediaProbe,
    build_ffmpeg_command,
    decide_release_action,
    load_media_profiles,
    parse_frame_rate,
    resolve_manifest_source,
)


class MediaReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.profiles = load_media_profiles(
            ROOT_DIR / "configs" / "datasets" / "media_profiles.toml"
        )

    def test_parse_fractional_frame_rate(self) -> None:
        self.assertAlmostEqual(parse_frame_rate("30000/1001") or 0.0, 29.97003, places=4)
        self.assertIsNone(parse_frame_rate("0/0"))

    def test_compliant_research_video_is_copied(self) -> None:
        probe = MediaProbe(
            status="ok",
            codec="h264",
            width=1280,
            height=720,
            pixel_format="yuv420p",
            frame_rate=29.97,
            duration_seconds=10.0,
            audio_streams=0,
        )
        decision = decide_release_action(self.profiles["research-720p"], probe)
        self.assertEqual(decision.action, "copy")
        self.assertEqual(decision.reasons, ("profile_compliant",))

    def test_noncompliant_video_is_planned_for_transcode(self) -> None:
        probe = MediaProbe(
            status="ok",
            codec="hevc",
            width=1920,
            height=1080,
            pixel_format="yuv420p10le",
            frame_rate=60.0,
            duration_seconds=10.0,
            audio_streams=0,
        )
        decision = decide_release_action(self.profiles["research-720p"], probe)
        self.assertEqual(decision.action, "transcode")
        self.assertIn("video_codec", decision.reasons)
        self.assertIn("width", decision.reasons)

    def test_research_audio_requires_explicit_review(self) -> None:
        probe = MediaProbe(
            status="ok",
            codec="h264",
            width=1280,
            height=720,
            pixel_format="yuv420p",
            frame_rate=30.0,
            duration_seconds=10.0,
            audio_streams=1,
        )
        decision = decide_release_action(self.profiles["research-720p"], probe)
        self.assertEqual(decision.action, "blocked")
        self.assertEqual(decision.reasons, ("audio_consent_review_required",))

    def test_missing_probe_fields_are_not_guessed(self) -> None:
        decision = decide_release_action(
            self.profiles["research-720p"],
            MediaProbe(status="ok", width=1280, height=720),
        )
        self.assertEqual(decision.action, "blocked")
        self.assertIn("unknown_codec", decision.reasons)
        self.assertIn("unknown_audio_streams", decision.reasons)

    def test_preview_command_never_overwrites(self) -> None:
        command = build_ffmpeg_command(
            "input.mp4", "output.mp4", self.profiles["preview-360p"]
        )
        self.assertIn("-an", command)
        self.assertIn("-n", command)
        self.assertIn("fps=15", command[command.index("-vf") + 1])

    def test_manifest_filename_cannot_escape_source_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertIsNone(resolve_manifest_source(root, "../outside.mp4"))
            self.assertEqual(
                resolve_manifest_source(root, "inside.mp4"),
                (root / "inside.mp4").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
