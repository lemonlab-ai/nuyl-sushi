import tempfile
import unittest
from pathlib import Path


from scripts.audit_source_dataset import infer_source_metadata, normalize_session, sha256_file


class AuditSourceDatasetTests(unittest.TestCase):
    def test_normalize_session(self) -> None:
        self.assertEqual(normalize_session(["edited", "230208", "view1"]), "2023-02-08")
        self.assertEqual(normalize_session(["20221215", "sensor"]), "2022-12-15")
        self.assertEqual(normalize_session(["unknown"]), "unknown")

    def test_infer_source_metadata(self) -> None:
        metadata = infer_source_metadata(
            [Path("root/20230204/view2/IMG_7321.mp4"), Path("root/edited/IMG_7321.mp4")]
        )
        self.assertEqual(metadata["view_guess"], "view2")
        self.assertEqual(metadata["session_guess"], "2023-02-04")

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.bin"
            path.write_bytes(b"nuyl-sushi")
            self.assertEqual(
                sha256_file(path),
                "f81e39dd2cff8a912fdfd8f91a0a9b6dcdace0c1a9a38bdca424e1d06eac6c3f",
            )


if __name__ == "__main__":
    unittest.main()
