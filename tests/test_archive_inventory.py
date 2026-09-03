import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nuyl_sushi.data.archive_inventory import build_archive_inventory


class ArchiveInventoryTests(unittest.TestCase):
    def test_inventory_is_complete_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            source = base / "source"
            output = base / "inventory"
            (source / "nested").mkdir(parents=True)
            (source / "a.txt").write_text("sushi", encoding="utf-8")
            (source / "nested" / "b.bin").write_bytes(b"\x00\x01")
            (source / "nested" / "a-copy.txt").write_text("sushi", encoding="utf-8")

            first = build_archive_inventory(source, output, "fixture-v1")
            second = build_archive_inventory(source, output, "fixture-v1")

            self.assertTrue(first["complete"])
            self.assertEqual(first["files"], 3)
            self.assertEqual(first["reused_files_current_run"], 0)
            self.assertEqual(second["reused_files_current_run"], 3)
            self.assertEqual(first["duplicate_hash_groups"], 1)
            self.assertEqual(first["duplicate_excess_bytes"], 5)
            self.assertEqual(first["tree_sha256"], second["tree_sha256"])
            rows = [
                json.loads(line)
                for line in (output / "inventory.checkpoint.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["status"] == "ok" for row in rows))

    def test_output_inside_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                build_archive_inventory(source, source / "inventory", "fixture-v1")


if __name__ == "__main__":
    unittest.main()
