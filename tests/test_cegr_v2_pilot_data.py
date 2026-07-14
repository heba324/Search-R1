import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class CEGRV2PilotDataTests(unittest.TestCase):
    def test_selection_is_deterministic_and_disjoint_from_final_evaluation(self):
        from scripts.improvement_v2.prepare_pilot_data import select_disjoint_indices

        excluded = {0, 2, 4, 6}
        first = select_disjoint_indices(20, excluded, count=5, seed=42)
        second = select_disjoint_indices(20, excluded, count=5, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(set(first).isdisjoint(excluded))

    def test_selection_fails_when_not_enough_unseen_examples_exist(self):
        from scripts.improvement_v2.prepare_pilot_data import select_disjoint_indices

        with self.assertRaisesRegex(ValueError, "unseen examples"):
            select_disjoint_indices(5, {0, 1, 2, 3}, count=2, seed=42)

    def test_manifest_verification_detects_pilot_or_final_data_changes(self):
        from scripts.improvement_v2.verify_pilot_data import verify_pilot_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / "pilot.parquet"
            final = root / "final.parquet"
            manifest = root / "manifest.json"
            pilot.write_bytes(b"pilot-v1")
            final.write_bytes(b"final-v1")
            manifest.write_text(
                json.dumps(
                    {
                        "pilot_sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(),
                        "pilot_bytes": pilot.stat().st_size,
                        "excluded_final_eval_sha256": hashlib.sha256(
                            final.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(verify_pilot_files(pilot, final, manifest), [])
            pilot.write_bytes(b"pilot-v2")
            self.assertIn(
                "pilot SHA-256 changed",
                verify_pilot_files(pilot, final, manifest)[0],
            )


if __name__ == "__main__":
    unittest.main()
