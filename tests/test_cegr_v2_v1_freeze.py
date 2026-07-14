import tempfile
import unittest
from pathlib import Path


class CEGRV2V1FreezeTests(unittest.TestCase):
    def test_manifest_detects_later_changes_to_frozen_v1_evidence(self):
        from scripts.improvement_v2.freeze_v1 import create_manifest, verify_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "checkpoint-config.json"
            second = root / "paired-analysis.json"
            first.write_text('{"model": "v1"}\n', encoding="utf-8")
            second.write_text('{"em": 0.2071}\n', encoding="utf-8")
            relative_paths = (first.name, second.name)
            manifest = create_manifest(root, relative_paths)

            self.assertEqual(verify_manifest(root, manifest), [])
            second.write_text('{"em": 0.9999}\n', encoding="utf-8")
            errors = verify_manifest(root, manifest)

            self.assertEqual(len(errors), 1)
            self.assertIn("SHA-256 changed", errors[0])

    def test_manifest_creation_fails_closed_when_v1_file_is_missing(self):
        from scripts.improvement_v2.freeze_v1 import create_manifest

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Missing V1 evidence"):
                create_manifest(Path(directory), ("missing.json",))

    def test_manifest_recursively_freezes_checkpoint_weights(self):
        from scripts.improvement_v2.freeze_v1 import create_manifest, verify_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            weights = checkpoint / "model.safetensors"
            config = checkpoint / "config.json"
            weights.write_bytes(b"frozen-weights")
            config.write_text('{"step": 120}\n', encoding="utf-8")

            manifest = create_manifest(root, ("checkpoint",))

            self.assertIn("checkpoint/model.safetensors", manifest["files"])
            weights.write_bytes(b"broken-weights")
            self.assertIn("SHA-256 changed", verify_manifest(root, manifest)[0])

    def test_manifest_detects_files_added_after_freeze(self):
        from scripts.improvement_v2.freeze_v1 import create_manifest, verify_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            frozen.mkdir()
            (frozen / "original.json").write_text("{}\n", encoding="utf-8")
            manifest = create_manifest(root, ("frozen",))

            (frozen / "added.json").write_text("{}\n", encoding="utf-8")

            errors = verify_manifest(root, manifest)
            self.assertTrue(any("Unexpected frozen V1 file" in error for error in errors))

    def test_freeze_contract_includes_original_evaluation_evidence(self):
        from scripts.improvement_v2.freeze_v1 import V1_EVIDENCE_PATHS

        self.assertIn(
            "artifacts/course-reproduction/evaluation/baseline-paired",
            V1_EVIDENCE_PATHS,
        )
        self.assertIn(
            "artifacts/course-reproduction/evaluation/cegr-post-rl",
            V1_EVIDENCE_PATHS,
        )


if __name__ == "__main__":
    unittest.main()
