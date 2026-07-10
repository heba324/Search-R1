import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.paper_v1.contract import (
    PAPER_V1,
    REQUIRED_EVALUATION_DATASETS,
    assert_paper_commit,
    assess_required_assets,
    assess_result_metrics,
)


class PaperV1ContractTests(unittest.TestCase):
    def test_frozen_paper_v1_constants(self):
        self.assertEqual(PAPER_V1.git_commit, "118c6e7361bb68e33c525b50d62f83b63462799e")
        self.assertEqual(PAPER_V1.model_id, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(PAPER_V1.algorithm, "ppo")
        self.assertEqual(PAPER_V1.training_steps, 305)
        self.assertEqual(PAPER_V1.max_turns, 4)
        self.assertEqual(PAPER_V1.topk, 3)
        self.assertEqual(PAPER_V1.target_average_em, 0.327)
        self.assertEqual(PAPER_V1.dataset_revision, "b7d80abfee334a7a91cb377544f09180d58b34f6")

    def test_commit_guard_rejects_wrong_source_revision(self):
        with self.assertRaisesRegex(ValueError, "118c6e7"):
            assert_paper_commit("598e61bd1d36895726d28a8d06b3a15bed19f5d3")

    def test_asset_assessment_requires_published_train_and_test_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            errors = assess_required_assets(root)
        message = "\n".join(errors)
        self.assertIn("train.parquet", message)
        self.assertIn("test.parquet", message)
        self.assertIn("e5_Flat.index", message)
        self.assertIn("wiki-18.jsonl", message)

    def test_metric_assessment_requires_all_paper_datasets(self):
        errors = assess_result_metrics({"nq": 0.4})
        message = "\n".join(errors)
        self.assertIn("triviaqa", message)
        self.assertEqual(
            REQUIRED_EVALUATION_DATASETS,
            ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"),
        )


if __name__ == "__main__":
    unittest.main()
