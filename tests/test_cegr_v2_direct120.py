import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("nq", "hotpotqa")


def _record(dataset, index, em):
    return {
        "example_id": f"{dataset}:test:{index}",
        "dataset": dataset,
        "em": float(em),
        "f1": float(em),
        "evidence_coverage": float(em),
        "searches": 1.0,
        "valid_searches": 1.0,
        "duplicate_searches": 0.0,
        "invalid_searches": 0.0,
        "response_tokens": 100,
    }


class CEGRV2Direct120Tests(unittest.TestCase):
    def test_direct120_contract_matches_the_existing_baseline_update_budget(self):
        from scripts.improvement_v2.direct120_contract import DIRECT120

        self.assertEqual(DIRECT120.initial_model, "data/models/Qwen2.5-1.5B-Instruct")
        self.assertEqual(DIRECT120.baseline_checkpoint_step, 120)
        self.assertEqual(DIRECT120.training_steps, 120)
        self.assertEqual(DIRECT120.train_batch_size, 32)
        self.assertEqual(DIRECT120.group_size, 5)
        self.assertEqual(DIRECT120.learning_rate, 1e-6)
        self.assertEqual(DIRECT120.lr_warmup_steps_ratio, 0.95)
        self.assertEqual(DIRECT120.minimum_em_gain, 0.02)
        self.assertEqual(DIRECT120.causal_estimand, "grouping_plus_eff")

    def test_direct_training_is_equal_budget_from_the_original_qwen_model(self):
        smoke = (REPO_ROOT / "scripts/improvement_v2/run_direct120_smoke.sh").read_text(
            encoding="utf-8"
        )
        train = (REPO_ROOT / "scripts/improvement_v2/run_direct120.sh").read_text(
            encoding="utf-8"
        )

        for script in (smoke, train):
            self.assertIn("data/models/Qwen2.5-1.5B-Instruct", script)
            self.assertIn("LEARNING_RATE=1e-6", script)
            self.assertIn("LR_WARMUP_STEPS_RATIO=0.95", script)
            self.assertIn("MODE=eff", script)
        self.assertIn("TOTAL_STEPS=2", smoke)
        self.assertIn("TOTAL_STEPS=120", train)
        self.assertIn("SAVE_FREQ=40", train)
        self.assertIn("verify_training_run.py", train)
        self.assertIn("--learning-rate 1e-6", train)
        self.assertIn("--lr-warmup-ratio 0.95", train)
        self.assertIn("--train-batch-size 32", train)
        self.assertIn("search-r1-cegr-v2-eff-direct120", train)

    def test_beginner_manual_explains_baseline_reuse_and_single_arm_limit(self):
        manual = (
            REPO_ROOT / "docs/cegr_v2_direct120_urgent_zh.md"
        ).read_text(encoding="utf-8")

        self.assertIn("baseline 不重新训练", manual)
        self.assertIn("baseline 必须重新评测", manual)
        self.assertIn("run_direct120_smoke.sh", manual)
        self.assertIn("run_direct120.sh", manual)
        self.assertIn("evaluate_direct120.sh", manual)
        self.assertIn("不能单独归因", manual)

    def test_direct_final_reuses_but_freshly_evaluates_the_baseline_checkpoint(self):
        script = (
            REPO_ROOT / "scripts/improvement_v2/evaluate_direct120.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("search-r1-course-qwen2.5-1.5b-grpo-bm25", script)
        self.assertIn('evaluate_one baseline "$BASELINE_MODEL"', script)
        self.assertIn('evaluate_one cegr-v2-direct120 "$DIRECT_MODEL"', script)
        self.assertIn("EVAL_BATCH_SIZE=28", script)
        self.assertIn("direct120_analysis.py", script)
        self.assertNotIn("train_grpo.sh", script)

    def test_direct_final_requires_a_two_point_gain(self):
        from scripts.improvement_v2.direct120_analysis import analyze_direct120

        baseline = []
        improved = []
        for dataset in DATASETS:
            for index in range(50):
                baseline.append(_record(dataset, index, 0))
                improved.append(_record(dataset, index, int(index == 0)))

        report = analyze_direct120(
            baseline,
            improved,
            expected_datasets=DATASETS,
            expected_per_dataset=50,
            bootstrap_samples=100,
        )

        self.assertEqual(report["comparison"]["overall"]["em_delta"], 0.02)
        self.assertTrue(report["effectiveness"]["predeclared_success"])
        self.assertEqual(report["effectiveness"]["minimum_em_gain"], 0.02)

        improved[0]["em"] = improved[0]["f1"] = 0.0
        report = analyze_direct120(
            baseline,
            improved,
            expected_datasets=DATASETS,
            expected_per_dataset=50,
            bootstrap_samples=100,
        )
        self.assertFalse(report["effectiveness"]["predeclared_success"])

    def test_direct_shell_entrypoints_are_executable(self):
        output = subprocess.check_output(
            ["git", "ls-files", "--stage", "scripts/improvement_v2/*.sh"],
            cwd=REPO_ROOT,
            text=True,
        )
        modes = {
            line.split("\t", 1)[1]: line.split()[0]
            for line in output.splitlines()
        }

        self.assertEqual(modes["scripts/improvement_v2/run_direct120_smoke.sh"], "100755")
        self.assertEqual(modes["scripts/improvement_v2/run_direct120.sh"], "100755")
        self.assertEqual(modes["scripts/improvement_v2/evaluate_direct120.sh"], "100755")


if __name__ == "__main__":
    unittest.main()
