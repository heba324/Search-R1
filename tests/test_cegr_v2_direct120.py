import subprocess
import sys
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
        self.assertEqual(DIRECT120.success.minimum_evidence_coverage_delta, -0.02)
        self.assertEqual(DIRECT120.success.maximum_response_length_growth, 0.15)

        payload = subprocess.check_output(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/improvement_v2/direct120_contract.py"),
                "--json",
            ],
            cwd=REPO_ROOT,
            text=True,
        )
        self.assertIn('"DIRECT120_GROUP_SIZE": 5', payload)
        self.assertIn('"DIRECT120_SEED": 42', payload)

    def test_direct_training_is_equal_budget_from_the_original_qwen_model(self):
        smoke = (REPO_ROOT / "scripts/improvement_v2/run_direct120_smoke.sh").read_text(
            encoding="utf-8"
        )
        train = (REPO_ROOT / "scripts/improvement_v2/run_direct120.sh").read_text(
            encoding="utf-8"
        )

        for script in (smoke, train):
            self.assertIn("direct120_contract.py", script)
            self.assertIn("--shell", script)
            self.assertIn('GROUP_SIZE="$DIRECT120_GROUP_SIZE"', script)
            self.assertIn('SEED="$DIRECT120_SEED"', script)
            self.assertIn('ROLLOUT_ENGINE_SEED="$DIRECT120_ROLLOUT_ENGINE_SEED"', script)
            self.assertIn('MODE="$DIRECT120_REWARD_MODE"', script)
            self.assertIn('--seed "$DIRECT120_SEED"', script)
            self.assertIn(
                '--rollout-engine-seed "$DIRECT120_ROLLOUT_ENGINE_SEED"',
                script,
            )
            self.assertIn("-m scripts.improvement_v2.verify_training_run", script)
        self.assertIn('TOTAL_STEPS="$DIRECT120_SMOKE_STEPS"', smoke)
        self.assertIn('TOTAL_STEPS="$DIRECT120_TRAINING_STEPS"', train)
        self.assertIn('SAVE_FREQ="$DIRECT120_SAVE_FREQ"', train)
        self.assertIn("scripts.improvement_v2.verify_training_run", train)
        self.assertIn('--learning-rate "$DIRECT120_LEARNING_RATE"', train)
        self.assertIn('--lr-warmup-ratio "$DIRECT120_LR_WARMUP_RATIO"', train)
        self.assertIn('--train-batch-size "$DIRECT120_TRAIN_BATCH_SIZE"', train)

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

        self.assertIn("DIRECT120_BASELINE_RUN", script)
        self.assertIn('evaluate_one baseline "$BASELINE_MODEL"', script)
        self.assertIn('evaluate_one cegr-v2-direct120 "$DIRECT_MODEL"', script)
        self.assertIn('EVAL_BATCH_SIZE="$DIRECT120_EVAL_BATCH_SIZE"', script)
        self.assertIn('SEED="$DIRECT120_SEED"', script)
        self.assertIn('ROLLOUT_ENGINE_SEED="$DIRECT120_ROLLOUT_ENGINE_SEED"', script)
        self.assertIn("scripts.improvement_v2.direct120_analysis", script)
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
        self.assertEqual(
            report["effectiveness"]["success_criteria"][
                "minimum_evidence_coverage_delta"
            ],
            -0.02,
        )

        improved[0]["em"] = improved[0]["f1"] = 0.0
        report = analyze_direct120(
            baseline,
            improved,
            expected_datasets=DATASETS,
            expected_per_dataset=50,
            bootstrap_samples=100,
        )
        self.assertFalse(report["effectiveness"]["predeclared_success"])

    def test_direct_final_distinguishes_primary_gain_from_guardrail_failure(self):
        from scripts.improvement_v2.direct120_analysis import analyze_direct120

        baseline = []
        improved = []
        for dataset in DATASETS:
            for index in range(50):
                baseline.append(_record(dataset, index, 0))
                row = _record(
                    dataset,
                    index,
                    int(index == 0 or (dataset == "nq" and index == 1)),
                )
                row["duplicate_searches"] = 0.03
                improved.append(row)

        report = analyze_direct120(
            baseline,
            improved,
            expected_datasets=DATASETS,
            expected_per_dataset=50,
            bootstrap_samples=100,
        )

        self.assertEqual(report["comparison"]["overall"]["em_delta"], 0.03)
        self.assertTrue(report["effectiveness"]["primary_metric_pass"])
        self.assertFalse(report["effectiveness"]["guardrails_pass"])
        self.assertEqual(
            report["effectiveness"]["claim_level"],
            "primary_gain_with_guardrail_failure",
        )

    def test_module_entrypoints_import_from_a_clean_checkout(self):
        for module in (
            "scripts.improvement_v2.verify_training_run",
            "scripts.improvement_v2.direct120_analysis",
        ):
            completed = subprocess.run(
                [sys.executable, "-m", module, "--help"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_evidence_collector_can_require_and_pack_the_direct_checkpoint(self):
        script = (
            REPO_ROOT / "scripts/improvement_v2/collect_evidence.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("REQUIRE_DIRECT120_CHECKPOINT", script)
        self.assertIn("DIRECT120_CHECKPOINT", script)
        self.assertIn("direct120-checkpoint.sha256", script)

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
