import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ImprovementContractTests(unittest.TestCase):
    def test_cegr_contract_is_a_single_variable_baseline_comparison(self):
        from scripts.improvement.contract import CEGR_EXPERIMENT

        self.assertEqual(CEGR_EXPERIMENT.model_id, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(CEGR_EXPERIMENT.algorithm, "grpo")
        self.assertEqual(CEGR_EXPERIMENT.retriever, "bm25")
        self.assertEqual(CEGR_EXPERIMENT.training_steps, 120)
        self.assertEqual(CEGR_EXPERIMENT.train_batch_size, 32)
        self.assertEqual(CEGR_EXPERIMENT.group_size, 5)
        self.assertEqual(CEGR_EXPERIMENT.max_turns, 4)
        self.assertEqual(CEGR_EXPERIMENT.topk, 3)
        self.assertEqual(CEGR_EXPERIMENT.seed, 42)
        self.assertEqual(CEGR_EXPERIMENT.changed_variable, "reward")

    def test_training_script_enables_cegr_without_changing_core_budget(self):
        script = (REPO_ROOT / "scripts/improvement/train_cegr.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('TOTAL_STEPS="${TOTAL_STEPS:-120}"', script)
        self.assertIn('TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"', script)
        self.assertIn('VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"', script)
        self.assertIn('GROUP_SIZE="${GROUP_SIZE:-5}"', script)
        self.assertIn("scripts.improvement.main_ppo_cegr", script)
        self.assertIn("+reward_strategy.name=cegr", script)
        self.assertIn("max_turns=4", script)
        self.assertIn("retriever.topk=3", script)
        self.assertIn("data.max_obs_length=500", script)
        self.assertIn("actor_rollout_ref.actor.optim.lr=1e-6", script)
        self.assertIn("PYTHONHASHSEED", script)

    def test_evaluation_reruns_both_models_and_saves_paired_records(self):
        script = (REPO_ROOT / "scripts/improvement/evaluate_cegr.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("BASELINE_MODEL_PATH", script)
        self.assertIn("SEARCH_R1_EVAL_TRAJECTORIES", script)
        self.assertIn("baseline.jsonl", script)
        self.assertIn("cegr.jsonl", script)
        self.assertIn("analyze_paired_results.py", script)
        self.assertIn("--bootstrap-samples 10000 --seed 42", script)

    def test_smoke_and_timing_validation_batches_cover_all_seven_rows(self):
        smoke = (REPO_ROOT / "scripts/improvement/run_smoke.sh").read_text(
            encoding="utf-8"
        )
        timing = (REPO_ROOT / "scripts/improvement/run_timing.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("VAL_BATCH_SIZE=7", smoke)
        self.assertIn("VAL_DATA_NUM=7", smoke)
        self.assertIn("VAL_BATCH_SIZE=7", timing)
        self.assertIn("VAL_DATA_NUM=7", timing)

    def test_fixed_evaluation_uses_a_batch_that_divides_700(self):
        script = (REPO_ROOT / "scripts/course_reproduction/evaluate.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-28}"', script)
        self.assertIn('data.val_batch_size="$EVAL_BATCH_SIZE"', script)
        self.assertIn('actor_rollout_ref.actor.ppo_mini_batch_size="$EVAL_BATCH_SIZE"', script)


if __name__ == "__main__":
    unittest.main()
