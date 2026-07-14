import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CEGRV2ContractTests(unittest.TestCase):
    def test_v2_is_an_equal_budget_refinement_from_the_frozen_baseline(self):
        from scripts.improvement_v2.contract import CEGR_V2

        self.assertEqual(CEGR_V2.method, "em_first_f1_fallback")
        self.assertEqual(
            CEGR_V2.v1_commit, "8672aad0f4089f0fca388601cd9ce20fc9b8b776"
        )
        self.assertEqual(CEGR_V2.initial_checkpoint_step, 120)
        self.assertEqual(CEGR_V2.refinement_steps, 40)
        self.assertEqual(CEGR_V2.equal_budget_control, "grouped_em")
        self.assertEqual(CEGR_V2.group_size, 5)
        self.assertEqual(CEGR_V2.seed, 42)
        self.assertEqual(CEGR_V2.rollout_engine_seed, 42)
        self.assertEqual(CEGR_V2.pilot_examples_per_dataset, 20)
        self.assertEqual(CEGR_V2.final_examples_per_dataset, 100)

    def test_training_entrypoint_isolated_from_v1_names_and_artifacts(self):
        script = (REPO_ROOT / "scripts/improvement_v2/train_refinement.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('MODE="${MODE:-eff}"', script)
        self.assertIn('TOTAL_STEPS="${TOTAL_STEPS:-40}"', script)
        self.assertIn("search-r1-course-qwen2.5-1.5b-grpo-bm25", script)
        self.assertIn("scripts.improvement_v2.main_ppo_refinement", script)
        self.assertIn("artifacts/improvement-v2", script)
        self.assertIn("search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25", script)
        self.assertIn("search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25", script)
        self.assertIn("Refusing to overwrite V2 artifact directory", script)
        self.assertNotIn("artifacts/improvement/", script)
        self.assertNotIn("search-r1-cegr-qwen2.5-1.5b-grpo-bm25", script)

    def test_pilot_runs_equal_compute_control_and_v2(self):
        script = (REPO_ROOT / "scripts/improvement_v2/run_pilot.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("run_arm grouped_em", script)
        self.assertIn("run_arm eff", script)
        self.assertIn("prepare_pilot_data.py", script)
        self.assertIn("verify_training_run.py", script)
        self.assertIn("Partial arm found", script)
        self.assertIn("Already completed", script)

    def test_both_training_arms_receive_the_same_explicit_seed(self):
        script = (REPO_ROOT / "scripts/improvement_v2/train_refinement.sh").read_text(
            encoding="utf-8"
        )
        entrypoint = (
            REPO_ROOT / "scripts/improvement_v2/main_ppo_refinement.py"
        ).read_text(encoding="utf-8")

        self.assertIn('SEED="${SEED:-42}"', script)
        self.assertIn('+reward_strategy.seed="$SEED"', script)
        self.assertIn(
            '+actor_rollout_ref.rollout.engine_seed="$ROLLOUT_ENGINE_SEED"',
            script,
        )
        self.assertIn("torch.manual_seed(seed)", entrypoint)
        self.assertIn("np.random.seed(seed)", entrypoint)
        self.assertIn("CEGRV2ActorRolloutRefWorker as ActorRolloutRefWorker", entrypoint)

    def test_vllm_engine_seed_is_not_mistaken_for_a_sampling_seed(self):
        from types import SimpleNamespace

        from scripts.improvement_v2.vllm_seed import override_vllm_engine_seed

        captured = {}

        def fake_llm(*args, **kwargs):
            captured.update(kwargs)
            return object()

        module = SimpleNamespace(LLM=fake_llm)
        with override_vllm_engine_seed(module, 42):
            module.LLM("actor", temperature=1.0)

        self.assertEqual(captured["seed"], 42)
        self.assertNotIn("engine_seed", captured)
        self.assertIs(module.LLM, fake_llm)

    def test_smoke_uses_disjoint_pilot_data_and_enforces_signal_gate(self):
        script = (REPO_ROOT / "scripts/improvement_v2/run_smoke.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("data/improvement_v2/pilot.parquet", script)
        self.assertIn("MIN_INFORMATIVE_FALLBACK_RATE=0.10", script)
        self.assertIn("verify_training_run.py", script)
        self.assertNotIn("data/course_eval/test.parquet", script)

    def test_v2_reward_manager_groups_rollouts_without_behavior_or_evidence_reward(self):
        manager = (REPO_ROOT / "scripts/improvement_v2/cegr_v2_manager.py").read_text(
            encoding="utf-8"
        )
        reward = (REPO_ROOT / "scripts/improvement_v2/cegr_v2_reward.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("CEGRV2RewardManager", manager)
        self.assertIn("build_group_uids", manager)
        self.assertNotIn("search_behavior_penalty", reward)
        self.assertNotIn("evidence_answer_coverage", reward)

    def test_v1_asset_linker_shares_only_inputs_and_keeps_v2_outputs_local(self):
        script = (REPO_ROOT / "scripts/improvement_v2/link_v1_assets.sh").read_text(
            encoding="utf-8"
        )

        for path in ("nq_hotpotqa_train", "models", "course_eval", "wiki18_bm25"):
            self.assertIn(path, script)
        self.assertIn("search-r1-course-qwen2.5-1.5b-grpo-bm25", script)
        self.assertIn("search-r1-cegr-qwen2.5-1.5b-grpo-bm25", script)
        self.assertNotIn('link_once "$V1_ROOT/data" "$REPO_ROOT/data"', script)
        self.assertNotIn("artifacts/improvement-v2", script)

    def test_offline_preparation_checks_v1_freeze_tests_and_core_diff(self):
        script = (
            REPO_ROOT / "scripts/improvement_v2/prepare_experiment.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("freeze_v1.py --repo-root", script)
        self.assertIn("--initialize", script)
        self.assertIn("prepare_pilot_data.py", script)
        self.assertIn("unittest discover", script)
        self.assertIn("search_r1 verl", script)
        self.assertIn("scripts/improvement docs/improvement_experiment_zh.md", script)

    def test_v1_linker_requires_the_frozen_source_commit(self):
        script = (REPO_ROOT / "scripts/improvement_v2/link_v1_assets.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('git -C "$V1_ROOT" rev-parse HEAD', script)
        self.assertIn("8672aad0f4089f0fca388601cd9ce20fc9b8b776", script)

    def test_v2_shell_entrypoints_are_executable(self):
        output = subprocess.check_output(
            ["git", "ls-files", "--stage", "scripts/improvement_v2/*.sh"],
            cwd=REPO_ROOT,
            text=True,
        )
        modes = [line.split()[0] for line in output.splitlines()]

        self.assertTrue(modes)
        self.assertEqual(set(modes), {"100755"})


if __name__ == "__main__":
    unittest.main()
