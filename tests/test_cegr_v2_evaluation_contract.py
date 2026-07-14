import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class CEGRV2EvaluationContractTests(unittest.TestCase):
    def test_evaluation_writes_only_to_v2_artifacts_and_captures_trajectories(self):
        script = (REPO_ROOT / "scripts/improvement_v2/evaluate_model.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("artifacts/improvement-v2/evaluation", script)
        self.assertIn("SEARCH_R1_EVAL_TRAJECTORIES", script)
        self.assertIn("scripts.improvement_v2.main_ppo_refinement", script)
        self.assertIn('+reward_strategy.seed="$SEED"', script)
        self.assertIn('+actor_rollout_ref.rollout.engine_seed="$ROLLOUT_ENGINE_SEED"', script)
        self.assertIn("Already completed; preserving evaluation", script)
        self.assertIn("Refusing to overwrite a partial evaluation", script)
        self.assertNotIn("artifacts/course-reproduction/evaluation", script)

    def test_pilot_evaluates_three_models_and_applies_fail_closed_gate(self):
        script = (REPO_ROOT / "scripts/improvement_v2/evaluate_pilot.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("baseline", script)
        self.assertIn("em-control", script)
        self.assertIn("cegr-v2", script)
        self.assertIn("EVAL_BATCH_SIZE=20", script)
        self.assertIn("verify_training_run.py", script)
        self.assertIn("pilot_gate.py", script)
        self.assertIn("verify_pilot_gate.py", script)

    def test_final_evaluation_uses_all_700_examples_and_both_comparators(self):
        script = (REPO_ROOT / "scripts/improvement_v2/evaluate_final.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("data/course_eval/test.parquet", script)
        self.assertIn("EVAL_BATCH_SIZE=28", script)
        self.assertIn("baseline", script)
        self.assertIn("em-control", script)
        self.assertIn("cegr-v2", script)
        self.assertIn("final_analysis.py", script)
        self.assertIn("artifacts/improvement/paired-evaluation/baseline.jsonl", script)
        self.assertIn('evaluate_one baseline "$BASELINE_MODEL"', script)
        self.assertIn("historical-baseline-rescored.jsonl", script)
        self.assertIn("verify_pilot_data.py", script)
        self.assertIn("verify_pilot_gate.py", script)
        self.assertNotIn('python3 "$SCRIPT_DIR/pilot_gate.py"', script)
        self.assertIn('--pilot-gate "$PILOT_GATE"', script)
        self.assertIn("rescore_frozen_baseline.py", script)

    def test_v2_evaluation_record_uses_strict_answer_tags(self):
        from scripts.improvement_v2.evaluation_record import build_evaluation_record

        strict = build_evaluation_record(
            "<ANSWER>Beijing</ANSWER>",
            dataset="nq",
            golden_answers=["Beijing"],
            extra_info={"split": "test", "index": 1},
        )
        valid = build_evaluation_record(
            "<answer>Beijing</answer>",
            dataset="nq",
            golden_answers=["Beijing"],
            extra_info={"split": "test", "index": 1},
        )

        self.assertEqual(strict["em"], 0.0)
        self.assertEqual(valid["em"], 1.0)

    def test_v2_behavior_diagnostics_follow_case_sensitive_environment_actions(self):
        from scripts.improvement_v2.evaluation_record import build_evaluation_record

        record = build_evaluation_record(
            (
                "<SEARCH>ignored query</SEARCH>"
                "<INFORMATION>Beijing</INFORMATION>"
                "<answer>wrong</answer>"
            ),
            dataset="nq",
            golden_answers=["Beijing"],
            extra_info={"split": "test", "index": 1},
        )

        self.assertEqual(record["searches"], 0)
        self.assertEqual(record["evidence_coverage"], 0.0)

    def test_frozen_baseline_is_rescored_without_changing_identity_or_trajectory(self):
        from scripts.improvement_v2.rescore_frozen_baseline import rescore_records

        original = [
            {
                "example_id": "nq:test:1",
                "dataset": "nq",
                "split": "test",
                "index": 1,
                "golden_answers": ["Beijing"],
                "trajectory": "<ANSWER>Beijing</ANSWER>",
                "em": 1.0,
                "response_tokens": 12,
            }
        ]

        rescored, report = rescore_records(original)

        self.assertEqual(rescored[0]["example_id"], original[0]["example_id"])
        self.assertEqual(rescored[0]["trajectory"], original[0]["trajectory"])
        self.assertEqual(rescored[0]["em"], 0.0)
        self.assertEqual(report["parser_mismatch_count"], 1)
        self.assertFalse(report["regenerated"])

    def test_record_em_must_match_official_evaluation_metrics(self):
        from scripts.improvement_v2.verify_evaluation_records import verify_records

        marker = {"metrics": {"nq": 0.5, "hotpotqa": 1.0}}
        records = [
            {"example_id": "nq:test:0", "dataset": "nq", "em": 1.0},
            {"example_id": "nq:test:1", "dataset": "nq", "em": 0.0},
            {"example_id": "hotpotqa:test:0", "dataset": "hotpotqa", "em": 1.0},
        ]

        self.assertEqual(verify_records(marker, records), [])
        records[0]["em"] = 0.0
        self.assertIn("does not match official metric", verify_records(marker, records)[0])


if __name__ == "__main__":
    unittest.main()
