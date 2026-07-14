import json
import tempfile
import unittest
from pathlib import Path


class CEGRV2MetricsTests(unittest.TestCase):
    def test_metric_validation_enforces_grouping_and_em_preservation(self):
        from scripts.improvement_v2.parse_v2_metrics import validate_metrics

        valid = [
            {
                "group_size": 5,
                "group_count": 32,
                "mixed_group_reward_mismatches": 0,
            }
        ]
        invalid = [
            {
                "group_size": 1,
                "group_count": 160,
                "mixed_group_reward_mismatches": 2,
            }
        ]

        self.assertEqual(validate_metrics(valid), [])
        errors = validate_metrics(invalid)
        self.assertEqual(len(errors), 2)

    def test_eff_smoke_requires_an_informative_fallback_signal(self):
        from scripts.improvement_v2.parse_v2_metrics import validate_metrics

        informative = [
            {
                "mode": "eff",
                "group_size": 5,
                "group_count": 8,
                "all_zero_group_count": 4,
                "informative_fallback_group_count": 1,
                "mixed_group_reward_mismatches": 0,
            }
        ]
        uninformative = [
            {
                "mode": "eff",
                "group_size": 5,
                "group_count": 8,
                "all_zero_group_count": 4,
                "informative_fallback_group_count": 0,
                "mixed_group_reward_mismatches": 0,
            }
        ]

        self.assertEqual(
            validate_metrics(informative, minimum_informative_fallback_rate=0.1), []
        )
        self.assertIn(
            "informative fallback rate",
            validate_metrics(
                uninformative, minimum_informative_fallback_rate=0.1
            )[0],
        )

    def test_metric_validation_rejects_early_or_nonconsecutive_completion(self):
        from scripts.improvement_v2.parse_v2_metrics import validate_metrics

        rows = [
            {
                "step": 1,
                "mode": "eff",
                "group_size": 5,
                "group_count": 8,
                "mixed_group_reward_mismatches": 0,
            },
            {
                "step": 3,
                "mode": "eff",
                "group_size": 5,
                "group_count": 8,
                "mixed_group_reward_mismatches": 0,
            },
        ]

        errors = validate_metrics(rows, expected_steps=3, expected_group_size=5)

        self.assertTrue(any("Expected 3" in error for error in errors))
        self.assertTrue(any("not consecutive" in error for error in errors))

    def test_training_completion_requires_marker_metrics_and_checkpoint(self):
        from scripts.improvement_v2.verify_training_run import verify_training_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_name = "smoke"
            artifact = root / "artifacts/improvement-v2" / run_name
            checkpoint = root / "verl_checkpoints" / run_name / "actor/global_step_2"
            artifact.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            (artifact / "training_completed.txt").write_text(
                "status=completed\nmethod=eff\ntraining_steps=2\ngroup_size=5\n"
                "seed=42\nrollout_engine_seed=42\ntrain_batch_size=8\n"
                "learning_rate=1e-6\nlr_warmup_steps_ratio=0.95\n",
                encoding="utf-8",
            )
            (artifact / "train.log").write_text(
                "actor/loss=0.25\n", encoding="utf-8"
            )
            (artifact / "reward_metrics.json").write_text(
                json.dumps(
                    {
                        "method": "eff",
                        "signal_summary": {
                            "all_zero_group_count": 8,
                            "informative_fallback_group_count": 2,
                            "informative_fallback_rate": 0.25,
                        },
                        "steps": [
                            {
                                "step": 1,
                                "mode": "eff",
                                "group_size": 5,
                                "group_count": 8,
                                "all_zero_group_count": 4,
                                "informative_fallback_group_count": 1,
                                "mixed_group_reward_mismatches": 0,
                            },
                            {
                                "step": 2,
                                "mode": "eff",
                                "group_size": 5,
                                "group_count": 8,
                                "all_zero_group_count": 4,
                                "informative_fallback_group_count": 1,
                                "mixed_group_reward_mismatches": 0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                verify_training_run(
                    root,
                    run_name,
                    "eff",
                    2,
                    5,
                    minimum_signal=0.1,
                    seed=42,
                    train_batch_size=8,
                    learning_rate=1e-6,
                    lr_warmup_ratio=0.95,
                ),
                [],
            )
            self.assertTrue(
                any(
                    "driver seed" in error
                    for error in verify_training_run(
                        root, run_name, "eff", 2, 5, seed=7
                    )
                )
            )
            self.assertTrue(
                any(
                    "warmup" in error
                    for error in verify_training_run(
                        root,
                        run_name,
                        "eff",
                        2,
                        5,
                        lr_warmup_ratio=0.0,
                    )
                )
            )
            (checkpoint / "config.json").unlink()
            self.assertIn(
                "checkpoint",
                verify_training_run(root, run_name, "eff", 2, 5)[0],
            )

    def test_training_completion_rechecks_reward_invariants(self):
        from scripts.improvement_v2.verify_training_run import verify_training_run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_name = "tampered"
            artifact = root / "artifacts/improvement-v2" / run_name
            checkpoint = root / "verl_checkpoints" / run_name / "actor/global_step_1"
            artifact.mkdir(parents=True)
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            (artifact / "training_completed.txt").write_text(
                "status=completed\nmethod=eff\ntraining_steps=1\ngroup_size=5\n"
                "rollout_engine_seed=42\n",
                encoding="utf-8",
            )
            (artifact / "train.log").write_text(
                "actor/loss=0.25\n", encoding="utf-8"
            )
            (artifact / "reward_metrics.json").write_text(
                json.dumps(
                    {
                        "method": "eff",
                        "signal_summary": {
                            "all_zero_group_count": 1,
                            "informative_fallback_group_count": 1,
                            "informative_fallback_rate": 1.0,
                        },
                        "steps": [
                            {
                                "step": 1,
                                "mode": "eff",
                                "group_size": 5,
                                "group_count": 8,
                                "all_zero_group_count": 1,
                                "informative_fallback_group_count": 1,
                                "mixed_group_reward_mismatches": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            errors = verify_training_run(root, run_name, "eff", 1, 5)

            self.assertTrue(any("diverged from EM" in error for error in errors))

    def test_training_completion_rejects_nonfinite_logged_metrics(self):
        from scripts.improvement_v2.parse_v2_metrics import find_nonfinite_tokens

        self.assertEqual(find_nonfinite_tokens("actor/loss=0.25 reward=1.0"), [])
        self.assertEqual(find_nonfinite_tokens("actor/loss=nan"), ["nan"])
        self.assertEqual(find_nonfinite_tokens("grad_norm=Inf"), ["inf"])


if __name__ == "__main__":
    unittest.main()
