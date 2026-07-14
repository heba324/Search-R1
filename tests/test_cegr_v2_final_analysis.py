import unittest


DATASETS = ("nq", "hotpotqa")


def _record(dataset, index, em, evidence=0):
    return {
        "example_id": f"{dataset}:test:{index}",
        "dataset": dataset,
        "em": float(em),
        "f1": float(em),
        "evidence_coverage": float(evidence),
        "searches": 1,
        "valid_searches": 1,
        "duplicate_searches": 0,
        "invalid_searches": 0,
        "response_tokens": 100,
    }


class CEGRV2FinalAnalysisTests(unittest.TestCase):
    def test_effectiveness_requires_beating_baseline_and_equal_budget_control(self):
        from scripts.improvement_v2.final_analysis import analyze_final

        baseline = [
            _record("nq", 0, 1),
            _record("nq", 1, 0),
            _record("hotpotqa", 0, 0),
            _record("hotpotqa", 1, 0),
        ]
        control = [
            _record("nq", 0, 1),
            _record("nq", 1, 1),
            _record("hotpotqa", 0, 0),
            _record("hotpotqa", 1, 0),
        ]
        v2 = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 1, 1),
            _record("hotpotqa", 0, 1, 1),
            _record("hotpotqa", 1, 0),
        ]

        report = analyze_final(
            baseline,
            control,
            v2,
            expected_datasets=DATASETS,
            expected_per_dataset=2,
            bootstrap_samples=200,
        )

        self.assertTrue(report["effectiveness"]["directional_success"])
        self.assertTrue(report["effectiveness"]["predeclared_success"])
        self.assertGreater(report["baseline_vs_v2"]["overall"]["em_delta"], 0)
        self.assertGreater(report["em_control_vs_v2"]["overall"]["em_delta"], 0)

    def test_extra_training_alone_is_not_attributed_to_v2_reward(self):
        from scripts.improvement_v2.final_analysis import analyze_final

        baseline = [
            _record("nq", 0, 1),
            _record("nq", 1, 0),
            _record("hotpotqa", 0, 0),
            _record("hotpotqa", 1, 0),
        ]
        control = [
            _record("nq", 0, 1),
            _record("nq", 1, 1),
            _record("hotpotqa", 0, 1),
            _record("hotpotqa", 1, 0),
        ]
        v2 = [dict(row) for row in control]

        report = analyze_final(
            baseline,
            control,
            v2,
            expected_datasets=DATASETS,
            expected_per_dataset=2,
            bootstrap_samples=200,
        )

        self.assertFalse(report["effectiveness"]["directional_success"])
        self.assertFalse(report["effectiveness"]["predeclared_success"])
        self.assertEqual(report["em_control_vs_v2"]["overall"]["em_delta"], 0.0)

    def test_pilot_selected_grouped_em_is_judged_against_frozen_baseline(self):
        from scripts.improvement_v2.final_analysis import analyze_final

        baseline = [
            _record("nq", 0, 1),
            _record("nq", 1, 0),
            _record("hotpotqa", 0, 0),
            _record("hotpotqa", 1, 0),
        ]
        control = [
            _record("nq", 0, 1),
            _record("nq", 1, 1),
            _record("hotpotqa", 0, 1),
            _record("hotpotqa", 1, 0),
        ]
        eff = [dict(row) for row in control]

        report = analyze_final(
            baseline,
            control,
            eff,
            selected_candidate="grouped_em",
            expected_datasets=DATASETS,
            expected_per_dataset=2,
            bootstrap_samples=200,
        )

        self.assertEqual(report["effectiveness"]["selected_candidate"], "grouped_em")
        self.assertTrue(report["effectiveness"]["predeclared_success"])
        self.assertGreater(
            report["baseline_vs_em_control"]["overall"]["em_delta"], 0.0
        )

    def test_final_success_requires_two_points_not_the_pilot_threshold(self):
        from scripts.improvement_v2.final_analysis import analyze_final

        baseline = [_record("nq", index, 0) for index in range(100)]
        control = [dict(row) for row in baseline]
        control[0]["em"] = control[0]["f1"] = 1.0
        eff = [dict(row) for row in control]

        report = analyze_final(
            baseline,
            control,
            eff,
            selected_candidate="grouped_em",
            expected_datasets=("nq",),
            expected_per_dataset=100,
            bootstrap_samples=100,
        )

        self.assertEqual(
            report["baseline_vs_em_control"]["overall"]["em_delta"], 0.01
        )
        self.assertFalse(report["effectiveness"]["predeclared_success"])
        self.assertEqual(report["effectiveness"]["minimum_final_em_gain"], 0.02)


if __name__ == "__main__":
    unittest.main()
