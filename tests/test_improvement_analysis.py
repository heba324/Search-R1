import unittest


class ImprovementAnalysisTests(unittest.TestCase):
    def test_evaluation_record_keeps_identity_and_mechanism_metrics(self):
        from scripts.improvement.evaluation_record import build_evaluation_record

        record = build_evaluation_record(
            "<search>Britney Spears birthplace</search>"
            "<information>Spears was born in McComb, Mississippi.</information>"
            "<answer>McComb, Mississippi</answer>",
            dataset="hotpotqa",
            golden_answers=["McComb, Mississippi"],
            extra_info={"split": "test", "index": 17},
        )

        self.assertEqual(record["example_id"], "hotpotqa:test:17")
        self.assertEqual(record["em"], 1.0)
        self.assertEqual(record["f1"], 1.0)
        self.assertEqual(record["evidence_coverage"], 1.0)
        self.assertEqual(record["valid_searches"], 1)

    def test_paired_analysis_reports_effect_and_exact_mcnemar(self):
        from scripts.improvement.analyze_paired_results import analyze_pairs

        baseline = [
            {"example_id": "nq:0", "dataset": "nq", "em": 0.0, "f1": 0.2},
            {"example_id": "nq:1", "dataset": "nq", "em": 1.0, "f1": 1.0},
            {"example_id": "nq:2", "dataset": "nq", "em": 0.0, "f1": 0.0},
        ]
        improved = [
            {"example_id": "nq:0", "dataset": "nq", "em": 1.0, "f1": 1.0},
            {"example_id": "nq:1", "dataset": "nq", "em": 1.0, "f1": 1.0},
            {"example_id": "nq:2", "dataset": "nq", "em": 0.0, "f1": 0.5},
        ]

        result = analyze_pairs(baseline, improved, bootstrap_samples=200, seed=42)

        self.assertAlmostEqual(result["overall"]["em_delta"], 1.0 / 3.0)
        self.assertEqual(result["overall"]["baseline_wrong_improved_right"], 1)
        self.assertEqual(result["overall"]["baseline_right_improved_wrong"], 0)
        self.assertEqual(result["overall"]["mcnemar_exact_p"], 1.0)
        self.assertEqual(result["datasets"]["nq"]["count"], 3)

    def test_paired_analysis_rejects_mismatched_examples(self):
        from scripts.improvement.analyze_paired_results import analyze_pairs

        baseline = [{"example_id": "nq:0", "dataset": "nq", "em": 0, "f1": 0}]
        improved = [{"example_id": "nq:1", "dataset": "nq", "em": 1, "f1": 1}]

        with self.assertRaises(ValueError):
            analyze_pairs(baseline, improved, bootstrap_samples=10, seed=42)


if __name__ == "__main__":
    unittest.main()
