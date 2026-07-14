import json
import tempfile
import unittest
from pathlib import Path


DATASETS = ("nq", "hotpotqa")


def _record(dataset, index, em, evidence, searches=1, f1=None):
    return {
        "example_id": f"{dataset}:test:{index}",
        "dataset": dataset,
        "em": float(em),
        "f1": float(em if f1 is None else f1),
        "evidence_coverage": float(evidence),
        "searches": searches,
        "valid_searches": searches,
        "duplicate_searches": 0.0,
        "response_tokens": 100,
    }


class CEGRV2PilotGateTests(unittest.TestCase):
    def test_gate_requires_v2_signal_beyond_baseline_and_equal_budget_control(self):
        from scripts.improvement_v2.pilot_gate import assess_pilot

        baseline = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 0, 0),
            _record("hotpotqa", 0, 1, 1, searches=2),
            _record("hotpotqa", 1, 0, 0, searches=2),
        ]
        control = [dict(row) for row in baseline]
        improved = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 1, 1),
            _record("hotpotqa", 0, 1, 1, searches=2),
            _record("hotpotqa", 1, 0, 0, searches=2),
        ]

        report = assess_pilot(
            baseline,
            control,
            improved,
            expected_datasets=DATASETS,
            expected_per_dataset=2,
            single_hop_datasets=("nq",),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_candidate"], "eff")
        self.assertTrue(all(gate["passed"] for gate in report["gates"]))
        self.assertGreater(report["comparisons"]["v2_minus_baseline"]["em"], 0.0)

    def test_gate_rejects_accuracy_regression_even_when_evidence_improves(self):
        from scripts.improvement_v2.pilot_gate import assess_pilot

        baseline = [
            _record("nq", 0, 1, 0),
            _record("nq", 1, 1, 0),
            _record("hotpotqa", 0, 1, 0),
            _record("hotpotqa", 1, 0, 0),
        ]
        control = [dict(row) for row in baseline]
        regressed = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 0, 1),
            _record("hotpotqa", 0, 1, 1),
            _record("hotpotqa", 1, 0, 1),
        ]

        report = assess_pilot(
            baseline,
            control,
            regressed,
            expected_datasets=DATASETS,
            expected_per_dataset=2,
            single_hop_datasets=("nq",),
        )

        self.assertFalse(report["passed"])
        failed = {
            gate["name"]
            for gates in report["candidate_gates"].values()
            for gate in gates
            if not gate["passed"]
        }
        self.assertIn("eff_em_gain_over_frozen_baseline", failed)

    def test_gate_selects_grouped_em_when_eff_adds_no_further_gain(self):
        from scripts.improvement_v2.pilot_gate import assess_pilot

        baseline = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 0, 0),
            _record("hotpotqa", 0, 0, 0),
            _record("hotpotqa", 1, 0, 0),
        ]
        control = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 1, 1),
            _record("hotpotqa", 0, 0, 0),
            _record("hotpotqa", 1, 0, 0),
        ]
        same_as_control = [dict(row) for row in control]

        report = assess_pilot(
            baseline,
            control,
            same_as_control,
            expected_datasets=DATASETS,
            expected_per_dataset=2,
            single_hop_datasets=("nq",),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_candidate"], "grouped_em")
        failed = {
            gate["name"]
            for gate in report["candidate_gates"]["eff"]
            if not gate["passed"]
        }
        self.assertIn("eff_em_gain_over_grouped_em", failed)

    def test_gate_rejects_duplicate_or_incomplete_paired_records(self):
        from scripts.improvement_v2.pilot_gate import assess_pilot

        records = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 0, 0),
            _record("hotpotqa", 0, 1, 1),
            _record("hotpotqa", 1, 0, 0),
        ]
        duplicate = [dict(row) for row in records]
        duplicate[-1]["example_id"] = duplicate[-2]["example_id"]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            assess_pilot(
                records,
                records,
                duplicate,
                expected_datasets=DATASETS,
                expected_per_dataset=2,
                single_hop_datasets=("nq",),
            )

    def test_eff_cannot_inherit_single_hop_regression_from_grouped_em(self):
        from scripts.improvement_v2.pilot_gate import assess_pilot

        baseline = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 0, 0),
            _record("hotpotqa", 0, 0, 0),
            _record("hotpotqa", 1, 0, 0),
        ]
        control = [
            _record("nq", 0, 0, 1),
            _record("nq", 1, 0, 0),
            _record("hotpotqa", 0, 1, 1),
            _record("hotpotqa", 1, 0, 0),
        ]
        eff = [
            _record("nq", 0, 0, 1),
            _record("nq", 1, 0, 0),
            _record("hotpotqa", 0, 1, 1),
            _record("hotpotqa", 1, 1, 1),
        ]

        report = assess_pilot(
            baseline,
            control,
            eff,
            expected_datasets=DATASETS,
            expected_per_dataset=2,
            single_hop_datasets=("nq",),
        )

        self.assertFalse(report["passed"])
        failed = {
            gate["name"]
            for gate in report["candidate_gates"]["eff"]
            if not gate["passed"]
        }
        self.assertIn("eff_single_hop_no_net_loss_vs_frozen_baseline", failed)

    def test_locked_gate_rejects_changed_selection_or_input_records(self):
        from scripts.improvement_v2.pilot_gate import build_locked_pilot_report
        from scripts.improvement_v2.verify_pilot_gate import verify_locked_pilot_gate

        baseline = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 0, 0),
            _record("hotpotqa", 0, 0, 0),
            _record("hotpotqa", 1, 0, 0),
        ]
        control = [
            _record("nq", 0, 1, 1),
            _record("nq", 1, 1, 1),
            _record("hotpotqa", 0, 0, 0),
            _record("hotpotqa", 1, 0, 0),
        ]
        eff = [dict(row) for row in control]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "baseline": root / "baseline.jsonl",
                "em_control": root / "control.jsonl",
                "cegr_v2": root / "eff.jsonl",
            }
            for path, records in zip(paths.values(), (baseline, control, eff)):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in records),
                    encoding="utf-8",
                )
            gate_path = root / "pilot-gate.json"
            report = build_locked_pilot_report(
                paths["baseline"],
                paths["em_control"],
                paths["cegr_v2"],
                expected_datasets=DATASETS,
                expected_per_dataset=2,
                single_hop_datasets=("nq",),
            )
            gate_path.write_text(json.dumps(report), encoding="utf-8")

            self.assertEqual(
                verify_locked_pilot_gate(
                    gate_path,
                    paths["baseline"],
                    paths["em_control"],
                    paths["cegr_v2"],
                    expected_datasets=DATASETS,
                    expected_per_dataset=2,
                    single_hop_datasets=("nq",),
                ),
                [],
            )

            tampered = dict(report)
            tampered["selected_candidate"] = "eff"
            gate_path.write_text(json.dumps(tampered), encoding="utf-8")
            self.assertTrue(
                verify_locked_pilot_gate(
                    gate_path,
                    *paths.values(),
                    expected_datasets=DATASETS,
                    expected_per_dataset=2,
                    single_hop_datasets=("nq",),
                )
            )

            gate_path.write_text(json.dumps(report), encoding="utf-8")
            paths["baseline"].write_text(
                paths["baseline"].read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            self.assertTrue(
                verify_locked_pilot_gate(
                    gate_path,
                    *paths.values(),
                    expected_datasets=DATASETS,
                    expected_per_dataset=2,
                    single_hop_datasets=("nq",),
                )
            )


if __name__ == "__main__":
    unittest.main()
