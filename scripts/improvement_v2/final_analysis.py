#!/usr/bin/env python3
"""Analyze CEGR V2 against both frozen baseline and equal-budget EM control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.improvement.analyze_paired_results import analyze_pairs
from scripts.paper_v1.contract import REQUIRED_EVALUATION_DATASETS


MINIMUM_FINAL_EM_GAIN = 0.02


def _statistically_supported(comparison):
    overall = comparison["overall"]
    return (
        overall["em_delta_bootstrap_95_ci"][0] > 0.0
        and overall["mcnemar_exact_p"] < 0.05
    )


def analyze_final(
    baseline,
    control,
    improved,
    expected_datasets=REQUIRED_EVALUATION_DATASETS,
    expected_per_dataset=100,
    bootstrap_samples=10000,
    seed=42,
    selected_candidate="eff",
):
    if selected_candidate not in {"grouped_em", "eff"}:
        raise ValueError("selected_candidate must be grouped_em or eff")
    baseline_control_comparison = analyze_pairs(
        baseline,
        control,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        expected_datasets=expected_datasets,
        expected_per_dataset=expected_per_dataset,
    )
    baseline_comparison = analyze_pairs(
        baseline,
        improved,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        expected_datasets=expected_datasets,
        expected_per_dataset=expected_per_dataset,
    )
    control_comparison = analyze_pairs(
        control,
        improved,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        expected_datasets=expected_datasets,
        expected_per_dataset=expected_per_dataset,
    )
    baseline_control_comparison["scope"] = (
        "frozen Search-R1 baseline versus Grouped-EM refinement"
    )
    baseline_comparison["scope"] = "frozen Search-R1 baseline versus CEGR V2"
    control_comparison["scope"] = "equal-budget EM continuation versus CEGR V2"

    grouped_overall = baseline_control_comparison["overall"]
    baseline_delta = baseline_comparison["overall"]["em_delta"]
    control_delta = control_comparison["overall"]["em_delta"]
    control_overall = control_comparison["overall"]
    try:
        grouped_single_hop_delta = baseline_control_comparison["groups"][
            "single_hop"
        ]["em_delta"]
        eff_baseline_single_hop_delta = baseline_comparison["groups"][
            "single_hop"
        ]["em_delta"]
        eff_single_hop_delta = control_comparison["groups"]["single_hop"][
            "em_delta"
        ]
        grouped_response_growth = grouped_overall["response_tokens_delta"] / max(
            grouped_overall["baseline_response_tokens"], 1.0
        )
        eff_response_growth = control_overall["response_tokens_delta"] / max(
            control_overall["baseline_response_tokens"], 1.0
        )
        grouped_evidence_delta = grouped_overall["evidence_coverage_delta"]
        grouped_valid_search_delta = grouped_overall["valid_searches_delta"]
        grouped_search_delta = grouped_overall["searches_delta"]
        grouped_duplicate_delta = grouped_overall["duplicate_searches_delta"]
        eff_evidence_delta = baseline_comparison["overall"][
            "evidence_coverage_delta"
        ]
        eff_valid_search_delta = baseline_comparison["overall"][
            "valid_searches_delta"
        ]
        eff_search_delta = control_overall["searches_delta"]
        eff_duplicate_delta = control_overall["duplicate_searches_delta"]
    except KeyError as error:
        raise ValueError(f"Final evaluation is missing required metric: {error}") from error
    grouped_predeclared_success = (
        grouped_overall["em_delta"] >= MINIMUM_FINAL_EM_GAIN
        and grouped_overall["f1_delta"] >= 0.0
        and grouped_single_hop_delta >= 0.0
        and grouped_evidence_delta >= -0.02
        and grouped_valid_search_delta >= -0.15
        and grouped_search_delta <= 0.20
        and grouped_duplicate_delta <= 0.02
        and grouped_response_growth <= 0.15
    )
    eff_predeclared_success = (
        baseline_delta >= MINIMUM_FINAL_EM_GAIN
        and control_delta >= MINIMUM_FINAL_EM_GAIN
        and baseline_comparison["overall"]["f1_delta"] >= 0.0
        and control_overall["f1_delta"] >= 0.0
        and eff_baseline_single_hop_delta >= 0.0
        and eff_single_hop_delta >= 0.0
        and eff_evidence_delta >= -0.02
        and eff_valid_search_delta >= -0.15
        and eff_search_delta <= 0.20
        and eff_duplicate_delta <= 0.02
        and eff_response_growth <= 0.15
    )
    supported_grouped = _statistically_supported(baseline_control_comparison)
    supported_baseline = _statistically_supported(baseline_comparison)
    supported_control = _statistically_supported(control_comparison)
    if selected_candidate == "grouped_em":
        directional_success = grouped_overall["em_delta"] > 0.0
        predeclared_success = grouped_predeclared_success
        statistically_supported = supported_grouped
        selected_response_growth = grouped_response_growth
    else:
        directional_success = baseline_delta > 0.0 and control_delta > 0.0
        predeclared_success = eff_predeclared_success
        statistically_supported = supported_baseline and supported_control
        selected_response_growth = eff_response_growth
    if predeclared_success and statistically_supported:
        claim_level = f"statistically_supported_{selected_candidate}_improvement"
    elif predeclared_success:
        claim_level = f"directional_{selected_candidate}_improvement"
    else:
        claim_level = "not_effective_on_primary_metric"

    return {
        "scope": "fixed final subset; primary claim is locked by the disjoint pilot selection",
        "baseline_vs_em_control": baseline_control_comparison,
        "baseline_vs_v2": baseline_comparison,
        "em_control_vs_v2": control_comparison,
        "candidate_effectiveness": {
            "grouped_em": {
                "predeclared_success": grouped_predeclared_success,
                "statistically_supported": supported_grouped,
                "em_delta_vs_baseline": grouped_overall["em_delta"],
            },
            "eff": {
                "predeclared_success": eff_predeclared_success,
                "statistically_supported_vs_baseline": supported_baseline,
                "statistically_supported_vs_grouped_em": supported_control,
                "em_delta_vs_baseline": baseline_delta,
                "em_delta_vs_grouped_em": control_delta,
            },
        },
        "effectiveness": {
            "primary_metric": "exact_match",
            "selected_candidate": selected_candidate,
            "directional_success": directional_success,
            "predeclared_success": predeclared_success,
            "minimum_final_em_gain": MINIMUM_FINAL_EM_GAIN,
            "response_length_relative_delta_for_selected_comparison": selected_response_growth,
            "statistically_supported_for_selected_candidate": statistically_supported,
            "statistically_supported_vs_grouped_baseline": supported_grouped,
            "statistically_supported_vs_baseline": supported_baseline,
            "statistically_supported_vs_em_control": supported_control,
            "claim_level": claim_level,
        },
    }


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-per-dataset", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-gate", type=Path, required=True)
    args = parser.parse_args()
    pilot_gate = json.loads(args.pilot_gate.read_text(encoding="utf-8"))
    if not pilot_gate.get("passed") or pilot_gate.get("selected_candidate") not in {
        "grouped_em",
        "eff",
    }:
        raise SystemExit("A passing pilot with a locked candidate is required")
    result = analyze_final(
        _read_jsonl(args.baseline),
        _read_jsonl(args.control),
        _read_jsonl(args.improved),
        expected_per_dataset=args.expected_per_dataset,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        selected_candidate=pilot_gate["selected_candidate"],
    )
    result["pilot_selection"] = {
        "selected_candidate": pilot_gate["selected_candidate"],
        "pilot_gate": str(args.pilot_gate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
