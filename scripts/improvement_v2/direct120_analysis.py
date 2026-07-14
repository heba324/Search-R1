#!/usr/bin/env python3
"""Analyze the time-constrained equal-update baseline versus EFF120 experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from scripts.improvement.analyze_paired_results import analyze_pairs
from scripts.improvement_v2.direct120_contract import DIRECT120
from scripts.paper_v1.contract import REQUIRED_EVALUATION_DATASETS


MINIMUM_EM_GAIN = DIRECT120.minimum_em_gain


def analyze_direct120(
    baseline,
    improved,
    expected_datasets=REQUIRED_EVALUATION_DATASETS,
    expected_per_dataset=100,
    bootstrap_samples=10000,
    seed=42,
):
    comparison = analyze_pairs(
        baseline,
        improved,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        expected_datasets=expected_datasets,
        expected_per_dataset=expected_per_dataset,
    )
    comparison["scope"] = (
        "equal-update original Search-R1 baseline versus group-corrected EFF120"
    )
    overall = comparison["overall"]
    criteria = DIRECT120.success
    try:
        single_hop_delta = comparison["groups"]["single_hop"]["em_delta"]
        response_growth = overall["response_tokens_delta"] / max(
            overall["baseline_response_tokens"], 1.0
        )
        required = {
            "evidence": overall["evidence_coverage_delta"],
            "valid_searches": overall["valid_searches_delta"],
            "searches": overall["searches_delta"],
            "duplicates": overall["duplicate_searches_delta"],
        }
    except KeyError as error:
        raise ValueError(
            f"Direct120 evaluation is missing required metric: {error}"
        ) from error

    primary_metric_pass = overall["em_delta"] >= MINIMUM_EM_GAIN
    guardrails_pass = (
        overall["f1_delta"] >= criteria.minimum_f1_delta
        and single_hop_delta >= criteria.minimum_single_hop_em_delta
        and required["evidence"] >= criteria.minimum_evidence_coverage_delta
        and required["valid_searches"] >= criteria.minimum_valid_searches_delta
        and required["searches"] <= criteria.maximum_searches_delta
        and required["duplicates"] <= criteria.maximum_duplicate_searches_delta
        and response_growth <= criteria.maximum_response_length_growth
    )
    predeclared_success = primary_metric_pass and guardrails_pass
    statistically_supported = (
        overall["em_delta_bootstrap_95_ci"][0] > 0.0
        and overall["mcnemar_exact_p"] < 0.05
    )
    if predeclared_success and statistically_supported:
        claim_level = "statistically_supported_direct120_improvement"
    elif predeclared_success:
        claim_level = "directional_direct120_improvement"
    elif not primary_metric_pass:
        claim_level = "not_effective_on_primary_metric"
    else:
        claim_level = "primary_gain_with_guardrail_failure"
    return {
        "protocol_id": DIRECT120.protocol_id,
        "scope": (
            "time-constrained single-new-arm experiment; baseline and EFF both "
            "start from Qwen2.5-1.5B and receive 120 updates"
        ),
        "causal_limit": (
            "The contrast estimates the combined grouping-plus-EFF method under "
            "a reused historical baseline. It does not isolate F1 fallback from "
            "group correction and is not a newly randomized two-arm trial."
        ),
        "comparison": comparison,
        "effectiveness": {
            "primary_metric": "exact_match",
            "minimum_em_gain": MINIMUM_EM_GAIN,
            "success_criteria": asdict(criteria),
            "primary_metric_pass": primary_metric_pass,
            "guardrails_pass": guardrails_pass,
            "predeclared_success": predeclared_success,
            "statistically_supported": statistically_supported,
            "response_length_relative_delta": response_growth,
            "claim_level": claim_level,
        },
    }


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-per-dataset", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze_direct120(
        _read_jsonl(args.baseline),
        _read_jsonl(args.improved),
        expected_per_dataset=args.expected_per_dataset,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Direct120 analysis: {args.output}")


if __name__ == "__main__":
    main()
