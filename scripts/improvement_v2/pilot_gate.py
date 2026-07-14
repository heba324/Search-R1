#!/usr/bin/env python3
"""Apply predeclared paired gates before a CEGR V2 full evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import fmean

from scripts.paper_v1.contract import REQUIRED_EVALUATION_DATASETS


DEFAULT_SINGLE_HOP_DATASETS = ("nq", "triviaqa", "popqa")


def _index_records(records, label, expected_datasets, expected_per_dataset):
    expected_total = len(expected_datasets) * expected_per_dataset
    if len(records) != expected_total:
        raise ValueError(
            f"Expected {expected_total} {label} records, found {len(records)}"
        )
    identifiers = [row["example_id"] for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{label} evaluation contains duplicate example_id values")
    counts = Counter(row["dataset"] for row in records)
    expected_counts = {dataset: expected_per_dataset for dataset in expected_datasets}
    if dict(counts) != expected_counts:
        raise ValueError(
            f"Expected {expected_per_dataset} rows per dataset in {label}; "
            f"found {dict(sorted(counts.items()))}"
        )
    return {row["example_id"]: row for row in records}


def _mean(index, key, datasets=None):
    rows = index.values()
    if datasets is not None:
        rows = [row for row in rows if row["dataset"] in datasets]
    values = [float(row[key]) for row in rows]
    if not values:
        raise ValueError(f"No records available for metric {key}")
    return fmean(values)


def _flips(before, after, datasets=None):
    identifiers = before.keys()
    if datasets is not None:
        identifiers = [
            identifier
            for identifier in identifiers
            if before[identifier]["dataset"] in datasets
        ]
    gains = sum(
        float(before[identifier]["em"]) == 0.0
        and float(after[identifier]["em"]) == 1.0
        for identifier in identifiers
    )
    losses = sum(
        float(before[identifier]["em"]) == 1.0
        and float(after[identifier]["em"]) == 0.0
        for identifier in identifiers
    )
    return {"gains": gains, "losses": losses, "net": gains - losses}


def _comparison(before, after):
    metrics = (
        "em",
        "f1",
        "evidence_coverage",
        "searches",
        "valid_searches",
        "duplicate_searches",
        "response_tokens",
    )
    result = {
        metric: _mean(after, metric) - _mean(before, metric) for metric in metrics
    }
    result["paired_flips"] = _flips(before, after)
    return result


def _gate(name, passed, observed, threshold):
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
    }


def assess_pilot(
    baseline,
    control,
    improved,
    expected_datasets=REQUIRED_EVALUATION_DATASETS,
    expected_per_dataset=20,
    single_hop_datasets=DEFAULT_SINGLE_HOP_DATASETS,
):
    """Return a paired pilot report whose gates fail closed."""
    baseline_by_id = _index_records(
        baseline, "baseline", expected_datasets, expected_per_dataset
    )
    control_by_id = _index_records(
        control, "EM-control", expected_datasets, expected_per_dataset
    )
    improved_by_id = _index_records(
        improved, "CEGR-V2", expected_datasets, expected_per_dataset
    )
    identifier_sets = [
        set(baseline_by_id),
        set(control_by_id),
        set(improved_by_id),
    ]
    if identifier_sets[0] != identifier_sets[1] or identifier_sets[0] != identifier_sets[2]:
        raise ValueError("Baseline, EM-control, and CEGR-V2 examples do not match")

    control_versus_baseline = _comparison(baseline_by_id, control_by_id)
    versus_baseline = _comparison(baseline_by_id, improved_by_id)
    versus_control = _comparison(control_by_id, improved_by_id)
    control_single_hop_flips = _flips(
        baseline_by_id, control_by_id, datasets=set(single_hop_datasets)
    )
    eff_baseline_single_hop_flips = _flips(
        baseline_by_id, improved_by_id, datasets=set(single_hop_datasets)
    )
    eff_single_hop_flips = _flips(
        control_by_id, improved_by_id, datasets=set(single_hop_datasets)
    )
    baseline_response_tokens = _mean(baseline_by_id, "response_tokens")
    control_response_tokens = _mean(control_by_id, "response_tokens")
    if baseline_response_tokens <= 0.0 or control_response_tokens <= 0.0:
        raise ValueError("Pilot response token means must be positive")
    control_response_length_relative_delta = (
        control_versus_baseline["response_tokens"] / baseline_response_tokens
    )
    eff_response_length_relative_delta = (
        versus_control["response_tokens"] / control_response_tokens
    )
    grouped_em_gates = [
        _gate(
            "grouped_em_gain_over_frozen_baseline",
            control_versus_baseline["em"] >= 0.01,
            control_versus_baseline["em"],
            ">= +0.01",
        ),
        _gate(
            "grouped_em_f1_not_below_frozen_baseline",
            control_versus_baseline["f1"] >= 0.0,
            control_versus_baseline["f1"],
            ">= 0.0",
        ),
        _gate(
            "grouped_em_single_hop_no_net_loss",
            control_single_hop_flips["net"] >= 0,
            control_single_hop_flips,
            "paired net >= 0 examples",
        ),
        _gate(
            "grouped_em_evidence_coverage_drop_at_most_two_points",
            control_versus_baseline["evidence_coverage"] >= -0.02,
            control_versus_baseline["evidence_coverage"],
            ">= -0.02",
        ),
        _gate(
            "grouped_em_valid_search_drop_at_most_point_fifteen",
            control_versus_baseline["valid_searches"] >= -0.15,
            control_versus_baseline["valid_searches"],
            ">= -0.15 searches/example",
        ),
        _gate(
            "grouped_em_search_increase_at_most_point_two",
            control_versus_baseline["searches"] <= 0.20,
            control_versus_baseline["searches"],
            "<= +0.20 searches/example",
        ),
        _gate(
            "grouped_em_duplicate_search_increase_at_most_point_zero_two",
            control_versus_baseline["duplicate_searches"] <= 0.02,
            control_versus_baseline["duplicate_searches"],
            "<= +0.02 duplicates/example",
        ),
        _gate(
            "grouped_em_response_length_increase_at_most_fifteen_percent",
            control_response_length_relative_delta <= 0.15,
            control_response_length_relative_delta,
            "<= +0.15 relative",
        ),
    ]
    eff_gates = [
        _gate(
            "eff_em_gain_over_frozen_baseline",
            versus_baseline["em"] >= 0.01,
            versus_baseline["em"],
            ">= +0.01",
        ),
        _gate(
            "eff_em_gain_over_grouped_em",
            versus_control["em"] >= 0.01,
            versus_control["em"],
            ">= +0.01",
        ),
        _gate(
            "eff_f1_not_below_frozen_baseline",
            versus_baseline["f1"] >= 0.0,
            versus_baseline["f1"],
            ">= 0.0",
        ),
        _gate(
            "eff_f1_not_below_grouped_em",
            versus_control["f1"] >= 0.0,
            versus_control["f1"],
            ">= 0.0",
        ),
        _gate(
            "eff_single_hop_no_net_loss_vs_frozen_baseline",
            eff_baseline_single_hop_flips["net"] >= 0,
            eff_baseline_single_hop_flips,
            "paired net >= 0 examples",
        ),
        _gate(
            "eff_single_hop_no_net_loss_vs_grouped_em",
            eff_single_hop_flips["net"] >= 0,
            eff_single_hop_flips,
            "paired net >= 0 examples",
        ),
        _gate(
            "eff_evidence_coverage_drop_at_most_two_points",
            versus_baseline["evidence_coverage"] >= -0.02,
            versus_baseline["evidence_coverage"],
            ">= -0.02",
        ),
        _gate(
            "eff_valid_search_drop_at_most_point_fifteen",
            versus_baseline["valid_searches"] >= -0.15,
            versus_baseline["valid_searches"],
            ">= -0.15 searches/example",
        ),
        _gate(
            "eff_search_increase_vs_grouped_em_at_most_point_two",
            versus_control["searches"] <= 0.20,
            versus_control["searches"],
            "<= +0.20 searches/example",
        ),
        _gate(
            "eff_duplicate_search_increase_vs_grouped_em_at_most_point_zero_two",
            versus_control["duplicate_searches"] <= 0.02,
            versus_control["duplicate_searches"],
            "<= +0.02 duplicates/example",
        ),
        _gate(
            "eff_response_length_increase_vs_grouped_em_at_most_fifteen_percent",
            eff_response_length_relative_delta <= 0.15,
            eff_response_length_relative_delta,
            "<= +0.15 relative",
        ),
    ]
    candidate_gates = {"grouped_em": grouped_em_gates, "eff": eff_gates}
    candidate_passed = {
        name: all(gate["passed"] for gate in gates)
        for name, gates in candidate_gates.items()
    }
    selected_candidate = (
        "eff"
        if candidate_passed["eff"]
        else "grouped_em"
        if candidate_passed["grouped_em"]
        else None
    )
    displayed_gates = (
        candidate_gates[selected_candidate]
        if selected_candidate is not None
        else eff_gates + grouped_em_gates
    )
    return {
        "scope": "disjoint paired CEGR V2 candidate selection; screening evidence, not final proof",
        "expected_datasets": list(expected_datasets),
        "expected_per_dataset": expected_per_dataset,
        "comparisons": {
            "grouped_em_minus_baseline": control_versus_baseline,
            "v2_minus_baseline": versus_baseline,
            "v2_minus_em_control": versus_control,
        },
        "single_hop_flips": {
            "grouped_em_vs_baseline": control_single_hop_flips,
            "eff_vs_baseline": eff_baseline_single_hop_flips,
            "eff_vs_grouped_em": eff_single_hop_flips,
        },
        "candidate_gates": candidate_gates,
        "candidate_passed": candidate_passed,
        "selection_rule": "prefer eff only when it beats both comparators; otherwise select passing grouped_em",
        "selected_candidate": selected_candidate,
        "gates": displayed_gates,
        "passed": selected_candidate is not None,
    }


def _load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_locked_pilot_report(
    baseline_path,
    control_path,
    improved_path,
    expected_datasets=REQUIRED_EVALUATION_DATASETS,
    expected_per_dataset=20,
    single_hop_datasets=DEFAULT_SINGLE_HOP_DATASETS,
):
    paths = {
        "baseline": Path(baseline_path),
        "em_control": Path(control_path),
        "cegr_v2": Path(improved_path),
    }
    report = assess_pilot(
        _load_jsonl(paths["baseline"]),
        _load_jsonl(paths["em_control"]),
        _load_jsonl(paths["cegr_v2"]),
        expected_datasets=expected_datasets,
        expected_per_dataset=expected_per_dataset,
        single_hop_datasets=single_hop_datasets,
    )
    report["pilot_lock"] = {
        "schema_version": 1,
        "input_sha256": {
            label: _sha256(path) for label, path in paths.items()
        },
        "selected_candidate": report["selected_candidate"],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-per-dataset", type=int, default=20)
    args = parser.parse_args()
    report = build_locked_pilot_report(
        args.baseline,
        args.control,
        args.improved,
        expected_per_dataset=args.expected_per_dataset,
    )
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != report:
            raise SystemExit(
                "Locked pilot gate or its input trajectories changed; refusing to overwrite"
            )
        print(f"Verified existing locked pilot gate: {args.output}")
        if not report["passed"]:
            raise SystemExit("CEGR V2 pilot gates failed; do not start a longer run")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    if not report["passed"]:
        raise SystemExit("CEGR V2 pilot gates failed; do not start a longer run")


if __name__ == "__main__":
    main()
