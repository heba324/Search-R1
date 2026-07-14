#!/usr/bin/env python3
"""Verify the immutable pilot selection and all trajectories bound to it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.improvement_v2.pilot_gate import (
    DEFAULT_SINGLE_HOP_DATASETS,
    build_locked_pilot_report,
)
from scripts.paper_v1.contract import REQUIRED_EVALUATION_DATASETS


def verify_locked_pilot_gate(
    gate_path,
    baseline_path,
    control_path,
    improved_path,
    expected_datasets=REQUIRED_EVALUATION_DATASETS,
    expected_per_dataset=20,
    single_hop_datasets=DEFAULT_SINGLE_HOP_DATASETS,
):
    try:
        stored = json.loads(Path(gate_path).read_text(encoding="utf-8"))
        recomputed = build_locked_pilot_report(
            baseline_path,
            control_path,
            improved_path,
            expected_datasets=expected_datasets,
            expected_per_dataset=expected_per_dataset,
            single_hop_datasets=single_hop_datasets,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return [f"Could not verify locked pilot gate: {error}"]
    errors = []
    if stored != recomputed:
        errors.append("pilot gate does not match its locked trajectories and selection")
    if not recomputed.get("passed"):
        errors.append("locked pilot did not pass")
    if recomputed.get("selected_candidate") not in {"grouped_em", "eff"}:
        errors.append("locked pilot has no valid selected candidate")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("--expected-per-dataset", type=int, default=20)
    args = parser.parse_args()
    errors = verify_locked_pilot_gate(
        args.gate,
        args.baseline,
        args.control,
        args.improved,
        expected_per_dataset=args.expected_per_dataset,
    )
    if errors:
        raise SystemExit("Pilot lock verification failed:\n- " + "\n- ".join(errors))
    print(f"Verified locked pilot selection: {args.gate}")


if __name__ == "__main__":
    main()
