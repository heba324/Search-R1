#!/usr/bin/env python3
"""Verify strict per-example EM against the official validation metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import fmean


def verify_records(marker, records, tolerance=1e-9):
    errors = []
    identifiers = [row["example_id"] for row in records]
    if len(set(identifiers)) != len(identifiers):
        errors.append("evaluation records contain duplicate example_id values")

    by_dataset = defaultdict(list)
    for row in records:
        by_dataset[str(row["dataset"])].append(float(row["em"]))
    official_metrics = marker.get("metrics", {})
    for dataset, official_em in official_metrics.items():
        if dataset not in by_dataset:
            errors.append(f"evaluation records are missing dataset {dataset}")
            continue
        record_em = fmean(by_dataset[dataset])
        if abs(record_em - float(official_em)) > tolerance:
            errors.append(
                f"record EM {record_em:.12f} for {dataset} does not match "
                f"official metric {float(official_em):.12f}"
            )
    unexpected = sorted(set(by_dataset) - set(official_metrics))
    if unexpected:
        errors.append(f"evaluation records contain unexpected datasets: {unexpected}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("marker", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    marker = json.loads(args.marker.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors = verify_records(marker, records)
    payload = {
        "records": len(records),
        "datasets": sorted({row["dataset"] for row in records}),
        "official_record_em_parity": not errors,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if errors:
        raise SystemExit("Evaluation record verification failed:\n- " + "\n- ".join(errors))


if __name__ == "__main__":
    main()
