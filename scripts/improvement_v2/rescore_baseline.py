#!/usr/bin/env python3
"""Rescore frozen V1 trajectories with the strict V2 parser without regenerating."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import fmean

from scripts.improvement_v2.evaluation_record import build_evaluation_record
from scripts.paper_v1.contract import REQUIRED_EVALUATION_DATASETS


def rescore_records(records):
    rescored = []
    parser_mismatches = 0
    for row in records:
        record = build_evaluation_record(
            row["trajectory"],
            dataset=str(row["dataset"]),
            golden_answers=row["golden_answers"],
            extra_info={"split": row["split"], "index": row["index"]},
        )
        if record["example_id"] != row["example_id"]:
            raise ValueError(f"Identity changed while rescoring {row['example_id']}")
        if "response_tokens" in row:
            record["response_tokens"] = int(row["response_tokens"])
        parser_mismatches += float(record["em"]) != float(row["em"])
        rescored.append(record)
    report = {
        "records": len(rescored),
        "original_em": fmean(float(row["em"]) for row in records),
        "strict_em": fmean(float(row["em"]) for row in rescored),
        "parser_mismatch_count": parser_mismatches,
        "regenerated": False,
    }
    return rescored, report


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-per-dataset", type=int, default=100)
    args = parser.parse_args()
    records = _read_jsonl(args.input)
    expected_total = len(REQUIRED_EVALUATION_DATASETS) * args.expected_per_dataset
    if len(records) != expected_total:
        raise SystemExit(f"Expected {expected_total} frozen baseline rows, found {len(records)}")
    identifiers = [row["example_id"] for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("Frozen baseline contains duplicate example_id values")
    counts = Counter(row["dataset"] for row in records)
    expected_counts = {
        dataset: args.expected_per_dataset for dataset in REQUIRED_EVALUATION_DATASETS
    }
    if dict(counts) != expected_counts:
        raise SystemExit(f"Frozen baseline dataset counts are invalid: {dict(counts)}")

    rescored, report = rescore_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rescored:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Strictly rescored {len(rescored)} frozen baseline trajectories")


if __name__ == "__main__":
    main()
