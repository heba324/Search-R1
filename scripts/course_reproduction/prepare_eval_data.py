#!/usr/bin/env python3
"""Create a deterministic balanced subset of all seven paper datasets."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.paper_v1.prepare_eval_data import DATASET_REPO, DATASET_REVISION, DATA_SOURCES, build_record


def select_indices(total: int, count: int, seed: int):
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return sorted(indices[: min(total, count)])


def main() -> None:
    from datasets import Dataset, load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/course_eval/test.parquet"))
    args = parser.parse_args()
    records = []
    counts = {}
    for offset, data_source in enumerate(DATA_SOURCES):
        dataset = load_dataset(DATASET_REPO, data_source, revision=DATASET_REVISION)
        split_name = "test" if "test" in dataset else "dev" if "dev" in dataset else "train"
        split = dataset[split_name]
        selected = select_indices(len(split), args.examples_per_dataset, args.seed + offset)
        counts[data_source] = len(selected)
        for index in selected:
            example = split[index]
            records.append(build_record(data_source, example["question"], example["golden_answers"], index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(args.output)
    print(f"Wrote {len(records)} deterministic evaluation examples to {args.output}")
    print("Per-dataset counts:", counts)


if __name__ == "__main__":
    main()
