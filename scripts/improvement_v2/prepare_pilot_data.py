#!/usr/bin/env python3
"""Create a deterministic pilot set disjoint from the fixed 700-example test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from scripts.paper_v1.prepare_eval_data import (
    DATASET_REPO,
    DATASET_REVISION,
    DATA_SOURCES,
    build_record,
)


def select_disjoint_indices(total, excluded, count, seed):
    eligible = [index for index in range(total) if index not in set(excluded)]
    if len(eligible) < count:
        raise ValueError(
            f"Requested {count} unseen examples but only {len(eligible)} are available"
        )
    random.Random(seed).shuffle(eligible)
    return sorted(eligible[:count])


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    from datasets import Dataset, load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-eval", type=Path, default=Path("data/course_eval/test.parquet")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/improvement_v2/pilot.parquet")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/improvement_v2/pilot_manifest.json")
    )
    parser.add_argument("--examples-per-dataset", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    if not args.final_eval.is_file():
        raise SystemExit(f"Missing fixed final evaluation data: {args.final_eval}")

    final_rows = load_dataset(
        "parquet", data_files=str(args.final_eval), split="train"
    )
    excluded = {dataset: set() for dataset in DATA_SOURCES}
    for row in final_rows:
        dataset = str(row["data_source"])
        if dataset not in excluded:
            raise ValueError(f"Unexpected dataset in final evaluation: {dataset}")
        excluded[dataset].add(int(row["extra_info"]["index"]))

    records = []
    selected_manifest = {}
    for offset, dataset_name in enumerate(DATA_SOURCES):
        dataset = load_dataset(
            DATASET_REPO, dataset_name, revision=DATASET_REVISION
        )
        split_name = "test" if "test" in dataset else "dev" if "dev" in dataset else "train"
        split = dataset[split_name]
        selected = select_disjoint_indices(
            len(split),
            excluded[dataset_name],
            args.examples_per_dataset,
            args.seed + offset,
        )
        selected_manifest[dataset_name] = selected
        for index in selected:
            example = split[index]
            records.append(
                build_record(
                    dataset_name,
                    example["question"],
                    example["golden_answers"],
                    index,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(args.output)
    manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "seed": args.seed,
        "examples_per_dataset": args.examples_per_dataset,
        "total_examples": len(records),
        "pilot_bytes": args.output.stat().st_size,
        "pilot_sha256": _sha256(args.output),
        "excluded_final_eval_sha256": _sha256(args.final_eval),
        "selected_indices": selected_manifest,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} disjoint pilot examples to {args.output}")
    print(f"Pilot manifest: {args.manifest}")


if __name__ == "__main__":
    main()
