#!/usr/bin/env python3
"""Build the seven-dataset Search-R1 v1 evaluation parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

DATASET_REPO = "RUC-NLPIR/FlashRAG_datasets"
DATASET_REVISION = "ea6d672"
DATA_SOURCES = ("nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle")


def build_record(data_source: str, question: str, answers: Sequence[str], index: int) -> Dict[str, Any]:
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    prompt = (
        "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get "
        "new information. After reasoning, if you find you lack some knowledge, you can call a search engine by "
        "<search> query </search> and it will return the top searched results between <information> and "
        "</information>. You can search as many times as your want. If you find no further external knowledge "
        "needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
        f"For example, <answer> Beijing </answer>. Question: {question}\n"
    )
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": list(answers)}},
        "extra_info": {"split": "test", "index": index},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/paper_v1_eval/test.parquet"))
    return parser.parse_args()


def main() -> None:
    from datasets import Dataset, load_dataset

    args = parse_args()
    records = []
    for data_source in DATA_SOURCES:
        dataset = load_dataset(DATASET_REPO, data_source, revision=DATASET_REVISION)
        split_name = "test" if "test" in dataset else "dev" if "dev" in dataset else "train"
        for index, example in enumerate(dataset[split_name]):
            records.append(build_record(data_source, example["question"], example["golden_answers"], index))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(args.output)
    print(f"Wrote {len(records)} paper-v1 evaluation examples to {args.output}")


if __name__ == "__main__":
    main()
