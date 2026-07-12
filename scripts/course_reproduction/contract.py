"""Frozen defaults for the resource-limited course reproduction."""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class CourseReproductionContract:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "989aa79"
    algorithm: str = "grpo"
    retriever: str = "bm25"
    bm25_revision: str = "2c7554f"
    training_steps: int = 120
    train_batch_size: int = 32
    group_size: int = 5
    max_turns: int = 4
    topk: int = 3
    eval_examples_per_dataset: int = 100
    seed: int = 42


COURSE_REPRODUCTION = CourseReproductionContract()

REQUIRED_ASSETS = (
    "data/nq_hotpotqa_train/train.parquet",
    "data/nq_hotpotqa_train/test.parquet",
    "data/models/Qwen2.5-1.5B-Instruct/config.json",
    "data/course_eval/test.parquet",
)


def assess_assets(repo_root: Path) -> List[str]:
    errors = []
    for relative in REQUIRED_ASSETS:
        path = repo_root / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing course reproduction asset: {relative}")
    bm25 = repo_root / "data/wiki18_bm25/bm25"
    if not bm25.is_dir() or not any(bm25.iterdir()):
        errors.append("Missing course reproduction asset: data/wiki18_bm25/bm25")
    return errors
