"""Frozen contract for the CEGR comparison experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CEGRExperimentContract:
    name: str = "CEGR"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    algorithm: str = "grpo"
    retriever: str = "bm25"
    training_steps: int = 120
    train_batch_size: int = 32
    group_size: int = 5
    max_turns: int = 4
    topk: int = 3
    eval_examples_per_dataset: int = 100
    seed: int = 42
    changed_variable: str = "reward"


CEGR_EXPERIMENT = CEGRExperimentContract()
