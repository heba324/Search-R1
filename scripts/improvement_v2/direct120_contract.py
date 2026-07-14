"""Frozen contract for the urgent CEGR V2 2-step plus 120-step route."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Direct120Contract:
    protocol_id: str = "cegr-v2-direct120-urgent-v1"
    initial_model: str = "data/models/Qwen2.5-1.5B-Instruct"
    baseline_run: str = "search-r1-course-qwen2.5-1.5b-grpo-bm25"
    baseline_checkpoint_step: int = 120
    method: str = "em_first_f1_fallback"
    causal_estimand: str = "grouping_plus_eff"
    smoke_steps: int = 2
    training_steps: int = 120
    train_batch_size: int = 32
    group_size: int = 5
    learning_rate: float = 1e-6
    lr_warmup_steps_ratio: float = 0.95
    seed: int = 42
    rollout_engine_seed: int = 42
    final_examples_per_dataset: int = 100
    minimum_em_gain: float = 0.02


DIRECT120 = Direct120Contract()
