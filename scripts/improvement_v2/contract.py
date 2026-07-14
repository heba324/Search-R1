"""Frozen contract for the CEGR V2 refinement experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CEGRV2Contract:
    v1_commit: str = "8672aad0f4089f0fca388601cd9ce20fc9b8b776"
    method: str = "em_first_f1_fallback"
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    initial_run: str = "search-r1-course-qwen2.5-1.5b-grpo-bm25"
    initial_checkpoint_step: int = 120
    refinement_steps: int = 40
    equal_budget_control: str = "grouped_em"
    learning_rate: float = 5e-7
    train_batch_size: int = 32
    group_size: int = 5
    max_turns: int = 4
    topk: int = 3
    seed: int = 42
    rollout_engine_seed: int = 42
    pilot_examples_per_dataset: int = 20
    final_examples_per_dataset: int = 100


CEGR_V2 = CEGRV2Contract()
