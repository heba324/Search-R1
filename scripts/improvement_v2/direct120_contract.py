"""Frozen contract for the urgent CEGR V2 2-step plus 120-step route."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import shlex


@dataclass(frozen=True)
class Direct120SuccessCriteria:
    minimum_f1_delta: float = 0.0
    minimum_single_hop_em_delta: float = 0.0
    minimum_evidence_coverage_delta: float = -0.02
    minimum_valid_searches_delta: float = -0.15
    maximum_searches_delta: float = 0.20
    maximum_duplicate_searches_delta: float = 0.02
    maximum_response_length_growth: float = 0.15


@dataclass(frozen=True)
class Direct120Contract:
    protocol_id: str = "cegr-v2-direct120-urgent-v1"
    initial_model: str = "data/models/Qwen2.5-1.5B-Instruct"
    baseline_run: str = "search-r1-course-qwen2.5-1.5b-grpo-bm25"
    smoke_run: str = "search-r1-cegr-v2-eff-direct120-smoke"
    training_run: str = (
        "search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25"
    )
    baseline_checkpoint_step: int = 120
    method: str = "em_first_f1_fallback"
    reward_mode: str = "eff"
    causal_estimand: str = "grouping_plus_eff"
    smoke_steps: int = 2
    training_steps: int = 120
    smoke_train_batch_size: int = 8
    train_batch_size: int = 32
    group_size: int = 5
    learning_rate: float = 1e-6
    lr_warmup_steps_ratio: float = 0.95
    save_freq: int = 40
    seed: int = 42
    rollout_engine_seed: int = 42
    minimum_smoke_signal: float = 0.10
    eval_batch_size: int = 28
    final_examples_per_dataset: int = 100
    bootstrap_samples: int = 10000
    minimum_em_gain: float = 0.02
    success: Direct120SuccessCriteria = field(
        default_factory=Direct120SuccessCriteria
    )


DIRECT120 = Direct120Contract()


def shell_values(contract=DIRECT120):
    """Return the protocol values consumed by the shell entrypoints."""
    return {
        "DIRECT120_INITIAL_MODEL": contract.initial_model,
        "DIRECT120_BASELINE_RUN": contract.baseline_run,
        "DIRECT120_SMOKE_RUN": contract.smoke_run,
        "DIRECT120_TRAINING_RUN": contract.training_run,
        "DIRECT120_BASELINE_STEP": contract.baseline_checkpoint_step,
        "DIRECT120_REWARD_MODE": contract.reward_mode,
        "DIRECT120_SMOKE_STEPS": contract.smoke_steps,
        "DIRECT120_TRAINING_STEPS": contract.training_steps,
        "DIRECT120_SMOKE_BATCH_SIZE": contract.smoke_train_batch_size,
        "DIRECT120_TRAIN_BATCH_SIZE": contract.train_batch_size,
        "DIRECT120_GROUP_SIZE": contract.group_size,
        "DIRECT120_LEARNING_RATE": contract.learning_rate,
        "DIRECT120_LR_WARMUP_RATIO": contract.lr_warmup_steps_ratio,
        "DIRECT120_SAVE_FREQ": contract.save_freq,
        "DIRECT120_SEED": contract.seed,
        "DIRECT120_ROLLOUT_ENGINE_SEED": contract.rollout_engine_seed,
        "DIRECT120_MINIMUM_SMOKE_SIGNAL": contract.minimum_smoke_signal,
        "DIRECT120_EVAL_BATCH_SIZE": contract.eval_batch_size,
        "DIRECT120_FINAL_PER_DATASET": contract.final_examples_per_dataset,
        "DIRECT120_BOOTSTRAP_SAMPLES": contract.bootstrap_samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--shell", action="store_true")
    output.add_argument("--json", action="store_true")
    args = parser.parse_args()
    values = shell_values()
    if args.json:
        print(json.dumps(values, indent=2, sort_keys=True))
        return
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
