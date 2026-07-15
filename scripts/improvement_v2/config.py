"""Frozen configuration for the completed CEGR V2 experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import shlex


@dataclass(frozen=True)
class SuccessCriteria:
    minimum_f1_delta: float = 0.0
    minimum_single_hop_em_delta: float = 0.0
    minimum_evidence_coverage_delta: float = -0.02
    minimum_valid_searches_delta: float = -0.15
    maximum_searches_delta: float = 0.20
    maximum_duplicate_searches_delta: float = 0.02
    maximum_response_length_growth: float = 0.15


@dataclass(frozen=True)
class ExperimentConfig:
    # Artifact identifiers are immutable because the completed run uses them.
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
    success: SuccessCriteria = field(default_factory=SuccessCriteria)


EXPERIMENT = ExperimentConfig()


def shell_values(config=EXPERIMENT):
    """Return the protocol values consumed by the shell entrypoints."""
    return {
        "CEGR_V2_INITIAL_MODEL": config.initial_model,
        "CEGR_V2_BASELINE_RUN": config.baseline_run,
        "CEGR_V2_SMOKE_RUN": config.smoke_run,
        "CEGR_V2_TRAINING_RUN": config.training_run,
        "CEGR_V2_BASELINE_STEP": config.baseline_checkpoint_step,
        "CEGR_V2_REWARD_MODE": config.reward_mode,
        "CEGR_V2_SMOKE_STEPS": config.smoke_steps,
        "CEGR_V2_TRAINING_STEPS": config.training_steps,
        "CEGR_V2_SMOKE_BATCH_SIZE": config.smoke_train_batch_size,
        "CEGR_V2_TRAIN_BATCH_SIZE": config.train_batch_size,
        "CEGR_V2_GROUP_SIZE": config.group_size,
        "CEGR_V2_LEARNING_RATE": config.learning_rate,
        "CEGR_V2_LR_WARMUP_RATIO": config.lr_warmup_steps_ratio,
        "CEGR_V2_SAVE_FREQ": config.save_freq,
        "CEGR_V2_SEED": config.seed,
        "CEGR_V2_ROLLOUT_ENGINE_SEED": config.rollout_engine_seed,
        "CEGR_V2_MINIMUM_SMOKE_SIGNAL": config.minimum_smoke_signal,
        "CEGR_V2_EVAL_BATCH_SIZE": config.eval_batch_size,
        "CEGR_V2_FINAL_PER_DATASET": config.final_examples_per_dataset,
        "CEGR_V2_BOOTSTRAP_SAMPLES": config.bootstrap_samples,
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
