#!/usr/bin/env python3
"""Verify that a V2 training arm completed its declared contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.improvement_v2.parse_metrics import (
    find_nonfinite_tokens,
    summarize_signal,
    validate_metrics,
)


def _read_marker(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def verify_training(
    repo_root,
    run_name,
    method,
    steps,
    group_size,
    minimum_signal=0.0,
    rollout_engine_seed=42,
    seed=None,
    initial_model=None,
    train_batch_size=None,
    learning_rate=None,
    lr_warmup_ratio=None,
):
    artifact = repo_root / "artifacts/improvement-v2" / run_name
    marker_path = artifact / "training_completed.txt"
    metrics_path = artifact / "reward_metrics.json"
    log_path = artifact / "train.log"
    checkpoint = (
        repo_root
        / "verl_checkpoints"
        / run_name
        / "actor"
        / f"global_step_{steps}"
        / "config.json"
    )
    errors = []
    for label, path in (
        ("checkpoint", checkpoint),
        ("completion marker", marker_path),
        ("reward metrics", metrics_path),
        ("training log", log_path),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing {label}: {path}")
    if errors:
        return errors

    marker = _read_marker(marker_path)
    nonfinite_tokens = find_nonfinite_tokens(
        log_path.read_text(encoding="utf-8", errors="replace")
    )
    if nonfinite_tokens:
        errors.append(
            "training log contains non-finite numeric tokens: "
            + ", ".join(sorted(set(nonfinite_tokens)))
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return [f"Could not read reward metrics: {error}"]
    if marker.get("status") != "completed":
        errors.append("training marker status is not completed")
    if marker.get("method") != method or metrics.get("method") != method:
        errors.append(f"training method does not match {method}")
    if marker.get("training_steps") != str(steps):
        errors.append(f"training marker does not record {steps} steps")
    if marker.get("group_size") != str(group_size):
        errors.append(f"training marker does not record group size {group_size}")
    if marker.get("rollout_engine_seed") != str(rollout_engine_seed):
        errors.append(
            f"training marker does not record vLLM engine seed {rollout_engine_seed}"
        )
    if seed is not None and marker.get("seed") != str(seed):
        errors.append(f"training marker does not record driver seed {seed}")
    if initial_model is not None:
        recorded_initial = marker.get("initial_checkpoint")
        if recorded_initial is None or Path(recorded_initial).resolve() != Path(
            initial_model
        ).resolve():
            errors.append(
                f"training marker does not use declared initial model {initial_model}"
            )
    if train_batch_size is not None and marker.get("train_batch_size") != str(
        train_batch_size
    ):
        errors.append(
            f"training marker does not record train batch size {train_batch_size}"
        )

    def verify_float(key, expected, label):
        if expected is None:
            return
        try:
            recorded = float(marker[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"training marker does not record {label} {expected}")
            return
        if not math.isclose(recorded, expected, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(
                f"training marker records {label} {recorded}, expected {expected}"
            )

    verify_float("learning_rate", learning_rate, "learning rate")
    verify_float("lr_warmup_steps_ratio", lr_warmup_ratio, "warmup ratio")

    rows = metrics.get("steps", [])
    if not isinstance(rows, list):
        return errors + ["reward metric steps are not a list"]
    errors.extend(
        validate_metrics(
            rows,
            minimum_informative_fallback_rate=minimum_signal,
            expected_steps=steps,
            expected_group_size=group_size,
        )
    )
    if any(row.get("mode") != method for row in rows):
        errors.append(f"reward metric rows do not all use method {method}")
    recorded_signal = metrics.get("signal_summary")
    if recorded_signal != summarize_signal(rows):
        errors.append("stored signal summary does not match the raw reward metrics")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", choices=("eff", "grouped_em"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--minimum-signal", type=float, default=0.0)
    parser.add_argument("--rollout-engine-seed", type=int, default=42)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-warmup-ratio", type=float)
    args = parser.parse_args()
    errors = verify_training(
        args.repo_root.resolve(),
        args.run_name,
        args.method,
        args.steps,
        args.group_size,
        args.minimum_signal,
        args.rollout_engine_seed,
        args.seed,
        args.initial_model,
        args.train_batch_size,
        args.learning_rate,
        args.lr_warmup_ratio,
    )
    if errors:
        raise SystemExit("Training completion verification failed:\n- " + "\n- ".join(errors))
    print(f"Verified completed V2 run: {args.run_name}")


if __name__ == "__main__":
    main()
