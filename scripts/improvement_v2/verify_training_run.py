#!/usr/bin/env python3
"""Verify that a V2 training arm completed its declared contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.improvement_v2.parse_v2_metrics import (
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


def verify_training_run(
    repo_root,
    run_name,
    method,
    steps,
    group_size,
    minimum_signal=0.0,
    rollout_engine_seed=42,
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
    args = parser.parse_args()
    errors = verify_training_run(
        args.repo_root.resolve(),
        args.run_name,
        args.method,
        args.steps,
        args.group_size,
        args.minimum_signal,
        args.rollout_engine_seed,
    )
    if errors:
        raise SystemExit("Training completion verification failed:\n- " + "\n- ".join(errors))
    print(f"Verified completed V2 run: {args.run_name}")


if __name__ == "__main__":
    main()
