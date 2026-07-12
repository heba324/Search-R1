#!/usr/bin/env python3
"""Compare fixed-subset pre-RL and post-RL EM plus search behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_payloads(before, after):
    if before.get("evaluation_data_sha256") != after.get("evaluation_data_sha256"):
        raise ValueError("Pre-RL and post-RL evaluation data hashes differ")
    datasets = sorted(before["metrics"])
    if datasets != sorted(after["metrics"]):
        raise ValueError("Pre-RL and post-RL evaluation datasets differ")
    result = {
        "scope": "paired fixed-subset comparison; not paper-table numerical reproduction",
        "evaluation_data_sha256": before["evaluation_data_sha256"],
        "before_model": {
            "path": before.get("model_path"),
            "config_sha256": before.get("model_config_sha256"),
        },
        "after_model": {
            "path": after.get("model_path"),
            "config_sha256": after.get("model_config_sha256"),
        },
        "em_delta": {
            name: after["metrics"][name] - before["metrics"][name] for name in datasets
        },
        "search_behavior_delta": {},
    }
    behavior_groups = sorted(before["search_behavior"])
    if behavior_groups != sorted(after["search_behavior"]):
        raise ValueError("Pre-RL and post-RL search behavior groups differ")
    for name in behavior_groups:
        before_behavior = before["search_behavior"][name]
        after_behavior = after["search_behavior"][name]
        if set(before_behavior) != set(after_behavior):
            raise ValueError(f"Search behavior metrics differ for {name}")
        result["search_behavior_delta"][name] = {
            metric: after_behavior[metric] - before_behavior[metric]
            for metric in sorted(before_behavior)
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before = json.loads(args.before.read_text(encoding="utf-8"))
    after = json.loads(args.after.read_text(encoding="utf-8"))
    comparison = compare_payloads(before, after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
