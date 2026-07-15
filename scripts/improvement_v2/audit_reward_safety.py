#!/usr/bin/env python3
"""Audit EFF group invariants before spending GPU time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.improvement_v2.reward import score_group


def audit_groups(groups, expected_group_size=5):
    if not groups:
        raise ValueError("Reward audit requires at least one rollout group")
    all_zero_groups = 0
    informative_fallback_groups = 0
    nonzero_groups = 0
    mixed_group_reward_mismatches = 0
    for group in groups:
        if len(group) != expected_group_size:
            raise ValueError(
                f"Every group must contain {expected_group_size} rollouts; found {len(group)}"
            )
        aliases = group[0]["golden_answers"]
        if any(row["golden_answers"] != aliases for row in group):
            raise ValueError("A rollout group contains inconsistent golden answers")
        scored = score_group([row["trajectory"] for row in group], aliases)
        if any(item.exact_match for item in scored):
            nonzero_groups += 1
            mixed_group_reward_mismatches += sum(
                item.total != item.exact_match for item in scored
            )
        else:
            all_zero_groups += 1
            if len({item.total for item in scored}) > 1:
                informative_fallback_groups += 1

    safety_passed = mixed_group_reward_mismatches == 0
    return {
        "groups": len(groups),
        "expected_group_size": expected_group_size,
        "mixed_or_all_correct_groups": nonzero_groups,
        "mixed_group_reward_mismatches": mixed_group_reward_mismatches,
        "all_zero_groups": all_zero_groups,
        "informative_fallback_groups": informative_fallback_groups,
        "informative_fallback_rate": (
            informative_fallback_groups / all_zero_groups if all_zero_groups else 0.0
        ),
        "safety_invariant_passed": safety_passed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", type=Path, help="JSON file containing a list of rollout groups")
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-group-size", type=int, default=5)
    parser.add_argument("--minimum-informative-rate", type=float, default=0.10)
    args = parser.parse_args()
    groups = json.loads(args.groups.read_text(encoding="utf-8"))
    result = audit_groups(groups, expected_group_size=args.expected_group_size)
    result["minimum_informative_rate"] = args.minimum_informative_rate
    result["signal_gate_passed"] = (
        result["all_zero_groups"] > 0
        and result["informative_fallback_rate"] >= args.minimum_informative_rate
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not result["safety_invariant_passed"] or not result["signal_gate_passed"]:
        raise SystemExit("CEGR V2 reward audit failed; do not start the pilot")


if __name__ == "__main__":
    main()
