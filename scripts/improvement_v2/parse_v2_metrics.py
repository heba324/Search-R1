#!/usr/bin/env python3
"""Extract CEGR V2 metrics and enforce grouping and reward invariants."""

import argparse
import json
import math
from pathlib import Path
import re


PREFIX = "CEGR_V2_METRICS "
NONFINITE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[-+]?nan|[-+]?inf)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def find_nonfinite_tokens(text):
    return [match.group(0).lower().lstrip("+") for match in NONFINITE_PATTERN.finditer(text)]


def _contains_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return False


def parse_metrics(text):
    rows = []
    for line in text.splitlines():
        marker = line.find(PREFIX)
        if marker >= 0:
            rows.append(json.loads(line[marker + len(PREFIX) :]))
    return rows


def summarize_signal(rows):
    eff_rows = [row for row in rows if row.get("mode") == "eff"]
    all_zero_groups = sum(row.get("all_zero_group_count", 0) for row in eff_rows)
    informative_groups = sum(
        row.get("informative_fallback_group_count", 0) for row in eff_rows
    )
    return {
        "all_zero_group_count": all_zero_groups,
        "informative_fallback_group_count": informative_groups,
        "informative_fallback_rate": (
            informative_groups / all_zero_groups if all_zero_groups else 0.0
        ),
    }


def validate_metrics(
    rows,
    minimum_informative_fallback_rate=0.0,
    expected_steps=None,
    expected_group_size=None,
):
    errors = []
    if any(_contains_nonfinite(row) for row in rows):
        errors.append("training reward metrics contain NaN or infinity")
    if expected_steps is not None and len(rows) != expected_steps:
        errors.append(f"Expected {expected_steps} training metric rows, found {len(rows)}")
    if expected_steps is not None:
        observed_steps = [row.get("step") for row in rows]
        if observed_steps != list(range(1, expected_steps + 1)):
            errors.append(
                f"training metric steps are not consecutive: {observed_steps}"
            )
    if expected_group_size is not None and any(
        row.get("group_size") != expected_group_size for row in rows
    ):
        errors.append(f"rollout group size differs from expected {expected_group_size}")
    if any(
        row.get("group_size", 0) <= 1 or row.get("group_count", 0) <= 0
        for row in rows
    ):
        errors.append("rollout groups were not restored")
    if any(row.get("mixed_group_reward_mismatches", -1) != 0 for row in rows):
        errors.append("a nonzero-EM group diverged from EM")
    if minimum_informative_fallback_rate > 0:
        signal = summarize_signal(rows)
        if signal["all_zero_group_count"] == 0:
            errors.append("no all-zero EM groups were observed for the EFF signal audit")
        elif (
            signal["informative_fallback_rate"]
            < minimum_informative_fallback_rate
        ):
            errors.append(
                "informative fallback rate "
                f"{signal['informative_fallback_rate']:.4f} is below "
                f"{minimum_informative_fallback_rate:.4f}"
            )
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--minimum-informative-fallback-rate", type=float, default=0.0
    )
    parser.add_argument("--expected-steps", type=int)
    parser.add_argument("--expected-group-size", type=int)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_informative_fallback_rate <= 1.0:
        raise SystemExit("minimum informative fallback rate must be in [0, 1]")
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    nonfinite_tokens = find_nonfinite_tokens(log_text)
    if nonfinite_tokens:
        raise SystemExit(
            "Training log contains non-finite numeric tokens: "
            + ", ".join(sorted(set(nonfinite_tokens)))
        )
    rows = parse_metrics(log_text)
    if not rows:
        raise SystemExit("No CEGR_V2_METRICS rows found")
    errors = validate_metrics(
        rows,
        args.minimum_informative_fallback_rate,
        expected_steps=args.expected_steps,
        expected_group_size=args.expected_group_size,
    )
    if errors:
        raise SystemExit("CEGR V2 invariant failed: " + "; ".join(errors))
    args.output.write_text(
        json.dumps(
            {
                "method": rows[0]["mode"],
                "signal_summary": summarize_signal(rows),
                "steps": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
