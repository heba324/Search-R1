#!/usr/bin/env python3
"""Analyze paired baseline and CEGR per-example evaluation records."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean


def _mcnemar_exact(baseline_wrong_improved_right: int, reverse: int) -> float:
    discordant = baseline_wrong_improved_right + reverse
    if discordant == 0:
        return 1.0
    lower = min(baseline_wrong_improved_right, reverse)
    probability = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def _bootstrap_ci(deltas, samples: int, seed: int):
    if not deltas:
        raise ValueError("Cannot bootstrap an empty paired sample")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(fmean(rng.choice(deltas) for _ in deltas))
    estimates.sort()
    low = estimates[int(0.025 * (samples - 1))]
    high = estimates[int(0.975 * (samples - 1))]
    return [low, high]


def _summarize(pairs, bootstrap_samples: int, seed: int):
    em_deltas = [improved["em"] - baseline["em"] for baseline, improved in pairs]
    f1_deltas = [improved["f1"] - baseline["f1"] for baseline, improved in pairs]
    positive = sum(
        baseline["em"] == 0 and improved["em"] == 1
        for baseline, improved in pairs
    )
    negative = sum(
        baseline["em"] == 1 and improved["em"] == 0
        for baseline, improved in pairs
    )
    summary = {
        "count": len(pairs),
        "baseline_em": fmean(baseline["em"] for baseline, _ in pairs),
        "improved_em": fmean(improved["em"] for _, improved in pairs),
        "em_delta": fmean(em_deltas),
        "em_delta_bootstrap_95_ci": _bootstrap_ci(
            em_deltas, bootstrap_samples, seed
        ),
        "baseline_f1": fmean(baseline["f1"] for baseline, _ in pairs),
        "improved_f1": fmean(improved["f1"] for _, improved in pairs),
        "f1_delta": fmean(f1_deltas),
        "baseline_wrong_improved_right": positive,
        "baseline_right_improved_wrong": negative,
        "mcnemar_exact_p": _mcnemar_exact(positive, negative),
    }
    for metric in (
        "evidence_coverage",
        "searches",
        "valid_searches",
        "duplicate_searches",
        "invalid_searches",
        "response_tokens",
    ):
        if all(metric in row for pair in pairs for row in pair):
            baseline_value = fmean(before[metric] for before, _ in pairs)
            improved_value = fmean(after[metric] for _, after in pairs)
            summary[f"baseline_{metric}"] = baseline_value
            summary[f"improved_{metric}"] = improved_value
            summary[f"{metric}_delta"] = improved_value - baseline_value
    return summary


def analyze_pairs(baseline, improved, bootstrap_samples: int = 10000, seed: int = 42):
    baseline_by_id = {row["example_id"]: row for row in baseline}
    improved_by_id = {row["example_id"]: row for row in improved}
    if set(baseline_by_id) != set(improved_by_id):
        raise ValueError("Baseline and improved evaluations contain different examples")
    if not baseline_by_id:
        raise ValueError("Evaluation files are empty")

    pairs = []
    by_dataset = {}
    for example_id in sorted(baseline_by_id):
        before = baseline_by_id[example_id]
        after = improved_by_id[example_id]
        if before["dataset"] != after["dataset"]:
            raise ValueError(f"Dataset mismatch for {example_id}")
        pair = (before, after)
        pairs.append(pair)
        by_dataset.setdefault(before["dataset"], []).append(pair)

    single_hop = {"nq", "triviaqa", "popqa"}
    multi_hop = {"hotpotqa", "2wikimultihopqa", "musique", "bamboogle"}
    groups = {}
    for name, members in (("single_hop", single_hop), ("multi_hop", multi_hop)):
        selected = [pair for pair in pairs if pair[0]["dataset"] in members]
        if selected:
            groups[name] = _summarize(selected, bootstrap_samples, seed)

    return {
        "scope": "paired fixed-subset Search-R1 baseline versus CEGR",
        "overall": _summarize(pairs, bootstrap_samples, seed),
        "groups": groups,
        "datasets": {
            dataset: _summarize(rows, bootstrap_samples, seed)
            for dataset, rows in sorted(by_dataset.items())
        },
    }


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("improved", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze_pairs(
        _read_jsonl(args.baseline),
        _read_jsonl(args.improved),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
