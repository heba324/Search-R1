#!/usr/bin/env python3
"""Extract and validate Search-R1 paper-v1 EM metrics from a veRL log."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.paper_v1.contract import PAPER_V1, assess_result_metrics

METRIC_PATTERN = re.compile(r"val/test_score/([a-z0-9_]+):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")


def parse_metrics(text: str) -> Dict[str, float]:
    return {name: float(value) for name, value in METRIC_PATTERN.findall(text)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metrics = parse_metrics(args.log.read_text(encoding="utf-8", errors="replace"))
    errors = assess_result_metrics(metrics)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    average = sum(metrics.values()) / len(metrics)
    payload = {
        "paper": "arXiv:2503.09516v1",
        "metrics": metrics,
        "average_em": average,
        "paper_target_average_em": PAPER_V1.target_average_em,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Seven-dataset average EM: {average:.3f} (paper target: {PAPER_V1.target_average_em:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
