#!/usr/bin/env python3
"""Parse seven-dataset course evaluation metrics into JSON."""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.paper_v1.contract import REQUIRED_EVALUATION_DATASETS
from scripts.paper_v1.parse_eval_metrics import parse_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    metrics = parse_metrics(args.log.read_text(encoding="utf-8", errors="replace"))
    missing = [name for name in REQUIRED_EVALUATION_DATASETS if name not in metrics]
    if missing:
        raise SystemExit("Missing evaluation metric(s): " + ", ".join(missing))
    selected = {name: metrics[name] for name in REQUIRED_EVALUATION_DATASETS}
    payload = {
        "run_name": args.run_name,
        "scope": "resource-limited method reproduction; not paper-table numerical reproduction",
        "metrics": selected,
        "average_em": sum(selected.values()) / len(selected),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
