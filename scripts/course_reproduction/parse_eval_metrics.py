#!/usr/bin/env python3
"""Parse seven-dataset course evaluation metrics into JSON."""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.paper_v1.contract import REQUIRED_EVALUATION_DATASETS
from scripts.paper_v1.parse_eval_metrics import parse_metrics
from scripts.course_reproduction.search_behavior import AGGREGATE_GROUP, METRIC_PREFIX


SEARCH_BEHAVIOR_PATTERN = re.compile(
    re.escape(METRIC_PREFIX) + r"/([a-z_]+)/([a-z0-9_]+)['\"]?:\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_search_behavior(text: str):
    behavior = {}
    for metric, dataset, value in SEARCH_BEHAVIOR_PATTERN.findall(text):
        behavior.setdefault(dataset, {})[metric] = float(value)
    return behavior


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    metrics = parse_metrics(log_text)
    search_behavior = parse_search_behavior(log_text)
    missing = [name for name in REQUIRED_EVALUATION_DATASETS if name not in metrics]
    if missing:
        raise SystemExit("Missing evaluation metric(s): " + ", ".join(missing))
    if AGGREGATE_GROUP not in search_behavior:
        raise SystemExit("Missing aggregate search behavior metrics")
    selected = {name: metrics[name] for name in REQUIRED_EVALUATION_DATASETS}
    payload = {
        "run_name": args.run_name,
        "scope": "resource-limited method reproduction; not paper-table numerical reproduction",
        "metrics": selected,
        "average_em": sum(selected.values()) / len(selected),
        "search_behavior": {
            AGGREGATE_GROUP: search_behavior[AGGREGATE_GROUP]
        },
        "elapsed_seconds": args.elapsed_seconds,
        "evaluation_data_path": str(args.eval_data.resolve()),
        "evaluation_data_sha256": sha256_file(args.eval_data),
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": sha256_file(args.model_path / "config.json"),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
