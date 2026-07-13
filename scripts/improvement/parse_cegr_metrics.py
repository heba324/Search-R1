#!/usr/bin/env python3
"""Extract CEGR component metrics from a training log."""

import argparse
import json
from pathlib import Path


PREFIX = "CEGR_METRICS "


def parse_metrics(text: str):
    rows = []
    for line in text.splitlines():
        marker = line.find(PREFIX)
        if marker >= 0:
            rows.append(json.loads(line[marker + len(PREFIX) :]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = parse_metrics(args.log.read_text(encoding="utf-8", errors="replace"))
    if not rows:
        raise SystemExit("No CEGR_METRICS rows found")
    args.output.write_text(
        json.dumps({"method": "cegr", "steps": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
