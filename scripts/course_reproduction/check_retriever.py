#!/usr/bin/env python3
"""Verify that the course BM25 API returns exactly three scored documents."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.paper_v1.check_retriever import validate_response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    args = parser.parse_args()
    body = json.dumps({"queries": ["Who wrote Hamlet?"], "topk": 3, "return_scores": True}).encode()
    request = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    validate_response(payload)
    print("Course BM25 API is ready and returned 3 valid documents.")


if __name__ == "__main__":
    main()
