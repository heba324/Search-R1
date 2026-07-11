#!/usr/bin/env python3
"""Send a real request to the paper-v1 retriever and validate its response."""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from typing import Any, Mapping


def validate_response(payload: Mapping[str, Any], expected: int = 3) -> None:
    result = payload.get("result")
    if not isinstance(result, list) or not result or not isinstance(result[0], list):
        raise ValueError("Retriever response is missing result[0].")
    documents = result[0]
    if len(documents) != expected:
        raise ValueError(f"Retriever must return exactly {expected} documents; found {len(documents)}.")
    for item in documents:
        contents = item.get("document", {}).get("contents") if isinstance(item, dict) else None
        score = item.get("score") if isinstance(item, dict) else None
        if not isinstance(contents, str) or not contents.strip():
            raise ValueError("Retriever returned a document without contents.")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Retriever returned an invalid score.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    args = parser.parse_args()
    body = json.dumps({"queries": ["Who wrote Hamlet?"], "topk": 3, "return_scores": True}).encode()
    request = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    validate_response(payload)
    print("Paper v1 retriever API is ready and returned 3 valid documents.")


if __name__ == "__main__":
    main()
