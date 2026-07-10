#!/usr/bin/env python3
"""Verify that the Search-R1 retriever API is reachable and well formed."""

import os
import sys
from typing import Any, Dict, List

import requests


def validate_response(data: Any, expected_topk: int) -> List[Dict[str, Any]]:
    if not isinstance(data, dict) or "result" not in data:
        raise ValueError("Retriever response must contain a 'result' field.")

    result = data["result"]
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], list):
        raise ValueError("Retriever 'result' must contain one list for the one test query.")

    items = result[0]
    if len(items) < expected_topk:
        raise ValueError(
            f"Retriever returned {len(items)} documents; expected {expected_topk}."
        )

    validated: List[Dict[str, Any]] = []
    for index, item in enumerate(items[:expected_topk], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Retriever result {index} must be an object.")
        document = item.get("document")
        if not isinstance(document, dict):
            raise ValueError(f"Retriever result {index} is missing its document object.")
        contents = document.get("contents")
        if not isinstance(contents, str) or not contents.strip():
            raise ValueError(f"Retriever result {index} document is missing non-empty contents.")
        score = item.get("score")
        if not isinstance(score, (int, float)):
            raise ValueError(f"Retriever result {index} is missing a numeric score.")
        validated.append(item)
    return validated


def main() -> int:
    url = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
    try:
        topk = int(os.environ.get("TOPK", "3"))
    except ValueError as exc:
        raise ValueError("TOPK must be an integer.") from exc
    if topk < 1:
        raise ValueError("TOPK must be at least 1.")

    payload = {
        "queries": ["Who wrote Hamlet?"],
        "topk": topk,
        "return_scores": True,
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    items = validate_response(response.json(), expected_topk=topk)

    print(f"retriever ok: {url}")
    for index, item in enumerate(items, start=1):
        contents = item["document"]["contents"]
        title = contents.split("\n", 1)[0]
        print(f"{index}. score={item['score']} title={title}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (requests.RequestException, ValueError) as exc:
        print(f"Retriever check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
