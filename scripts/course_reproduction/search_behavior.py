"""Pure helpers for measuring search behavior in generated trajectories."""

from __future__ import annotations

import re
from typing import Dict


SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
INFORMATION_PATTERN = re.compile(
    r"<information>.*?</information>", re.IGNORECASE | re.DOTALL
)
METRIC_PREFIX = "val/search_behavior"
AGGREGATE_GROUP = "overall"


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def summarize_search_behavior(text: str) -> Dict[str, int]:
    """Count search attempts, malformed/empty queries, and repeated queries."""
    policy_text = INFORMATION_PATTERN.sub("", text)
    attempts = len(re.findall(r"<search>", policy_text, flags=re.IGNORECASE))
    queries = [_normalize_query(query) for query in SEARCH_PATTERN.findall(policy_text)]
    valid_queries = [query for query in queries if query]
    return {
        "searches": attempts,
        "valid_searches": len(valid_queries),
        "duplicate_searches": len(valid_queries) - len(set(valid_queries)),
        "invalid_searches": attempts - len(valid_queries),
        "used_search": int(attempts > 0),
    }
