"""Construct V2 evaluation records with the official case-sensitive answer syntax."""

from __future__ import annotations

import re

from scripts.improvement.cegr_reward import (
    exact_match,
    normalize_answer,
    token_f1,
)
from scripts.improvement_v2.cegr_v2_reward import extract_final_answer


SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.DOTALL)
INFORMATION_PATTERN = re.compile(r"<information>(.*?)</information>", re.DOTALL)
UNINFORMATIVE_ALIASES = {"yes", "no", "true", "false", "unknown", "none"}


def _normalize_query(query):
    return " ".join(query.casefold().split())


def summarize_search_behavior(trajectory):
    policy_text = INFORMATION_PATTERN.sub("", trajectory)
    attempts = len(re.findall(r"<search>", policy_text))
    queries = [_normalize_query(query) for query in SEARCH_PATTERN.findall(policy_text)]
    valid_queries = [query for query in queries if query]
    return {
        "searches": attempts,
        "valid_searches": len(valid_queries),
        "duplicate_searches": len(valid_queries) - len(set(valid_queries)),
        "invalid_searches": attempts - len(valid_queries),
        "used_search": int(attempts > 0),
    }


def evidence_answer_coverage(trajectory, aliases):
    evidence = normalize_answer(" ".join(INFORMATION_PATTERN.findall(trajectory)))
    if not evidence:
        return 0.0
    padded_evidence = f" {evidence} "
    for alias in aliases:
        normalized_alias = normalize_answer(str(alias))
        if len(normalized_alias) < 4 or normalized_alias in UNINFORMATIVE_ALIASES:
            continue
        if f" {normalized_alias} " in padded_evidence:
            return 1.0
    return 0.0


def build_evaluation_record(
    trajectory: str, dataset: str, golden_answers, extra_info
):
    aliases = [golden_answers] if isinstance(golden_answers, str) else list(golden_answers)
    prediction = extract_final_answer(trajectory) or ""
    split = str(extra_info.get("split", "unknown"))
    index = int(extra_info.get("index", -1))
    behavior = summarize_search_behavior(trajectory)
    return {
        "example_id": f"{dataset}:{split}:{index}",
        "dataset": dataset,
        "split": split,
        "index": index,
        "prediction": prediction,
        "golden_answers": [str(alias) for alias in aliases],
        "em": float(exact_match(prediction, aliases)),
        "f1": token_f1(prediction, aliases),
        "evidence_coverage": evidence_answer_coverage(trajectory, aliases),
        "trajectory": trajectory,
        **behavior,
    }
