"""Pure construction of per-example evaluation evidence."""

from __future__ import annotations

from scripts.improvement.cegr_reward import (
    evidence_answer_coverage,
    exact_match,
    extract_final_answer,
    token_f1,
)
from scripts.course_reproduction.search_behavior import summarize_search_behavior


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
