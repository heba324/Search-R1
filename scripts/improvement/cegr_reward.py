"""Curriculum Evidence-Guided Reward (CEGR) for search-augmented QA."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from scripts.course_reproduction.search_behavior import summarize_search_behavior


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
INFORMATION_PATTERN = re.compile(
    r"<information>(.*?)</information>", re.IGNORECASE | re.DOTALL
)
UNINFORMATIVE_ALIASES = {"yes", "no", "true", "false", "unknown", "none"}
PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Apply the same SQuAD-style normalization used by Search-R1 EM."""
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in PUNCTUATION
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, golden_answers: Iterable[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return any(
        normalized_prediction == normalize_answer(alias) for alias in golden_answers
    )


@dataclass(frozen=True)
class RewardWeights:
    em_share: float
    evidence_weight: float
    behavior_penalty_weight: float


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    exact_match: float
    token_f1: float
    evidence_coverage: float
    behavior_penalty: float
    weights: RewardWeights


def _as_aliases(golden_answers: str | Iterable[str]) -> list[str]:
    if isinstance(golden_answers, str):
        return [golden_answers]
    return [str(answer) for answer in golden_answers]


def extract_final_answer(trajectory: str) -> str | None:
    matches = ANSWER_PATTERN.findall(trajectory)
    return matches[-1].strip() if matches else None


def token_f1(prediction: str, golden_answers: str | Iterable[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    best = 0.0
    for alias in _as_aliases(golden_answers):
        gold_tokens = normalize_answer(alias).split()
        if not gold_tokens:
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def _is_informative_alias(alias: str) -> bool:
    normalized = normalize_answer(alias)
    return len(normalized) >= 4 and normalized not in UNINFORMATIVE_ALIASES


def evidence_answer_coverage(
    trajectory: str, golden_answers: str | Iterable[str]
) -> float:
    """Return one when retrieved evidence contains an informative gold alias."""
    evidence = normalize_answer(" ".join(INFORMATION_PATTERN.findall(trajectory)))
    if not evidence:
        return 0.0
    padded_evidence = f" {evidence} "
    for alias in _as_aliases(golden_answers):
        if not _is_informative_alias(alias):
            continue
        normalized_alias = normalize_answer(alias)
        if f" {normalized_alias} " in padded_evidence:
            return 1.0
    return 0.0


def search_behavior_penalty(trajectory: str, max_turns: int = 4) -> float:
    """Penalize invalid, duplicate, and fourth-or-later search actions."""
    summary = summarize_search_behavior(trajectory)
    attempts = max(summary["searches"], 1)
    invalid = summary["invalid_searches"] / attempts
    duplicate = summary["duplicate_searches"] / attempts
    excess = max(summary["valid_searches"] - 3, 0) / max(max_turns, 1)
    return min(1.0, 0.4 * invalid + 0.4 * duplicate + 0.2 * excess)


def reward_weights(step: int, total_steps: int) -> RewardWeights:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    bounded_step = min(max(step, 1), total_steps)
    progress = 1.0 if total_steps == 1 else (bounded_step - 1) / (total_steps - 1)
    return RewardWeights(
        em_share=0.60 + 0.30 * progress,
        evidence_weight=0.15 - 0.10 * progress,
        behavior_penalty_weight=0.02 + 0.06 * progress,
    )


def score_trajectory(
    trajectory: str,
    golden_answers: str | Sequence[str],
    step: int,
    total_steps: int,
    max_turns: int = 4,
) -> RewardBreakdown:
    """Score one policy trajectory while preserving EM as the final priority."""
    aliases = _as_aliases(golden_answers)
    answer = extract_final_answer(trajectory)
    exact_match_score = float(answer is not None and exact_match(answer, aliases))
    f1 = 0.0 if answer is None else token_f1(answer, aliases)
    evidence = evidence_answer_coverage(trajectory, aliases)
    behavior = search_behavior_penalty(trajectory, max_turns=max_turns)
    weights = reward_weights(step=step, total_steps=total_steps)

    answer_quality = (
        weights.em_share * exact_match_score + (1 - weights.em_share) * f1
    )
    total = (
        (1 - weights.evidence_weight) * answer_quality
        + weights.evidence_weight * evidence
        - weights.behavior_penalty_weight * behavior
    )
    total = min(1.0, max(-1.0, total))
    return RewardBreakdown(
        total=total,
        exact_match=exact_match_score,
        token_f1=f1,
        evidence_coverage=evidence,
        behavior_penalty=behavior,
        weights=weights,
    )
