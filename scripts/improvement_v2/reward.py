"""EM-first, F1-fallback group reward for CEGR V2."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from scripts.improvement.cegr_reward import exact_match, token_f1


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class CEGRV2RewardBreakdown:
    total: float
    exact_match: float
    token_f1: float
    fallback_used: bool


def _as_aliases(golden_answers: str | Iterable[str]) -> list[str]:
    if isinstance(golden_answers, str):
        return [golden_answers]
    try:
        return [str(answer) for answer in golden_answers]
    except TypeError:
        return [str(golden_answers)]


def extract_final_answer(trajectory: str) -> str | None:
    """Match the environment's case-sensitive answer tags and use the final answer."""
    matches = ANSWER_PATTERN.findall(trajectory)
    return matches[-1].strip() if matches else None


def score_group(
    trajectories: Sequence[str],
    golden_answers: str | Iterable[str],
    use_f1_fallback: bool = True,
) -> list[CEGRV2RewardBreakdown]:
    """Use pure EM if any rollout is exact; otherwise rank failures by token F1."""
    if not trajectories:
        raise ValueError("score_group requires at least one trajectory")
    aliases = _as_aliases(golden_answers)
    answers = [extract_final_answer(trajectory) for trajectory in trajectories]
    exact_scores = [
        float(answer is not None and exact_match(answer, aliases)) for answer in answers
    ]
    f1_scores = [
        0.0 if answer is None else token_f1(answer, aliases) for answer in answers
    ]
    fallback_used = use_f1_fallback and not any(exact_scores)
    totals = f1_scores if fallback_used else exact_scores
    return [
        CEGRV2RewardBreakdown(
            total=total,
            exact_match=exact,
            token_f1=f1,
            fallback_used=fallback_used,
        )
        for total, exact, f1 in zip(totals, exact_scores, f1_scores)
    ]


def score_batch_by_uid(
    uids: Sequence[str],
    trajectories: Sequence[str],
    golden_answers: Sequence[str | Iterable[str]],
    mode: str,
) -> list[CEGRV2RewardBreakdown]:
    """Score each prompt group while preserving the original rollout order."""
    if mode not in {"eff", "grouped_em"}:
        raise ValueError("mode must be eff or grouped_em")
    if not (len(uids) == len(trajectories) == len(golden_answers)):
        raise ValueError("uids, trajectories, and golden_answers must have equal lengths")
    if not uids:
        raise ValueError("score_batch_by_uid requires at least one rollout")

    group_indices: dict[str, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        group_indices[str(uid)].append(index)

    breakdowns: list[CEGRV2RewardBreakdown | None] = [None] * len(uids)
    for indices in group_indices.values():
        aliases = _as_aliases(golden_answers[indices[0]])
        canonical_aliases = tuple(sorted(aliases))
        for index in indices[1:]:
            if tuple(sorted(_as_aliases(golden_answers[index]))) != canonical_aliases:
                raise ValueError("Rollouts in one prompt group have different gold answers")

        scored = score_group(
            [trajectories[index] for index in indices],
            aliases,
            use_f1_fallback=mode == "eff",
        )
        for index, breakdown in zip(indices, scored):
            breakdowns[index] = breakdown

    if any(breakdown is None for breakdown in breakdowns):
        raise RuntimeError("A rollout was not assigned a reward")
    return [breakdown for breakdown in breakdowns if breakdown is not None]
