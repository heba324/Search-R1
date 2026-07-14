"""Restore prompt-level rollout identities before Search-R1 computes GRPO advantage."""

from __future__ import annotations

from collections import Counter


def build_group_uids(data_sources, extra_infos, expected_group_size):
    if expected_group_size <= 1:
        raise ValueError("CEGR V2 requires a rollout group size greater than one")
    if len(data_sources) != len(extra_infos):
        raise ValueError("data_sources and extra_infos must have equal lengths")
    uids = []
    for data_source, extra_info in zip(data_sources, extra_infos):
        try:
            split = str(extra_info["split"])
            index = int(extra_info["index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid extra_info for rollout grouping: {extra_info!r}") from error
        uids.append(f"{data_source}:{split}:{index}")
    counts = Counter(uids)
    invalid = {uid: count for uid, count in counts.items() if count != expected_group_size}
    if invalid:
        raise ValueError(
            f"Every rollout group expected {expected_group_size} members; found {invalid}"
        )
    return uids
