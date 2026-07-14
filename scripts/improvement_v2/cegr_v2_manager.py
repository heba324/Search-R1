"""Grouped training-time reward managers for CEGR V2 and its EM control."""

from __future__ import annotations

from collections import defaultdict
import json
from statistics import mean

import numpy as np
import torch

from scripts.improvement_v2.cegr_v2_reward import score_batch_by_uid
from scripts.improvement_v2.grouping import build_group_uids
from verl import DataProto


class CEGRV2RewardManager:
    def __init__(self, tokenizer, num_examine: int, group_size: int, mode: str):
        if mode not in {"eff", "grouped_em"}:
            raise ValueError("mode must be eff or grouped_em")
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.group_size = group_size
        self.mode = mode
        self.training_step = 0

    def __call__(self, data: DataProto):
        self.training_step += 1
        uids = build_group_uids(
            data.non_tensor_batch["data_source"],
            data.non_tensor_batch["extra_info"],
            expected_group_size=self.group_size,
        )
        data.non_tensor_batch["uid"] = np.array(uids, dtype=object)
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        trajectories = []
        targets = []
        response_lengths = []
        for index in range(len(data)):
            item = data[index]
            prompt_length = item.batch["prompts"].shape[-1]
            response_ids = item.batch["responses"]
            response_length = int(item.batch["attention_mask"][prompt_length:].sum())
            trajectories.append(self.tokenizer.decode(response_ids[:response_length]))
            targets.append(item.non_tensor_batch["reward_model"]["ground_truth"]["target"])
            response_lengths.append(response_length)

        group_indices = defaultdict(list)
        for index, uid in enumerate(uids):
            group_indices[uid].append(index)

        breakdowns = score_batch_by_uid(uids, trajectories, targets, mode=self.mode)
        all_zero_groups = 0
        informative_fallback_groups = 0
        mixed_group_reward_mismatches = 0
        for indices in group_indices.values():
            scored = [breakdowns[index] for index in indices]
            if not any(item.exact_match for item in scored):
                all_zero_groups += 1
                if len({item.total for item in scored}) > 1:
                    informative_fallback_groups += 1
            else:
                mixed_group_reward_mismatches += sum(
                    item.total != item.exact_match for item in scored
                )
            for index, breakdown in zip(indices, scored):
                if response_lengths[index] > 0:
                    reward_tensor[index, response_lengths[index] - 1] = breakdown.total

        group_count = len(group_indices)
        metrics = {
            "step": self.training_step,
            "mode": self.mode,
            "reward": mean(item.total for item in breakdowns),
            "em": mean(item.exact_match for item in breakdowns),
            "f1_diagnostic": mean(item.token_f1 for item in breakdowns),
            "fallback_rollout_rate": mean(float(item.fallback_used) for item in breakdowns),
            "group_count": group_count,
            "group_size": self.group_size,
            "all_zero_group_count": all_zero_groups,
            "all_zero_group_rate": all_zero_groups / group_count,
            "informative_fallback_group_count": informative_fallback_groups,
            "informative_fallback_group_rate": (
                informative_fallback_groups / all_zero_groups if all_zero_groups else 0.0
            ),
            "mixed_group_reward_mismatches": mixed_group_reward_mismatches,
        }
        print("CEGR_V2_METRICS " + json.dumps(metrics, sort_keys=True))
        return reward_tensor
