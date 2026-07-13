"""Training-time reward manager for CEGR."""

from __future__ import annotations

import json
from statistics import mean

import torch

from scripts.improvement.cegr_reward import score_trajectory
from verl import DataProto


class CEGRRewardManager:
    def __init__(self, tokenizer, num_examine: int, total_steps: int, max_turns: int = 4):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.total_steps = total_steps
        self.max_turns = max_turns
        self.training_step = 0

    def __call__(self, data: DataProto):
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        self.training_step += 1
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        breakdowns = []

        for index in range(len(data)):
            item = data[index]
            prompt_length = item.batch["prompts"].shape[-1]
            response_ids = item.batch["responses"]
            response_length = int(item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:response_length]
            trajectory = self.tokenizer.decode(valid_response_ids)
            targets = item.non_tensor_batch["reward_model"]["ground_truth"]["target"]
            breakdown = score_trajectory(
                trajectory,
                targets,
                step=self.training_step,
                total_steps=self.total_steps,
                max_turns=self.max_turns,
            )
            breakdowns.append(breakdown)
            reward_tensor[index, response_length - 1] = breakdown.total

        if breakdowns:
            weights = breakdowns[0].weights
            metrics = {
                "step": self.training_step,
                "reward": mean(item.total for item in breakdowns),
                "em": mean(item.exact_match for item in breakdowns),
                "f1": mean(item.token_f1 for item in breakdowns),
                "evidence_coverage": mean(
                    item.evidence_coverage for item in breakdowns
                ),
                "behavior_penalty": mean(item.behavior_penalty for item in breakdowns),
                "em_share": weights.em_share,
                "evidence_weight": weights.evidence_weight,
                "behavior_penalty_weight": weights.behavior_penalty_weight,
            }
            print("CEGR_METRICS " + json.dumps(metrics, sort_keys=True))

        return reward_tensor
