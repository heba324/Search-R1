"""Evaluation-only instrumentation for search behavior and paired evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.course_reproduction.search_behavior import (
    AGGREGATE_GROUP,
    METRIC_PREFIX,
    summarize_search_behavior,
)
from search_r1.llm_agent.generation import LLMGenerationManager
from scripts.improvement.evaluation_record import build_evaluation_record
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def install_search_behavior_instrumentation() -> None:
    """Add metrics without changing Search-R1 interaction or training logic."""
    original_validate = RayPPOTrainer._validate
    original_run_loop = LLMGenerationManager.run_llm_loop

    def validate_with_search_behavior(self) -> Dict[str, float]:
        rows: List[Dict[str, int]] = []
        evaluation_records = []
        original_reward_fn = self.val_reward_fn

        def traced_run_loop(manager, *args, **kwargs):
            output = original_run_loop(manager, *args, **kwargs)
            responses = output.batch["responses"]
            texts = manager.tokenizer.batch_decode(responses, skip_special_tokens=False)
            token_counts = (responses != manager.tokenizer.pad_token_id).sum(-1).tolist()
            for text, token_count in zip(texts, token_counts):
                row = summarize_search_behavior(text)
                row["response_tokens"] = int(token_count)
                rows.append(row)
            return output

        def traced_reward_fn(data):
            for index in range(len(data)):
                item = data[index]
                prompt_length = item.batch["prompts"].shape[-1]
                response_length = int(
                    item.batch["attention_mask"][prompt_length:].sum()
                )
                trajectory = self.tokenizer.decode(
                    item.batch["responses"][:response_length]
                )
                ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
                record = build_evaluation_record(
                    trajectory,
                    dataset=str(item.non_tensor_batch["data_source"]),
                    golden_answers=ground_truth["target"],
                    extra_info=item.non_tensor_batch.get("extra_info", {}),
                )
                record["response_tokens"] = response_length
                evaluation_records.append(record)
            return original_reward_fn(data)

        LLMGenerationManager.run_llm_loop = traced_run_loop
        self.val_reward_fn = traced_reward_fn
        try:
            metrics = original_validate(self)
        finally:
            LLMGenerationManager.run_llm_loop = original_run_loop
            self.val_reward_fn = original_reward_fn

        if rows:
            for key in rows[0]:
                metric_name = "search_rate" if key == "used_search" else f"avg_{key}"
                metrics[f"{METRIC_PREFIX}/{metric_name}/{AGGREGATE_GROUP}"] = float(
                    np.mean([row[key] for row in rows])
                )
        if evaluation_records:
            datasets = sorted({record["dataset"] for record in evaluation_records})
            for dataset in datasets:
                selected = [
                    record for record in evaluation_records if record["dataset"] == dataset
                ]
                metrics[f"val/answer_f1/{dataset}"] = float(
                    np.mean([record["f1"] for record in selected])
                )
                metrics[f"val/evidence_coverage/{dataset}"] = float(
                    np.mean([record["evidence_coverage"] for record in selected])
                )
                for key in (
                    "searches",
                    "valid_searches",
                    "duplicate_searches",
                    "invalid_searches",
                    "used_search",
                ):
                    metric_name = "search_rate" if key == "used_search" else f"avg_{key}"
                    metrics[f"{METRIC_PREFIX}/{metric_name}/{dataset}"] = float(
                        np.mean([record[key] for record in selected])
                    )

            output_path = os.environ.get("SEARCH_R1_EVAL_TRAJECTORIES")
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as handle:
                    for record in evaluation_records:
                        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return metrics

    RayPPOTrainer._validate = validate_with_search_behavior
