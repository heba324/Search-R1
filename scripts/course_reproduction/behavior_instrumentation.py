"""Evaluation-only instrumentation for aggregate search behavior metrics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from scripts.course_reproduction.search_behavior import (
    AGGREGATE_GROUP,
    METRIC_PREFIX,
    summarize_search_behavior,
)
from search_r1.llm_agent.generation import LLMGenerationManager
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def install_search_behavior_instrumentation() -> None:
    """Add metrics without changing Search-R1 interaction or training logic."""
    original_validate = RayPPOTrainer._validate
    original_run_loop = LLMGenerationManager.run_llm_loop

    def validate_with_search_behavior(self) -> Dict[str, float]:
        rows: List[Dict[str, int]] = []

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

        LLMGenerationManager.run_llm_loop = traced_run_loop
        try:
            metrics = original_validate(self)
        finally:
            LLMGenerationManager.run_llm_loop = original_run_loop

        if rows:
            for key in rows[0]:
                metric_name = "search_rate" if key == "used_search" else f"avg_{key}"
                metrics[f"{METRIC_PREFIX}/{metric_name}/{AGGREGATE_GROUP}"] = float(
                    np.mean([row[key] for row in rows])
                )
        return metrics

    RayPPOTrainer._validate = validate_with_search_behavior
