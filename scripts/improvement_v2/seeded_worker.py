"""V2-only FSDP worker that passes the declared seed to the vLLM engine."""

from __future__ import annotations

import importlib

from scripts.improvement_v2.vllm_seed import override_vllm_engine_seed
from verl.workers.fsdp_workers import ActorRolloutRefWorker


class CEGRV2ActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_rollout(self):
        if self.config.rollout.name != "vllm":
            return super()._build_rollout()
        rollout_module = importlib.import_module(
            "verl.workers.rollout.vllm_rollout.vllm_rollout"
        )
        engine_seed = int(self.config.rollout.get("engine_seed", 42))
        with override_vllm_engine_seed(rollout_module, engine_seed):
            return super()._build_rollout()
