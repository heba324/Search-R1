"""Install V2's vLLM engine seed without subclassing the verl worker."""

from __future__ import annotations

import importlib

from scripts.improvement_v2.vllm_seed import override_vllm_engine_seed


def install_seeded_rollout_patch(worker_cls):
    """Patch the upstream worker in place and preserve its Ray inheritance shape."""
    original_build_rollout = worker_cls._build_rollout
    if getattr(original_build_rollout, "_cegr_v2_seed_patch", False):
        return worker_cls

    def seeded_build_rollout(self):
        if self.config.rollout.name != "vllm":
            return original_build_rollout(self)
        rollout_module = importlib.import_module(
            "verl.workers.rollout.vllm_rollout.vllm_rollout"
        )
        engine_seed = int(self.config.rollout.get("engine_seed", 42))
        with override_vllm_engine_seed(rollout_module, engine_seed):
            return original_build_rollout(self)

    seeded_build_rollout._cegr_v2_seed_patch = True
    worker_cls._build_rollout = seeded_build_rollout
    return worker_cls
