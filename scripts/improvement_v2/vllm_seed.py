"""Inject an engine seed without turning it into a per-request sampling seed."""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def override_vllm_engine_seed(vllm_rollout_module, seed):
    seed = int(seed)
    if seed < 0:
        raise ValueError("vLLM engine seed must be nonnegative")
    original_llm = vllm_rollout_module.LLM

    def seeded_llm(*args, **kwargs):
        existing = kwargs.get("seed")
        if existing is not None and int(existing) != seed:
            raise ValueError(
                f"Conflicting vLLM engine seeds: {existing} and {seed}"
            )
        kwargs["seed"] = seed
        return original_llm(*args, **kwargs)

    vllm_rollout_module.LLM = seeded_llm
    try:
        yield
    finally:
        vllm_rollout_module.LLM = original_llm
