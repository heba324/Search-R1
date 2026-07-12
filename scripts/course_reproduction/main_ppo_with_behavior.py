"""Run Search-R1 PPO/GRPO with evaluation-only search behavior metrics."""

from scripts.course_reproduction.behavior_instrumentation import (
    install_search_behavior_instrumentation,
)

install_search_behavior_instrumentation()

from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    main()
