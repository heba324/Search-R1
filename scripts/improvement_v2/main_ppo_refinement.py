"""Run equal-budget EM control or CEGR V2 refinement without editing upstream code."""

import random

import hydra
import numpy as np
import ray
import torch

from scripts.improvement_v2.cegr_v2_manager import CEGRV2RewardManager
from verl.trainer.main_ppo import RewardManager


@hydra.main(
    config_path="../../verl/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config):
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            }
        )
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from omegaconf import OmegaConf
    from pprint import pprint
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    reward_strategy = config.get("reward_strategy", {})
    seed = int(reward_strategy.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if bool(config.trainer.get("val_only", False)):
        import scripts.course_reproduction.behavior_instrumentation as instrumentation
        from scripts.improvement_v2.evaluation_record import (
            build_evaluation_record as build_v2_evaluation_record,
            summarize_search_behavior as summarize_v2_search_behavior,
        )

        instrumentation.build_evaluation_record = build_v2_evaluation_record
        instrumentation.summarize_search_behavior = summarize_v2_search_behavior
        instrumentation.install_search_behavior_instrumentation()
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from scripts.improvement_v2.seeded_worker import install_seeded_rollout_patch
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        ActorRolloutRefWorker = install_seeded_rollout_patch(ActorRolloutRefWorker)
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, RayPPOTrainer

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_name = reward_strategy.get("name")
    if reward_name not in {"eff", "grouped_em"}:
        raise ValueError("reward_strategy.name must be eff or grouped_em")
    reward_fn = CEGRV2RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        group_size=int(reward_strategy["group_size"]),
        mode=reward_name,
    )
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        ),
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
