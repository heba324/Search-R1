#!/usr/bin/env bash
set -euo pipefail

# Full GRPO run using the official Search-R1 scale. Run this only after the
# smoke test succeeds.

SEARCH_ENV="${SEARCH_ENV:-searchr1}"
DATA_DIR="${DATA_DIR:-$PWD/data/nq_search}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nq-search-r1-grpo-qwen2.5-3b-em}"
WAND_PROJECT="${WAND_PROJECT:-Search-R1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SEARCH_ENV"

python scripts/cloud_check_retriever.py

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=512 \
  data.val_batch_size=256 \
  data.max_prompt_length=4096 \
  data.max_response_length=500 \
  data.max_start_length=2048 \
  data.max_obs_length=500 \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size=64 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent=5 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  "trainer.logger=['wandb']" \
  +trainer.val_only=false \
  +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=50 \
  trainer.project_name="$WAND_PROJECT" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs=15 \
  trainer.total_training_steps=1005 \
  trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
  max_turns=2 \
  retriever.url="http://127.0.0.1:8000/retrieve" \
  retriever.topk=3 \
  2>&1 | tee "$EXPERIMENT_NAME.log"
