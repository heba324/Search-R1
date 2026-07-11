#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/nq_hotpotqa_train}"
BASE_MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
BASE_MODEL="${BASE_MODEL:-$REPO_ROOT/data/models/Qwen2.5-3B-Instruct}"
EXPERIMENT_NAME="search-r1-v1-qwen2.5-3b-it-ppo-em"
ARTIFACT_DIR="$REPO_ROOT/artifacts/paper-v1"
LOG_FILE="$ARTIFACT_DIR/$EXPERIMENT_NAME.log"
MARKER="$ARTIFACT_DIR/training_completed.txt"

mkdir -p "$ARTIFACT_DIR"
rm -f "$MARKER" "$MARKER.tmp"
python3 scripts/paper_v1/preflight.py --repo-root "$REPO_ROOT" --require-assets
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SEARCH_ENV"
python scripts/paper_v1/check_retriever.py

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1

python3 -m verl.trainer.main_ppo \
  data.train_files="$REPO_ROOT/data/nq_hotpotqa_train/train.parquet" \
  data.val_files="$REPO_ROOT/data/nq_hotpotqa_train/test.parquet" \
  data.train_data_num=null data.val_data_num=null \
  data.train_batch_size=512 data.val_batch_size=256 \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=500 \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=gae \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
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
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  critic.optim.lr=1e-5 critic.model.use_remove_padding=True \
  critic.optim.lr_warmup_steps_ratio=0.05 critic.model.path="$BASE_MODEL" \
  critic.model.enable_gradient_checkpointing=true critic.ppo_micro_batch_size=8 \
  critic.model.fsdp_config.param_offload=true \
  critic.model.fsdp_config.grad_offload=true \
  critic.model.fsdp_config.optimizer_offload=true \
  algorithm.kl_ctrl.kl_coef=0.001 algorithm.no_think_rl=false \
  trainer.critic_warmup=0 "trainer.logger=['wandb']" \
  +trainer.val_only=false +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  trainer.save_freq=100 trainer.test_freq=50 \
  trainer.project_name=Search-R1-v1 trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs=15 trainer.total_training_steps=305 \
  trainer.default_local_dir="$REPO_ROOT/verl_checkpoints/$EXPERIMENT_NAME" \
  max_turns=4 retriever.url="http://127.0.0.1:8000/retrieve" retriever.topk=3 \
  2>&1 | tee "$LOG_FILE"

printf 'status=completed\nmodel=%s\nalgorithm=ppo\ntraining_steps=305\nmax_turns=4\ntopk=3\n' \
  "$BASE_MODEL_ID" > "$MARKER.tmp"
mv "$MARKER.tmp" "$MARKER"
echo "Training completed: $MARKER"
