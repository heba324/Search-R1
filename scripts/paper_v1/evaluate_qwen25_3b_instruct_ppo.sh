#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/verl_checkpoints/search-r1-v1-qwen2.5-3b-it-ppo-em/actor/global_step_300}"
EVAL_DATA="${EVAL_DATA:-$REPO_ROOT/data/paper_v1_eval/test.parquet}"
ARTIFACT_DIR="$REPO_ROOT/artifacts/paper-v1"
LOG_FILE="$ARTIFACT_DIR/evaluation-seven-datasets.log"
MARKER="$ARTIFACT_DIR/evaluation_completed.txt"

[ -s "$EVAL_DATA" ] || { echo "Missing seven-dataset parquet: $EVAL_DATA" >&2; exit 1; }
[ -s "$MODEL_PATH/config.json" ] || { echo "Missing Hugging Face actor checkpoint: $MODEL_PATH" >&2; exit 1; }
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
  data.train_files="$EVAL_DATA" data.val_files="$EVAL_DATA" \
  data.train_data_num=null data.val_data_num=null \
  data.train_batch_size=512 data.val_batch_size=256 \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=500 \
  algorithm.adv_estimator=gae actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size=64 actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
  actor_rollout_ref.rollout.n_agent=1 actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true critic.optim.lr=1e-5 critic.model.path="$MODEL_PATH" \
  critic.ppo_micro_batch_size=8 algorithm.kl_ctrl.kl_coef=0.001 algorithm.no_think_rl=false \
  trainer.critic_warmup=0 "trainer.logger=['console']" \
  +trainer.val_only=true +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null trainer.n_gpus_per_node=8 trainer.nnodes=1 \
  max_turns=4 retriever.url="http://127.0.0.1:8000/retrieve" retriever.topk=3 \
  2>&1 | tee "$LOG_FILE"

python3 scripts/paper_v1/parse_eval_metrics.py "$LOG_FILE" "$MARKER.tmp"
mv "$MARKER.tmp" "$MARKER"
echo "Evaluation completed: $MARKER"
