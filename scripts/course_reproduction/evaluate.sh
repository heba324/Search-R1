#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-search-r1-course-qwen2.5-1.5b-grpo-bm25}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/verl_checkpoints/$TRAIN_RUN_NAME/actor/global_step_120}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-post-rl}"
EVAL_DATA="${EVAL_DATA:-$REPO_ROOT/data/course_eval/test.parquet}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-28}"
ARTIFACT_DIR="$REPO_ROOT/artifacts/course-reproduction/evaluation/$EVAL_RUN_NAME"
LOG_FILE="$ARTIFACT_DIR/evaluation.log"
MARKER="$ARTIFACT_DIR/evaluation_completed.json"

[ -s "$EVAL_DATA" ] || { echo "Missing fixed evaluation parquet: $EVAL_DATA" >&2; exit 1; }
[ -s "$MODEL_PATH/config.json" ] || { echo "Missing Hugging Face model or actor checkpoint: $MODEL_PATH" >&2; exit 1; }
mkdir -p "$ARTIFACT_DIR"
rm -f "$MARKER" "$MARKER.tmp"
python3 scripts/course_reproduction/preflight.py --repo-root "$REPO_ROOT" --require-assets
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SEARCH_ENV"
python scripts/course_reproduction/check_retriever.py

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
START_TIME="$(date +%s)"

python3 -m scripts.course_reproduction.main_ppo_with_behavior \
  data.train_files="$EVAL_DATA" data.val_files="$EVAL_DATA" \
  data.train_data_num=null data.val_data_num=null \
  data.train_batch_size="$EVAL_BATCH_SIZE" data.val_batch_size="$EVAL_BATCH_SIZE" \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=500 \
  algorithm.adv_estimator=grpo actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$EVAL_BATCH_SIZE" actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.n_agent=1 actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true algorithm.no_think_rl=false \
  "trainer.logger=['console']" +trainer.val_only=true +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.project_name=Search-R1-course trainer.experiment_name="course-eval-$EVAL_RUN_NAME" \
  max_turns=4 retriever.url="http://127.0.0.1:8000/retrieve" retriever.topk=3 \
  2>&1 | tee "$LOG_FILE"

ELAPSED="$(( $(date +%s) - START_TIME ))"
python3 scripts/course_reproduction/parse_eval_metrics.py \
  "$LOG_FILE" "$MARKER.tmp" --run-name "$EVAL_RUN_NAME" --elapsed-seconds "$ELAPSED" \
  --eval-data "$EVAL_DATA" --model-path "$MODEL_PATH"
mv "$MARKER.tmp" "$MARKER"
PRE_RL_MARKER="$REPO_ROOT/artifacts/course-reproduction/evaluation/pre-rl/evaluation_completed.json"
if [ "$EVAL_RUN_NAME" = "post-rl" ] && [ -s "$PRE_RL_MARKER" ]; then
  python3 scripts/course_reproduction/compare_evaluations.py \
    "$PRE_RL_MARKER" "$MARKER" \
    "$REPO_ROOT/artifacts/course-reproduction/evaluation/pre-post-comparison.json"
fi
echo "Evaluation completed: $MARKER"
