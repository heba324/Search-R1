#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
BASE_MODEL="${BASE_MODEL:-$REPO_ROOT/data/models/Qwen2.5-1.5B-Instruct}"
TOTAL_STEPS="${TOTAL_STEPS:-120}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
GROUP_SIZE="${GROUP_SIZE:-5}"
SAVE_FREQ="${SAVE_FREQ:-40}"
TEST_FREQ="${TEST_FREQ:-50}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
ENGINE_STOP_STEP="$(( TOTAL_STEPS + 1 ))"
RUN_NAME="${RUN_NAME:-search-r1-cegr-qwen2.5-1.5b-grpo-bm25}"
ARTIFACT_DIR="$REPO_ROOT/artifacts/improvement/$RUN_NAME"
LOG_FILE="$ARTIFACT_DIR/train.log"
MARKER="$ARTIFACT_DIR/training_completed.txt"

(( TRAIN_BATCH_SIZE > 0 && TOTAL_STEPS > 0 && GROUP_SIZE > 1 )) || { echo "Invalid training dimensions." >&2; exit 1; }
(( (TRAIN_BATCH_SIZE * GROUP_SIZE) % PPO_MINI_BATCH_SIZE == 0 )) || { echo "Expanded batch must be divisible by PPO mini batch." >&2; exit 1; }
(( PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE == 0 )) || { echo "PPO mini batch must be divisible by micro batch." >&2; exit 1; }

mkdir -p "$ARTIFACT_DIR"
rm -f "$MARKER" "$MARKER.tmp"
python3 scripts/course_reproduction/preflight.py --repo-root "$REPO_ROOT" --require-assets
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SEARCH_ENV"
python scripts/course_reproduction/check_retriever.py

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="${SEED:-42}"
START_TIME="$(date +%s)"

python3 -m scripts.improvement.main_ppo_cegr \
  data.train_files="$REPO_ROOT/data/nq_hotpotqa_train/train.parquet" \
  data.val_files="$REPO_ROOT/data/course_eval/test.parquet" \
  data.train_data_num=null data.val_data_num="$VAL_DATA_NUM" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=500 \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent="$GROUP_SIZE" \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  +reward_strategy.name=cegr +reward_strategy.total_steps="$TOTAL_STEPS" \
  "trainer.logger=['console','wandb']" \
  +trainer.val_only=false +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.default_hdfs_dir=null trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" trainer.test_freq="$TEST_FREQ" \
  trainer.project_name=Search-R1-course trainer.experiment_name="$RUN_NAME" \
  trainer.total_epochs=15 trainer.total_training_steps="$ENGINE_STOP_STEP" \
  trainer.default_local_dir="$REPO_ROOT/verl_checkpoints/$RUN_NAME" \
  max_turns=4 retriever.url="http://127.0.0.1:8000/retrieve" retriever.topk=3 \
  2>&1 | tee "$LOG_FILE"

python3 scripts/improvement/parse_cegr_metrics.py \
  "$LOG_FILE" "$ARTIFACT_DIR/cegr_metrics.json"
ELAPSED="$(( $(date +%s) - START_TIME ))"
printf 'status=completed\nmethod=cegr\nmodel=Qwen/Qwen2.5-1.5B-Instruct\nalgorithm=grpo\nretriever=bm25\ntraining_steps=%s\ntrain_batch_size=%s\ngroup_size=%s\nmax_turns=4\ntopk=3\nseed=%s\nelapsed_seconds=%s\n' \
  "$TOTAL_STEPS" "$TRAIN_BATCH_SIZE" "$GROUP_SIZE" "${SEED:-42}" "$ELAPSED" > "$MARKER.tmp"
mv "$MARKER.tmp" "$MARKER"
echo "CEGR training completed: $MARKER"
