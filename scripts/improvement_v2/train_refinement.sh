#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
MODE="${MODE:-eff}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-search-r1-course-qwen2.5-1.5b-grpo-bm25}"
BASE_MODEL="${BASE_MODEL:-$REPO_ROOT/verl_checkpoints/$BASELINE_RUN_NAME/actor/global_step_120}"
TOTAL_STEPS="${TOTAL_STEPS:-40}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-20}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
GROUP_SIZE="${GROUP_SIZE:-5}"
SEED="${SEED:-42}"
ROLLOUT_ENGINE_SEED="${ROLLOUT_ENGINE_SEED:-42}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.0}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_DATA="${VAL_DATA:-$REPO_ROOT/data/improvement_v2/pilot.parquet}"
VAL_DATA_NUM="${VAL_DATA_NUM:-null}"
MIN_INFORMATIVE_FALLBACK_RATE="${MIN_INFORMATIVE_FALLBACK_RATE:-0.0}"
ENGINE_STOP_STEP="$(( TOTAL_STEPS + 1 ))"

case "$MODE" in
  eff)
    DEFAULT_RUN_NAME="search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25"
    ;;
  grouped_em)
    DEFAULT_RUN_NAME="search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25"
    ;;
  *)
    echo "MODE must be eff or grouped_em, found: $MODE" >&2
    exit 1
    ;;
esac

RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
ARTIFACT_DIR="$REPO_ROOT/artifacts/improvement-v2/$RUN_NAME"
CHECKPOINT_DIR="$REPO_ROOT/verl_checkpoints/$RUN_NAME"
LOG_FILE="$ARTIFACT_DIR/train.log"
MARKER="$ARTIFACT_DIR/training_completed.txt"

(( TRAIN_BATCH_SIZE > 0 && TOTAL_STEPS > 0 && GROUP_SIZE > 1 )) || { echo "Invalid training dimensions." >&2; exit 1; }
(( (TRAIN_BATCH_SIZE * GROUP_SIZE) % PPO_MINI_BATCH_SIZE == 0 )) || { echo "Expanded batch must be divisible by PPO mini batch." >&2; exit 1; }
(( PPO_MINI_BATCH_SIZE % PPO_MICRO_BATCH_SIZE == 0 )) || { echo "PPO mini batch must be divisible by micro batch." >&2; exit 1; }
[ -s "$BASE_MODEL/config.json" ] || { echo "Missing frozen baseline checkpoint: $BASE_MODEL" >&2; exit 1; }
[ -s "$VAL_DATA" ] || { echo "Missing validation data: $VAL_DATA" >&2; exit 1; }
[ ! -e "$CHECKPOINT_DIR" ] || { echo "Refusing to overwrite V2 checkpoint directory: $CHECKPOINT_DIR" >&2; exit 1; }
[ ! -e "$ARTIFACT_DIR" ] || { echo "Refusing to overwrite V2 artifact directory: $ARTIFACT_DIR" >&2; exit 1; }

mkdir -p "$ARTIFACT_DIR"
python3 scripts/improvement_v2/freeze_v1.py --repo-root "$REPO_ROOT"
python3 scripts/course_reproduction/preflight.py --repo-root "$REPO_ROOT" --require-assets
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SEARCH_ENV"
python scripts/course_reproduction/check_retriever.py

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"
START_TIME="$(date +%s)"

python3 -m scripts.improvement_v2.main_ppo_refinement \
  data.train_files="$REPO_ROOT/data/nq_hotpotqa_train/train.parquet" \
  data.val_files="$VAL_DATA" data.train_data_num=null data.val_data_num="$VAL_DATA_NUM" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=500 data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  +actor_rollout_ref.rollout.engine_seed="$ROLLOUT_ENGINE_SEED" \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.n_agent="$GROUP_SIZE" \
  actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.actor.state_masking=true \
  +reward_strategy.name="$MODE" +reward_strategy.group_size="$GROUP_SIZE" \
  +reward_strategy.seed="$SEED" \
  "trainer.logger=['console','wandb']" +trainer.val_only=false \
  +trainer.val_before_train="$VAL_BEFORE_TRAIN" trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" trainer.project_name=Search-R1-course \
  trainer.experiment_name="$RUN_NAME" trainer.total_epochs=15 \
  trainer.total_training_steps="$ENGINE_STOP_STEP" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  max_turns=4 retriever.url="http://127.0.0.1:8000/retrieve" retriever.topk=3 \
  2>&1 | tee "$LOG_FILE"

python3 scripts/improvement_v2/parse_v2_metrics.py \
  "$LOG_FILE" "$ARTIFACT_DIR/reward_metrics.json" \
  --minimum-informative-fallback-rate "$MIN_INFORMATIVE_FALLBACK_RATE" \
  --expected-steps "$TOTAL_STEPS" --expected-group-size "$GROUP_SIZE"
ELAPSED="$(( $(date +%s) - START_TIME ))"
printf 'status=completed\nmethod=%s\ninitial_checkpoint=%s\ntraining_steps=%s\ntrain_batch_size=%s\ngroup_size=%s\nlearning_rate=%s\nlr_warmup_steps_ratio=%s\nseed=%s\nrollout_engine_seed=%s\nelapsed_seconds=%s\n' \
  "$MODE" "$BASE_MODEL" "$TOTAL_STEPS" "$TRAIN_BATCH_SIZE" "$GROUP_SIZE" \
  "$LEARNING_RATE" "$LR_WARMUP_STEPS_RATIO" "$SEED" \
  "$ROLLOUT_ENGINE_SEED" "$ELAPSED" > "$MARKER.tmp"
mv "$MARKER.tmp" "$MARKER"
if ! python3 scripts/improvement_v2/verify_training_run.py \
  --repo-root "$REPO_ROOT" --run-name "$RUN_NAME" --method "$MODE" \
  --steps "$TOTAL_STEPS" --group-size "$GROUP_SIZE" \
  --minimum-signal "$MIN_INFORMATIVE_FALLBACK_RATE" \
  --rollout-engine-seed "$ROLLOUT_ENGINE_SEED"; then
  rm -f "$MARKER"
  exit 1
fi
echo "CEGR V2 refinement completed: $MARKER"
