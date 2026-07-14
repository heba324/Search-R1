#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
eval "$(python3 "$SCRIPT_DIR/direct120_contract.py" --shell)"
BASE_MODEL="$REPO_ROOT/$DIRECT120_INITIAL_MODEL"
RUN_NAME="$DIRECT120_SMOKE_RUN"
ARTIFACT_DIR="$REPO_ROOT/artifacts/improvement-v2/$RUN_NAME"
CHECKPOINT_DIR="$REPO_ROOT/verl_checkpoints/$RUN_NAME"

cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SEARCH_ENV:-Search-R1}"
python -m scripts.improvement_v2.verify_pilot_data

if [ -s "$ARTIFACT_DIR/training_completed.txt" ] && \
   [ -s "$CHECKPOINT_DIR/actor/global_step_$DIRECT120_SMOKE_STEPS/config.json" ]; then
  python -m scripts.improvement_v2.verify_training_run \
    --repo-root "$REPO_ROOT" --run-name "$RUN_NAME" \
    --method "$DIRECT120_REWARD_MODE" --steps "$DIRECT120_SMOKE_STEPS" \
    --group-size "$DIRECT120_GROUP_SIZE" \
    --seed "$DIRECT120_SEED" \
    --rollout-engine-seed "$DIRECT120_ROLLOUT_ENGINE_SEED" \
    --minimum-signal "$DIRECT120_MINIMUM_SMOKE_SIGNAL" \
    --initial-model "$BASE_MODEL" \
    --train-batch-size "$DIRECT120_SMOKE_BATCH_SIZE" \
    --learning-rate "$DIRECT120_LEARNING_RATE" \
    --lr-warmup-ratio "$DIRECT120_LR_WARMUP_RATIO"
  echo "Already completed; preserving direct120 smoke."
  exit 0
fi
if [ -e "$ARTIFACT_DIR" ] || [ -e "$CHECKPOINT_DIR" ]; then
  echo "Partial direct120 smoke found; inspect it instead of overwriting." >&2
  exit 1
fi

BASE_MODEL="$BASE_MODEL" MODE="$DIRECT120_REWARD_MODE" \
  TOTAL_STEPS="$DIRECT120_SMOKE_STEPS" \
  GROUP_SIZE="$DIRECT120_GROUP_SIZE" SEED="$DIRECT120_SEED" \
  ROLLOUT_ENGINE_SEED="$DIRECT120_ROLLOUT_ENGINE_SEED" \
  TRAIN_BATCH_SIZE="$DIRECT120_SMOKE_BATCH_SIZE" \
  PPO_MINI_BATCH_SIZE="$DIRECT120_SMOKE_BATCH_SIZE" \
  LEARNING_RATE="$DIRECT120_LEARNING_RATE" \
  LR_WARMUP_STEPS_RATIO="$DIRECT120_LR_WARMUP_RATIO" \
  VAL_DATA="$REPO_ROOT/data/improvement_v2/pilot.parquet" \
  VAL_DATA_NUM=20 VAL_BATCH_SIZE=20 \
  MIN_INFORMATIVE_FALLBACK_RATE="$DIRECT120_MINIMUM_SMOKE_SIGNAL" \
  SAVE_FREQ="$DIRECT120_SMOKE_STEPS" TEST_FREQ=0 RUN_NAME="$RUN_NAME" \
  bash "$SCRIPT_DIR/train_refinement.sh"

python -m scripts.improvement_v2.verify_training_run \
  --repo-root "$REPO_ROOT" --run-name "$RUN_NAME" \
  --method "$DIRECT120_REWARD_MODE" --steps "$DIRECT120_SMOKE_STEPS" \
  --group-size "$DIRECT120_GROUP_SIZE" \
  --seed "$DIRECT120_SEED" \
  --rollout-engine-seed "$DIRECT120_ROLLOUT_ENGINE_SEED" \
  --minimum-signal "$DIRECT120_MINIMUM_SMOKE_SIGNAL" \
  --initial-model "$BASE_MODEL" \
  --train-batch-size "$DIRECT120_SMOKE_BATCH_SIZE" \
  --learning-rate "$DIRECT120_LEARNING_RATE" \
  --lr-warmup-ratio "$DIRECT120_LR_WARMUP_RATIO"
