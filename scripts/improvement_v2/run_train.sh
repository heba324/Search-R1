#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py" --shell)"
BASE_MODEL="$REPO_ROOT/$CEGR_V2_INITIAL_MODEL"
SMOKE_RUN="$CEGR_V2_SMOKE_RUN"
RUN_NAME="$CEGR_V2_TRAINING_RUN"
ARTIFACT_DIR="$REPO_ROOT/artifacts/improvement-v2/$RUN_NAME"
CHECKPOINT_DIR="$REPO_ROOT/verl_checkpoints/$RUN_NAME"

cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SEARCH_ENV:-Search-R1}"
python -m scripts.improvement_v2.verify_pilot_data
python -m scripts.improvement_v2.verify_training \
  --repo-root "$REPO_ROOT" --run-name "$SMOKE_RUN" \
  --method "$CEGR_V2_REWARD_MODE" --steps "$CEGR_V2_SMOKE_STEPS" \
  --group-size "$CEGR_V2_GROUP_SIZE" \
  --seed "$CEGR_V2_SEED" \
  --rollout-engine-seed "$CEGR_V2_ROLLOUT_ENGINE_SEED" \
  --minimum-signal "$CEGR_V2_MINIMUM_SMOKE_SIGNAL" \
  --initial-model "$BASE_MODEL" \
  --train-batch-size "$CEGR_V2_SMOKE_BATCH_SIZE" \
  --learning-rate "$CEGR_V2_LEARNING_RATE" \
  --lr-warmup-ratio "$CEGR_V2_LR_WARMUP_RATIO"

if [ -s "$ARTIFACT_DIR/training_completed.txt" ] && \
   [ -s "$CHECKPOINT_DIR/actor/global_step_$CEGR_V2_TRAINING_STEPS/config.json" ]; then
  python -m scripts.improvement_v2.verify_training \
    --repo-root "$REPO_ROOT" --run-name "$RUN_NAME" \
    --method "$CEGR_V2_REWARD_MODE" --steps "$CEGR_V2_TRAINING_STEPS" \
    --group-size "$CEGR_V2_GROUP_SIZE" --seed "$CEGR_V2_SEED" \
    --rollout-engine-seed "$CEGR_V2_ROLLOUT_ENGINE_SEED" \
    --initial-model "$BASE_MODEL" \
    --train-batch-size "$CEGR_V2_TRAIN_BATCH_SIZE" \
    --learning-rate "$CEGR_V2_LEARNING_RATE" \
    --lr-warmup-ratio "$CEGR_V2_LR_WARMUP_RATIO"
  echo "Already completed; preserving CEGR V2 training."
  exit 0
fi
if [ -e "$ARTIFACT_DIR" ] || [ -e "$CHECKPOINT_DIR" ]; then
  echo "Partial CEGR V2 training run found; inspect it instead of overwriting." >&2
  exit 1
fi

BASE_MODEL="$BASE_MODEL" MODE="$CEGR_V2_REWARD_MODE" \
  TOTAL_STEPS="$CEGR_V2_TRAINING_STEPS" \
  GROUP_SIZE="$CEGR_V2_GROUP_SIZE" SEED="$CEGR_V2_SEED" \
  ROLLOUT_ENGINE_SEED="$CEGR_V2_ROLLOUT_ENGINE_SEED" \
  LEARNING_RATE="$CEGR_V2_LEARNING_RATE" \
  LR_WARMUP_STEPS_RATIO="$CEGR_V2_LR_WARMUP_RATIO" \
  TRAIN_BATCH_SIZE="$CEGR_V2_TRAIN_BATCH_SIZE" \
  PPO_MINI_BATCH_SIZE="$CEGR_V2_TRAIN_BATCH_SIZE" \
  SAVE_FREQ="$CEGR_V2_SAVE_FREQ" TEST_FREQ=0 \
  RUN_NAME="$RUN_NAME" bash "$SCRIPT_DIR/train.sh"

python -m scripts.improvement_v2.verify_training \
  --repo-root "$REPO_ROOT" --run-name "$RUN_NAME" \
  --method "$CEGR_V2_REWARD_MODE" --steps "$CEGR_V2_TRAINING_STEPS" \
  --group-size "$CEGR_V2_GROUP_SIZE" --seed "$CEGR_V2_SEED" \
  --rollout-engine-seed "$CEGR_V2_ROLLOUT_ENGINE_SEED" \
  --initial-model "$BASE_MODEL" \
  --train-batch-size "$CEGR_V2_TRAIN_BATCH_SIZE" \
  --learning-rate "$CEGR_V2_LEARNING_RATE" \
  --lr-warmup-ratio "$CEGR_V2_LR_WARMUP_RATIO"
