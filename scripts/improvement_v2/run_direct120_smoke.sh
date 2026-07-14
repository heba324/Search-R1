#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_MODEL="$REPO_ROOT/data/models/Qwen2.5-1.5B-Instruct"
RUN_NAME="search-r1-cegr-v2-eff-direct120-smoke"
ARTIFACT_DIR="$REPO_ROOT/artifacts/improvement-v2/$RUN_NAME"
CHECKPOINT_DIR="$REPO_ROOT/verl_checkpoints/$RUN_NAME"

cd "$REPO_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SEARCH_ENV:-Search-R1}"
python scripts/improvement_v2/verify_pilot_data.py

if [ -s "$ARTIFACT_DIR/training_completed.txt" ] && \
   [ -s "$CHECKPOINT_DIR/actor/global_step_2/config.json" ]; then
  python scripts/improvement_v2/verify_training_run.py \
    --repo-root "$REPO_ROOT" --run-name "$RUN_NAME" --method eff \
    --steps 2 --group-size 5 --minimum-signal 0.10 \
    --initial-model "$BASE_MODEL" --train-batch-size 8 \
    --learning-rate 1e-6 --lr-warmup-ratio 0.95
  echo "Already completed; preserving direct120 smoke."
  exit 0
fi
if [ -e "$ARTIFACT_DIR" ] || [ -e "$CHECKPOINT_DIR" ]; then
  echo "Partial direct120 smoke found; inspect it instead of overwriting." >&2
  exit 1
fi

BASE_MODEL="$BASE_MODEL" MODE=eff TOTAL_STEPS=2 LR_WARMUP_STEPS_RATIO=0.95 \
  TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 LEARNING_RATE=1e-6 \
  VAL_DATA="$REPO_ROOT/data/improvement_v2/pilot.parquet" \
  VAL_DATA_NUM=20 VAL_BATCH_SIZE=20 MIN_INFORMATIVE_FALLBACK_RATE=0.10 \
  SAVE_FREQ=2 TEST_FREQ=0 RUN_NAME="$RUN_NAME" \
  bash "$SCRIPT_DIR/train_refinement.sh"

python scripts/improvement_v2/verify_training_run.py \
  --repo-root "$REPO_ROOT" --run-name "$RUN_NAME" --method eff \
  --steps 2 --group-size 5 --minimum-signal 0.10 \
  --initial-model "$BASE_MODEL" --train-batch-size 8 \
  --learning-rate 1e-6 --lr-warmup-ratio 0.95
