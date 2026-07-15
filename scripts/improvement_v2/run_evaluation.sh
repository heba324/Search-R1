#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py" --shell)"
FINAL_DATA="$REPO_ROOT/data/course_eval/test.parquet"
INITIAL_MODEL="$REPO_ROOT/$CEGR_V2_INITIAL_MODEL"
BASELINE_MODEL="$REPO_ROOT/verl_checkpoints/$CEGR_V2_BASELINE_RUN/actor/global_step_$CEGR_V2_BASELINE_STEP"
TRAINING_RUN="$CEGR_V2_TRAINING_RUN"
V2_MODEL="$REPO_ROOT/verl_checkpoints/$TRAINING_RUN/actor/global_step_$CEGR_V2_TRAINING_STEPS"
HISTORICAL_BASELINE="$REPO_ROOT/artifacts/improvement/paired-evaluation/baseline.jsonl"
PAIR_DIR="$REPO_ROOT/artifacts/improvement-v2/direct120-final-evaluation/step-$CEGR_V2_TRAINING_STEPS"

cd "$REPO_ROOT"
python3 -m scripts.improvement_v2.freeze_v1 --repo-root "$REPO_ROOT"
python3 -m scripts.improvement_v2.verify_pilot_data
python3 -m scripts.improvement_v2.verify_training \
  --repo-root "$REPO_ROOT" --run-name "$TRAINING_RUN" \
  --method "$CEGR_V2_REWARD_MODE" --steps "$CEGR_V2_TRAINING_STEPS" \
  --group-size "$CEGR_V2_GROUP_SIZE" --seed "$CEGR_V2_SEED" \
  --rollout-engine-seed "$CEGR_V2_ROLLOUT_ENGINE_SEED" \
  --initial-model "$INITIAL_MODEL" \
  --train-batch-size "$CEGR_V2_TRAIN_BATCH_SIZE" \
  --learning-rate "$CEGR_V2_LEARNING_RATE" \
  --lr-warmup-ratio "$CEGR_V2_LR_WARMUP_RATIO"

[ -s "$HISTORICAL_BASELINE" ] || {
  echo "Missing frozen V1 baseline trajectories: $HISTORICAL_BASELINE" >&2
  exit 1
}
mkdir -p "$PAIR_DIR"
python3 -m scripts.improvement_v2.rescore_baseline \
  "$HISTORICAL_BASELINE" "$PAIR_DIR/historical-baseline-rescored.jsonl" \
  --report "$PAIR_DIR/historical-baseline-rescore.json" \
  --expected-per-dataset "$CEGR_V2_FINAL_PER_DATASET"

evaluate_one() {
  local label="$1"
  local model="$2"
  SEED="$CEGR_V2_SEED" \
    ROLLOUT_ENGINE_SEED="$CEGR_V2_ROLLOUT_ENGINE_SEED" \
    MODEL_PATH="$model" EVAL_RUN_NAME="direct120-final-$label" \
    EVAL_DATA="$FINAL_DATA" EVAL_BATCH_SIZE="$CEGR_V2_EVAL_BATCH_SIZE" \
    TRAJECTORIES_PATH="$PAIR_DIR/$label.jsonl" \
    bash "$SCRIPT_DIR/evaluate_model.sh"
}

evaluate_one baseline "$BASELINE_MODEL"
evaluate_one cegr-v2-direct120 "$V2_MODEL"

python3 -m scripts.improvement_v2.analysis \
  "$PAIR_DIR/baseline.jsonl" "$PAIR_DIR/cegr-v2-direct120.jsonl" \
  "$PAIR_DIR/direct120-analysis.json" \
  --expected-per-dataset "$CEGR_V2_FINAL_PER_DATASET" \
  --bootstrap-samples "$CEGR_V2_BOOTSTRAP_SAMPLES" \
  --seed "$CEGR_V2_SEED"
echo "CEGR V2 final analysis: $PAIR_DIR/direct120-analysis.json"
