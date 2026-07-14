#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
eval "$(python3 "$SCRIPT_DIR/direct120_contract.py" --shell)"
FINAL_DATA="$REPO_ROOT/data/course_eval/test.parquet"
INITIAL_MODEL="$REPO_ROOT/$DIRECT120_INITIAL_MODEL"
BASELINE_MODEL="$REPO_ROOT/verl_checkpoints/$DIRECT120_BASELINE_RUN/actor/global_step_$DIRECT120_BASELINE_STEP"
DIRECT_RUN="$DIRECT120_TRAINING_RUN"
DIRECT_MODEL="$REPO_ROOT/verl_checkpoints/$DIRECT_RUN/actor/global_step_$DIRECT120_TRAINING_STEPS"
HISTORICAL_BASELINE="$REPO_ROOT/artifacts/improvement/paired-evaluation/baseline.jsonl"
PAIR_DIR="$REPO_ROOT/artifacts/improvement-v2/direct120-final-evaluation/step-$DIRECT120_TRAINING_STEPS"

cd "$REPO_ROOT"
python3 -m scripts.improvement_v2.freeze_v1 --repo-root "$REPO_ROOT"
python3 -m scripts.improvement_v2.verify_pilot_data
python3 -m scripts.improvement_v2.verify_training_run \
  --repo-root "$REPO_ROOT" --run-name "$DIRECT_RUN" \
  --method "$DIRECT120_REWARD_MODE" --steps "$DIRECT120_TRAINING_STEPS" \
  --group-size "$DIRECT120_GROUP_SIZE" --seed "$DIRECT120_SEED" \
  --rollout-engine-seed "$DIRECT120_ROLLOUT_ENGINE_SEED" \
  --initial-model "$INITIAL_MODEL" \
  --train-batch-size "$DIRECT120_TRAIN_BATCH_SIZE" \
  --learning-rate "$DIRECT120_LEARNING_RATE" \
  --lr-warmup-ratio "$DIRECT120_LR_WARMUP_RATIO"

[ -s "$HISTORICAL_BASELINE" ] || {
  echo "Missing frozen V1 baseline trajectories: $HISTORICAL_BASELINE" >&2
  exit 1
}
mkdir -p "$PAIR_DIR"
python3 -m scripts.improvement_v2.rescore_frozen_baseline \
  "$HISTORICAL_BASELINE" "$PAIR_DIR/historical-baseline-rescored.jsonl" \
  --report "$PAIR_DIR/historical-baseline-rescore.json" \
  --expected-per-dataset "$DIRECT120_FINAL_PER_DATASET"

evaluate_one() {
  local label="$1"
  local model="$2"
  SEED="$DIRECT120_SEED" \
    ROLLOUT_ENGINE_SEED="$DIRECT120_ROLLOUT_ENGINE_SEED" \
    MODEL_PATH="$model" EVAL_RUN_NAME="direct120-final-$label" \
    EVAL_DATA="$FINAL_DATA" EVAL_BATCH_SIZE="$DIRECT120_EVAL_BATCH_SIZE" \
    TRAJECTORIES_PATH="$PAIR_DIR/$label.jsonl" \
    bash "$SCRIPT_DIR/evaluate_model.sh"
}

evaluate_one baseline "$BASELINE_MODEL"
evaluate_one cegr-v2-direct120 "$DIRECT_MODEL"

python3 -m scripts.improvement_v2.direct120_analysis \
  "$PAIR_DIR/baseline.jsonl" "$PAIR_DIR/cegr-v2-direct120.jsonl" \
  "$PAIR_DIR/direct120-analysis.json" \
  --expected-per-dataset "$DIRECT120_FINAL_PER_DATASET" \
  --bootstrap-samples "$DIRECT120_BOOTSTRAP_SAMPLES" \
  --seed "$DIRECT120_SEED"
echo "Direct120 final analysis: $PAIR_DIR/direct120-analysis.json"
