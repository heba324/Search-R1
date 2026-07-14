#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FINAL_DATA="$REPO_ROOT/data/course_eval/test.parquet"
INITIAL_MODEL="$REPO_ROOT/data/models/Qwen2.5-1.5B-Instruct"
BASELINE_MODEL="$REPO_ROOT/verl_checkpoints/search-r1-course-qwen2.5-1.5b-grpo-bm25/actor/global_step_120"
DIRECT_RUN="search-r1-cegr-v2-eff-direct120-qwen2.5-1.5b-grpo-bm25"
DIRECT_MODEL="$REPO_ROOT/verl_checkpoints/$DIRECT_RUN/actor/global_step_120"
HISTORICAL_BASELINE="$REPO_ROOT/artifacts/improvement/paired-evaluation/baseline.jsonl"
PAIR_DIR="$REPO_ROOT/artifacts/improvement-v2/direct120-final-evaluation/step-120"

cd "$REPO_ROOT"
python3 scripts/improvement_v2/freeze_v1.py --repo-root "$REPO_ROOT"
python3 scripts/improvement_v2/verify_pilot_data.py
python3 scripts/improvement_v2/verify_training_run.py \
  --repo-root "$REPO_ROOT" --run-name "$DIRECT_RUN" --method eff \
  --steps 120 --group-size 5 --initial-model "$INITIAL_MODEL" \
  --train-batch-size 32 --learning-rate 1e-6 --lr-warmup-ratio 0.95

[ -s "$HISTORICAL_BASELINE" ] || {
  echo "Missing frozen V1 baseline trajectories: $HISTORICAL_BASELINE" >&2
  exit 1
}
mkdir -p "$PAIR_DIR"
python3 "$SCRIPT_DIR/rescore_frozen_baseline.py" \
  "$HISTORICAL_BASELINE" "$PAIR_DIR/historical-baseline-rescored.jsonl" \
  --report "$PAIR_DIR/historical-baseline-rescore.json" \
  --expected-per-dataset 100

evaluate_one() {
  local label="$1"
  local model="$2"
  MODEL_PATH="$model" EVAL_RUN_NAME="direct120-final-$label" \
    EVAL_DATA="$FINAL_DATA" EVAL_BATCH_SIZE=28 \
    TRAJECTORIES_PATH="$PAIR_DIR/$label.jsonl" \
    bash "$SCRIPT_DIR/evaluate_model.sh"
}

evaluate_one baseline "$BASELINE_MODEL"
evaluate_one cegr-v2-direct120 "$DIRECT_MODEL"

python3 "$SCRIPT_DIR/direct120_analysis.py" \
  "$PAIR_DIR/baseline.jsonl" "$PAIR_DIR/cegr-v2-direct120.jsonl" \
  "$PAIR_DIR/direct120-analysis.json" --expected-per-dataset 100 \
  --bootstrap-samples 10000 --seed 42
echo "Direct120 final analysis: $PAIR_DIR/direct120-analysis.json"
