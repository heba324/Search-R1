#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-40}"
PILOT_DATA="${PILOT_DATA:-$REPO_ROOT/data/improvement_v2/pilot.parquet}"
BASELINE_MODEL="$REPO_ROOT/verl_checkpoints/search-r1-course-qwen2.5-1.5b-grpo-bm25/actor/global_step_120"
CONTROL_MODEL="$REPO_ROOT/verl_checkpoints/search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25/actor/global_step_$CHECKPOINT_STEP"
V2_MODEL="$REPO_ROOT/verl_checkpoints/search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25/actor/global_step_$CHECKPOINT_STEP"
PAIR_DIR="$REPO_ROOT/artifacts/improvement-v2/pilot-evaluation/step-$CHECKPOINT_STEP"

cd "$REPO_ROOT"
python3 scripts/improvement_v2/freeze_v1.py --repo-root "$REPO_ROOT"
python3 scripts/improvement_v2/verify_pilot_data.py
python3 scripts/improvement_v2/verify_training_run.py \
  --repo-root "$REPO_ROOT" \
  --run-name search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25 \
  --method grouped_em --steps 40 --group-size 5
python3 scripts/improvement_v2/verify_training_run.py \
  --repo-root "$REPO_ROOT" \
  --run-name search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25 \
  --method eff --steps 40 --group-size 5

evaluate_one() {
  local label="$1"
  local model="$2"
  MODEL_PATH="$model" EVAL_RUN_NAME="pilot-step-$CHECKPOINT_STEP-$label" \
    EVAL_DATA="$PILOT_DATA" EVAL_BATCH_SIZE=20 \
    TRAJECTORIES_PATH="$PAIR_DIR/$label.jsonl" \
    bash "$SCRIPT_DIR/evaluate_model.sh"
}

evaluate_one baseline "$BASELINE_MODEL"
evaluate_one em-control "$CONTROL_MODEL"
evaluate_one cegr-v2 "$V2_MODEL"

python3 "$SCRIPT_DIR/pilot_gate.py" \
  "$PAIR_DIR/baseline.jsonl" "$PAIR_DIR/em-control.jsonl" "$PAIR_DIR/cegr-v2.jsonl" \
  "$PAIR_DIR/pilot-gate.json" --expected-per-dataset 20
python3 "$SCRIPT_DIR/verify_pilot_gate.py" \
  "$PAIR_DIR/pilot-gate.json" "$PAIR_DIR/baseline.jsonl" \
  "$PAIR_DIR/em-control.jsonl" "$PAIR_DIR/cegr-v2.jsonl" \
  --expected-per-dataset 20
echo "Pilot passed: $PAIR_DIR/pilot-gate.json"
