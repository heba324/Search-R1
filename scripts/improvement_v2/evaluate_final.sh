#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-40}"
FINAL_DATA="$REPO_ROOT/data/course_eval/test.parquet"
HISTORICAL_BASELINE="$REPO_ROOT/artifacts/improvement/paired-evaluation/baseline.jsonl"
BASELINE_MODEL="$REPO_ROOT/verl_checkpoints/search-r1-course-qwen2.5-1.5b-grpo-bm25/actor/global_step_120"
CONTROL_MODEL="$REPO_ROOT/verl_checkpoints/search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25/actor/global_step_$CHECKPOINT_STEP"
V2_MODEL="$REPO_ROOT/verl_checkpoints/search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25/actor/global_step_$CHECKPOINT_STEP"
PILOT_GATE="$REPO_ROOT/artifacts/improvement-v2/pilot-evaluation/step-$CHECKPOINT_STEP/pilot-gate.json"
PAIR_DIR="$REPO_ROOT/artifacts/improvement-v2/final-evaluation/step-$CHECKPOINT_STEP"

cd "$REPO_ROOT"
python3 scripts/improvement_v2/freeze_v1.py --repo-root "$REPO_ROOT"
python3 scripts/improvement_v2/verify_pilot_data.py
python3 "$SCRIPT_DIR/verify_pilot_gate.py" \
  "$PILOT_GATE" \
  "$REPO_ROOT/artifacts/improvement-v2/pilot-evaluation/step-$CHECKPOINT_STEP/baseline.jsonl" \
  "$REPO_ROOT/artifacts/improvement-v2/pilot-evaluation/step-$CHECKPOINT_STEP/em-control.jsonl" \
  "$REPO_ROOT/artifacts/improvement-v2/pilot-evaluation/step-$CHECKPOINT_STEP/cegr-v2.jsonl" \
  --expected-per-dataset 20

python3 - "$PILOT_GATE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file() or not json.loads(path.read_text(encoding="utf-8"))["passed"]:
    raise SystemExit("A passing disjoint pilot gate is required before final evaluation")
PY

[ -s "$HISTORICAL_BASELINE" ] || { echo "Missing frozen V1 baseline records: $HISTORICAL_BASELINE" >&2; exit 1; }
mkdir -p "$PAIR_DIR"
python3 "$SCRIPT_DIR/rescore_frozen_baseline.py" \
  "$HISTORICAL_BASELINE" "$PAIR_DIR/historical-baseline-rescored.jsonl" \
  --report "$PAIR_DIR/baseline-rescore.json" --expected-per-dataset 100

evaluate_one() {
  local label="$1"
  local model="$2"
  MODEL_PATH="$model" EVAL_RUN_NAME="final-step-$CHECKPOINT_STEP-$label" \
    EVAL_DATA="$FINAL_DATA" EVAL_BATCH_SIZE=28 \
    TRAJECTORIES_PATH="$PAIR_DIR/$label.jsonl" \
    bash "$SCRIPT_DIR/evaluate_model.sh"
}

evaluate_one baseline "$BASELINE_MODEL"
evaluate_one em-control "$CONTROL_MODEL"
evaluate_one cegr-v2 "$V2_MODEL"

python3 "$SCRIPT_DIR/final_analysis.py" \
  "$PAIR_DIR/baseline.jsonl" "$PAIR_DIR/em-control.jsonl" "$PAIR_DIR/cegr-v2.jsonl" \
  "$PAIR_DIR/final-analysis.json" --expected-per-dataset 100 \
  --bootstrap-samples 10000 --seed 42 --pilot-gate "$PILOT_GATE"
echo "Final CEGR V2 analysis: $PAIR_DIR/final-analysis.json"
