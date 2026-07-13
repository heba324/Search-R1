#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_NAME="${RUN_NAME:-search-r1-cegr-qwen2.5-1.5b-grpo-bm25}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/verl_checkpoints/$RUN_NAME/actor/global_step_120}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-search-r1-course-qwen2.5-1.5b-grpo-bm25}"
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-$REPO_ROOT/verl_checkpoints/$BASELINE_RUN_NAME/actor/global_step_120}"
PAIRED_DIR="$REPO_ROOT/artifacts/improvement/paired-evaluation"
BASELINE_TRAJECTORIES="$PAIRED_DIR/baseline.jsonl"
CEGR_TRAJECTORIES="$PAIRED_DIR/cegr.jsonl"
CEGR_MARKER="$REPO_ROOT/artifacts/course-reproduction/evaluation/cegr-post-rl/evaluation_completed.json"
BASELINE_MARKER="$REPO_ROOT/artifacts/course-reproduction/evaluation/baseline-paired/evaluation_completed.json"
COMPARISON="$REPO_ROOT/artifacts/improvement/baseline-vs-cegr.json"
PAIRED_ANALYSIS="$REPO_ROOT/artifacts/improvement/paired-statistical-analysis.json"

mkdir -p "$PAIRED_DIR"
SEARCH_R1_EVAL_TRAJECTORIES="$BASELINE_TRAJECTORIES" \
  MODEL_PATH="$BASELINE_MODEL_PATH" EVAL_RUN_NAME=baseline-paired \
  bash "$REPO_ROOT/scripts/course_reproduction/evaluate.sh"

SEARCH_R1_EVAL_TRAJECTORIES="$CEGR_TRAJECTORIES" \
  MODEL_PATH="$MODEL_PATH" EVAL_RUN_NAME=cegr-post-rl \
  bash "$REPO_ROOT/scripts/course_reproduction/evaluate.sh"

[ -s "$BASELINE_MARKER" ] || { echo "Missing baseline post-RL evaluation: $BASELINE_MARKER" >&2; exit 1; }
python3 "$REPO_ROOT/scripts/course_reproduction/compare_evaluations.py" \
  "$BASELINE_MARKER" "$CEGR_MARKER" "$COMPARISON"
python3 "$REPO_ROOT/scripts/improvement/analyze_paired_results.py" \
  "$BASELINE_TRAJECTORIES" "$CEGR_TRAJECTORIES" "$PAIRED_ANALYSIS" \
  --bootstrap-samples 10000 --seed 42
echo "Baseline-vs-CEGR comparison: $COMPARISON"
echo "Paired statistical analysis: $PAIRED_ANALYSIS"
