#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SEARCH_ENV:-Search-R1}"
python -m scripts.improvement_v2.prepare_pilot_data
python -m scripts.improvement_v2.verify_pilot_data
python -m scripts.improvement_v2.verify_training_run \
  --repo-root "$REPO_ROOT" --run-name search-r1-cegr-v2-smoke \
  --method eff --steps 2 --group-size 5 --minimum-signal 0.10

run_arm() {
  local mode="$1"
  local run_name="$2"
  local marker="$REPO_ROOT/artifacts/improvement-v2/$run_name/training_completed.txt"
  local checkpoint="$REPO_ROOT/verl_checkpoints/$run_name/actor/global_step_40/config.json"
  if [ -s "$marker" ] && [ -s "$checkpoint" ]; then
    python -m scripts.improvement_v2.verify_training_run \
      --repo-root "$REPO_ROOT" --run-name "$run_name" \
      --method "$mode" --steps 40 --group-size 5
    echo "Already completed; preserving arm: $run_name"
    return
  fi
  if [ -e "$REPO_ROOT/verl_checkpoints/$run_name" ] || \
     [ -e "$REPO_ROOT/artifacts/improvement-v2/$run_name" ]; then
    echo "Partial arm found; inspect it instead of overwriting: $run_name" >&2
    exit 1
  fi
  MODE="$mode" TOTAL_STEPS=40 RUN_NAME="$run_name" \
    bash "$SCRIPT_DIR/train_refinement.sh"
  python -m scripts.improvement_v2.verify_training_run \
    --repo-root "$REPO_ROOT" --run-name "$run_name" \
    --method "$mode" --steps 40 --group-size 5
}

run_arm grouped_em search-r1-cegr-v2-em-control-qwen2.5-1.5b-grpo-bm25
run_arm eff search-r1-cegr-v2-qwen2.5-1.5b-grpo-bm25
