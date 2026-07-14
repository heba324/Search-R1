#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
python3 -m scripts.improvement_v2.verify_pilot_data

MODE=eff TOTAL_STEPS=2 TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 \
  VAL_DATA="$REPO_ROOT/data/improvement_v2/pilot.parquet" \
  VAL_DATA_NUM=20 VAL_BATCH_SIZE=20 MIN_INFORMATIVE_FALLBACK_RATE=0.10 \
  SAVE_FREQ=2 TEST_FREQ=0 RUN_NAME=search-r1-cegr-v2-smoke \
  bash "$SCRIPT_DIR/train_refinement.sh"

python3 -m scripts.improvement_v2.verify_training_run \
  --repo-root "$REPO_ROOT" --run-name search-r1-cegr-v2-smoke \
  --method eff --steps 2 --group-size 5 --minimum-signal 0.10
