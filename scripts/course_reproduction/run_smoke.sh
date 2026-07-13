#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME=course-smoke TOTAL_STEPS=2 TRAIN_BATCH_SIZE=8 VAL_BATCH_SIZE=7 \
  PPO_MINI_BATCH_SIZE=8 PPO_MICRO_BATCH_SIZE=1 SAVE_FREQ=1 TEST_FREQ=-1 VAL_BEFORE_TRAIN=false VAL_DATA_NUM=7 \
  bash "$SCRIPT_DIR/train_grpo.sh"
