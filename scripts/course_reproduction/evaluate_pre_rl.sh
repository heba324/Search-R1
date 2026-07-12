#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_PATH="$REPO_ROOT/data/models/Qwen2.5-1.5B-Instruct" EVAL_RUN_NAME=pre-rl \
  bash "$SCRIPT_DIR/evaluate.sh"
