#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export HF_HOME="${HF_HOME:-$REPO_ROOT/data/huggingface}"
exec bash "$SCRIPT_DIR/../paper_v1/prepare_train_data.sh"
