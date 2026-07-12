#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/data/models}"
MODEL_REPO="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION="989aa79"
MODEL_DIR="$MODEL_ROOT/Qwen2.5-1.5B-Instruct"
export HF_HOME="${HF_HOME:-$REPO_ROOT/data/huggingface}"

mkdir -p "$MODEL_DIR" "$HF_HOME"
huggingface-cli download --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR" "$MODEL_REPO"
[ -s "$MODEL_DIR/config.json" ] || { echo "Qwen model snapshot is incomplete." >&2; exit 1; }
printf 'repo=%s\nrevision=%s\n' "$MODEL_REPO" "$MODEL_REVISION" > "$MODEL_DIR/course-model-revision.txt"
