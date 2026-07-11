#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-$REPO_ROOT/data/models}"
QWEN_REPO="Qwen/Qwen2.5-3B-Instruct"
QWEN_REVISION="aa8e725"
E5_REPO="intfloat/e5-base-v2"
E5_REVISION="f52bf8e"

mkdir -p "$MODEL_ROOT/Qwen2.5-3B-Instruct" "$MODEL_ROOT/e5-base-v2"
huggingface-cli download --revision "$QWEN_REVISION" --local-dir "$MODEL_ROOT/Qwen2.5-3B-Instruct" "$QWEN_REPO"
huggingface-cli download --revision "$E5_REVISION" --local-dir "$MODEL_ROOT/e5-base-v2" "$E5_REPO"
[ -s "$MODEL_ROOT/Qwen2.5-3B-Instruct/config.json" ] || { echo "Qwen snapshot is incomplete." >&2; exit 1; }
[ -s "$MODEL_ROOT/e5-base-v2/config.json" ] || { echo "E5 snapshot is incomplete." >&2; exit 1; }
printf 'Qwen repo=%s revision=%s\nE5 repo=%s revision=%s\n' \
  "$QWEN_REPO" "$QWEN_REVISION" "$E5_REPO" "$E5_REVISION" > "$MODEL_ROOT/paper-v1-model-revisions.txt"
