#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
INDEX_PATH="${INDEX_PATH:-$REPO_ROOT/data/wiki18_bm25/bm25}"

[ -d "$INDEX_PATH" ] || { echo "Missing BM25 index: $INDEX_PATH" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_ENV"
export CUDA_VISIBLE_DEVICES=
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms2g -Xmx16g}"
cd "$REPO_ROOT"
exec python scripts/course_reproduction/bm25_server.py \
  --index-path "$INDEX_PATH" \
  --topk 3
