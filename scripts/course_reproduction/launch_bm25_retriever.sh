#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
INDEX_PATH="${INDEX_PATH:-$REPO_ROOT/data/wiki18_bm25/bm25}"
CORPUS_PATH="${CORPUS_PATH:-$REPO_ROOT/data/wiki18_bm25/corpus/wiki-18.jsonl.gz}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/data/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/data/huggingface/datasets}"

[ -d "$INDEX_PATH" ] || { echo "Missing BM25 index: $INDEX_PATH" >&2; exit 1; }
[ -s "$CORPUS_PATH" ] || { echo "Missing Wikipedia corpus: $CORPUS_PATH" >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_ENV"
export JAVA_HOME="$CONDA_PREFIX"
export PATH="$JAVA_HOME/bin:$PATH"
JAVA_VERSION_OUTPUT="$(java -version 2>&1)"
case "$JAVA_VERSION_OUTPUT" in
  *'version "17.'*|*'version "17"'*) ;;
  *)
    printf 'Expected Java 17 in %s, but found:\n%s\n' "$RETRIEVER_ENV" "$JAVA_VERSION_OUTPUT" >&2
    exit 1
    ;;
esac
mkdir -p "$HF_DATASETS_CACHE"
export CUDA_VISIBLE_DEVICES=
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms2g -Xmx16g}"
cd "$REPO_ROOT"
exec python scripts/course_reproduction/bm25_server.py \
  --index-path "$INDEX_PATH" \
  --corpus-path "$CORPUS_PATH" \
  --topk 3
