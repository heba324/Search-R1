#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INDEX_ROOT="${INDEX_ROOT:-$REPO_ROOT/data/wiki18_bm25}"
INDEX_REPO="PeterJinGo/wiki-18-bm25-index"
INDEX_REVISION="2c7554f"
CORPUS_REPO="PeterJinGo/wiki-18-corpus"
CORPUS_REVISION="69c1c00"
CORPUS_ROOT="$INDEX_ROOT/corpus"
export HF_HOME="${HF_HOME:-$REPO_ROOT/data/huggingface}"

mkdir -p "$INDEX_ROOT" "$CORPUS_ROOT" "$HF_HOME"
huggingface-cli download --repo-type dataset --revision "$INDEX_REVISION" --local-dir "$INDEX_ROOT" "$INDEX_REPO"
huggingface-cli download --repo-type dataset --revision "$CORPUS_REVISION" --local-dir "$CORPUS_ROOT" "$CORPUS_REPO"
[ -d "$INDEX_ROOT/bm25" ] || { echo "BM25 index directory is missing." >&2; exit 1; }
[ -n "$(find "$INDEX_ROOT/bm25" -maxdepth 1 -type f -print -quit)" ] || { echo "BM25 index is empty." >&2; exit 1; }
[ -s "$CORPUS_ROOT/wiki-18.jsonl.gz" ] || { echo "Compressed Wikipedia corpus is missing." >&2; exit 1; }
printf 'index_repo=%s\nindex_revision=%s\ncorpus_repo=%s\ncorpus_revision=%s\n' \
  "$INDEX_REPO" "$INDEX_REVISION" "$CORPUS_REPO" "$CORPUS_REVISION" > "$INDEX_ROOT/course-bm25-revision.txt"
