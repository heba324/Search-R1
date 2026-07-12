#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INDEX_ROOT="${INDEX_ROOT:-$REPO_ROOT/data/wiki18_bm25}"
INDEX_REPO="PeterJinGo/wiki-18-bm25-index"
INDEX_REVISION="2c7554f"

mkdir -p "$INDEX_ROOT"
huggingface-cli download --repo-type dataset --revision "$INDEX_REVISION" --local-dir "$INDEX_ROOT" "$INDEX_REPO"
[ -d "$INDEX_ROOT/bm25" ] || { echo "BM25 index directory is missing." >&2; exit 1; }
[ -n "$(find "$INDEX_ROOT/bm25" -maxdepth 1 -type f -print -quit)" ] || { echo "BM25 index is empty." >&2; exit 1; }
printf 'repo=%s\nrevision=%s\n' "$INDEX_REPO" "$INDEX_REVISION" > "$INDEX_ROOT/course-bm25-revision.txt"
