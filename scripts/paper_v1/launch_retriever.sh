#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
INDEX_PATH="${INDEX_PATH:-$REPO_ROOT/data/wiki18/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-$REPO_ROOT/data/wiki18/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$REPO_ROOT/data/models/e5-base-v2}"

cd "$REPO_ROOT"
python3 scripts/paper_v1/preflight.py --repo-root "$REPO_ROOT" --require-assets
if command -v ss >/dev/null && ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8000$'; then
  echo "Port 8000 is already occupied." >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_ENV"
exec python search_r1/search/retrieval_server.py \
  --index_path "$INDEX_PATH" \
  --corpus_path "$CORPUS_PATH" \
  --topk 3 \
  --retriever_model "$RETRIEVER_MODEL"
