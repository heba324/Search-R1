#!/usr/bin/env bash
set -euo pipefail

# Launch the local dense retriever server.
# Keep this process running in its own tmux pane/session.

RETRIEVER_ENV="${RETRIEVER_ENV:-retriever}"
SAVE_PATH="${SAVE_PATH:-$PWD/data/wiki18}"
INDEX_FILE="${INDEX_FILE:-$SAVE_PATH/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-$SAVE_PATH/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"
PORT="${PORT:-8000}"

if [ ! -f "$INDEX_FILE" ]; then
  echo "Missing index file: $INDEX_FILE" >&2
  exit 1
fi

if [ ! -f "$CORPUS_FILE" ]; then
  echo "Missing corpus file: $CORPUS_FILE" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_ENV"

python search_r1/search/retrieval_server.py \
  --index_path "$INDEX_FILE" \
  --corpus_path "$CORPUS_FILE" \
  --topk "$TOPK" \
  --retriever_name "$RETRIEVER_NAME" \
  --retriever_model "$RETRIEVER_MODEL" \
  --faiss_gpu
