#!/usr/bin/env bash
set -euo pipefail

# Build a tiny retriever index and prepare NQ without downloading Wikipedia.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/nq_search}"
SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/data/smoke_retriever}"
CORPUS_FILE="$REPO_ROOT/example/corpus.jsonl"
INDEX_FILE="$SAVE_PATH/e5_Flat.index"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Run this on the rented Linux host after setup." >&2
  exit 1
fi
if [ ! -s "$CORPUS_FILE" ]; then
  echo "Missing tracked smoke corpus: $CORPUS_FILE" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$DATA_DIR" "$SAVE_PATH"

conda activate "$SEARCH_ENV"
if [ ! -s "$DATA_DIR/train.parquet" ] || [ ! -s "$DATA_DIR/test.parquet" ]; then
  python scripts/data_process/nq_search.py --local_dir "$DATA_DIR"
fi

conda activate "$RETRIEVER_ENV"
if [ ! -s "$INDEX_FILE" ]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    python search_r1/search/index_builder.py \
      --retrieval_method e5 \
      --model_path intfloat/e5-base-v2 \
      --corpus_path "$CORPUS_FILE" \
      --save_dir "$SAVE_PATH" \
      --use_fp16 \
      --max_length 256 \
      --batch_size 32 \
      --pooling_method mean \
      --faiss_type Flat
fi

for required_file in \
  "$INDEX_FILE" \
  "$DATA_DIR/train.parquet" \
  "$DATA_DIR/test.parquet"; do
  if [ ! -s "$required_file" ]; then
    echo "Smoke asset is missing or empty: $required_file" >&2
    exit 1
  fi
done

echo "Smoke index: $INDEX_FILE"
echo "Smoke corpus: $CORPUS_FILE"
echo "NQ train data: $DATA_DIR/train.parquet"
echo "NQ test data: $DATA_DIR/test.parquet"
echo "Smoke assets are for startup validation only, not paper evaluation."
