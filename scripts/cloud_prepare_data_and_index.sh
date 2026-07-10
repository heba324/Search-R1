#!/usr/bin/env bash
set -euo pipefail

# Download the full Wikipedia retrieval resources and process NQ data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/data/wiki18}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/nq_search}"
PART_AA="$SAVE_PATH/part_aa"
PART_AB="$SAVE_PATH/part_ab"
INDEX_FILE="$SAVE_PATH/e5_Flat.index"
INDEX_TMP="$SAVE_PATH/e5_Flat.index.tmp"
CORPUS_GZ="$SAVE_PATH/wiki-18.jsonl.gz"
CORPUS_FILE="$SAVE_PATH/wiki-18.jsonl"
CORPUS_TMP="$SAVE_PATH/wiki-18.jsonl.tmp"

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$SAVE_PATH" "$DATA_DIR"

conda activate "$SEARCH_ENV"
python scripts/download.py --save_path "$SAVE_PATH"

for part in "$PART_AA" "$PART_AB"; do
  if [ ! -s "$part" ]; then
    echo "Downloaded index part is missing or empty: $part" >&2
    exit 1
  fi
done

expected_index_size=$(( $(stat -c%s "$PART_AA") + $(stat -c%s "$PART_AB") ))
actual_index_size=0
if [ -f "$INDEX_FILE" ]; then
  actual_index_size="$(stat -c%s "$INDEX_FILE")"
fi
if [ "$actual_index_size" -ne "$expected_index_size" ]; then
  rm -f "$INDEX_TMP"
  trap 'rm -f "$INDEX_TMP"' EXIT
  cat "$PART_AA" "$PART_AB" > "$INDEX_TMP"
  if [ "$(stat -c%s "$INDEX_TMP")" -ne "$expected_index_size" ]; then
    echo "Joined FAISS index size does not match its two downloaded parts." >&2
    exit 1
  fi
  mv "$INDEX_TMP" "$INDEX_FILE"
  trap - EXIT
fi

if [ -s "$CORPUS_GZ" ] && [ ! -s "$CORPUS_FILE" ]; then
  rm -f "$CORPUS_TMP"
  trap 'rm -f "$CORPUS_TMP"' EXIT
  gzip -cd "$CORPUS_GZ" > "$CORPUS_TMP"
  if [ ! -s "$CORPUS_TMP" ]; then
    echo "Decompressed Wikipedia corpus is empty." >&2
    exit 1
  fi
  mv "$CORPUS_TMP" "$CORPUS_FILE"
  trap - EXIT
fi

if [ ! -s "$DATA_DIR/train.parquet" ] || [ ! -s "$DATA_DIR/test.parquet" ]; then
  python scripts/data_process/nq_search.py --local_dir "$DATA_DIR"
fi

for required_file in \
  "$INDEX_FILE" \
  "$CORPUS_FILE" \
  "$DATA_DIR/train.parquet" \
  "$DATA_DIR/test.parquet"; do
  if [ ! -s "$required_file" ]; then
    echo "Full reproduction asset is missing or empty: $required_file" >&2
    exit 1
  fi
done

echo "Index: $INDEX_FILE"
echo "Corpus: $CORPUS_FILE"
echo "Train data: $DATA_DIR/train.parquet"
echo "Test data: $DATA_DIR/test.parquet"
