#!/usr/bin/env bash
set -euo pipefail

# Download Search-R1 retrieval resources and process NQ data.
# Run from the repository root after creating both conda environments.

SEARCH_ENV="${SEARCH_ENV:-searchr1}"
SAVE_PATH="${SAVE_PATH:-$PWD/data/wiki18}"
DATA_DIR="${DATA_DIR:-$PWD/data/nq_search}"

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$SAVE_PATH" "$DATA_DIR"

conda activate "$SEARCH_ENV"
python scripts/download.py --save_path "$SAVE_PATH"

cat "$SAVE_PATH"/part_* > "$SAVE_PATH/e5_Flat.index"

if [ -f "$SAVE_PATH/wiki-18.jsonl.gz" ] && [ ! -f "$SAVE_PATH/wiki-18.jsonl" ]; then
  gzip -d "$SAVE_PATH/wiki-18.jsonl.gz"
fi

python scripts/data_process/nq_search.py --local_dir "$DATA_DIR"

echo "Index: $SAVE_PATH/e5_Flat.index"
echo "Corpus: $SAVE_PATH/wiki-18.jsonl"
echo "Train data: $DATA_DIR/train.parquet"
echo "Test data: $DATA_DIR/test.parquet"
