#!/usr/bin/env bash
set -euo pipefail

# Download the full Wikipedia retrieval resources and process NQ data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/data/wiki18}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/nq_search}"
PART_AA="$SAVE_PATH/part_aa"
PART_AB="$SAVE_PATH/part_ab"
INDEX_FILE="$SAVE_PATH/e5_Flat.index"
INDEX_TMP="$SAVE_PATH/e5_Flat.index.tmp"
CORPUS_GZ="$SAVE_PATH/wiki-18.jsonl.gz"
CORPUS_FILE="$SAVE_PATH/wiki-18.jsonl"
CORPUS_TMP="$SAVE_PATH/wiki-18.jsonl.tmp"
INDEX_VALIDATION="$SAVE_PATH/e5_Flat.index.validated"
CORPUS_VALIDATION="$SAVE_PATH/wiki-18.jsonl.validated"
DOWNLOAD_MANIFEST="$SAVE_PATH/downloads.sha256"
PART_AA_SIZE=42949672960
PART_AB_SIZE=21609402413
CORPUS_GZ_SIZE=5123307260
PART_AA_SHA256=a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023
PART_AB_SHA256=b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11
CORPUS_GZ_SHA256=7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db

verify_download() {
  local file="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local actual_sha256

  if [ ! -s "$file" ]; then
    echo "Downloaded file is missing or empty: $file" >&2
    exit 1
  fi
  if [ "$(stat -c%s "$file")" -ne "$expected_size" ]; then
    echo "Downloaded file has the wrong byte size: $file" >&2
    echo "Remove it and rerun the download command." >&2
    exit 1
  fi

  actual_sha256="$(sha256sum "$file" | awk '{print $1}')"
  if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "Downloaded file failed SHA-256 verification: $file" >&2
    echo "Remove it and rerun the download command." >&2
    exit 1
  fi
}

source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$SAVE_PATH" "$DATA_DIR"

conda activate "$SEARCH_ENV"
python scripts/download.py --save_path "$SAVE_PATH"

verify_download "$PART_AA" "$PART_AA_SIZE" "$PART_AA_SHA256"
verify_download "$PART_AB" "$PART_AB_SIZE" "$PART_AB_SHA256"
verify_download "$CORPUS_GZ" "$CORPUS_GZ_SIZE" "$CORPUS_GZ_SHA256"
cat > "$DOWNLOAD_MANIFEST.tmp" <<EOF
$PART_AA_SHA256  part_aa
$PART_AB_SHA256  part_ab
$CORPUS_GZ_SHA256  wiki-18.jsonl.gz
EOF
mv "$DOWNLOAD_MANIFEST.tmp" "$DOWNLOAD_MANIFEST"

expected_index_size=$((PART_AA_SIZE + PART_AB_SIZE))
actual_index_size=0
if [ -f "$INDEX_FILE" ]; then
  actual_index_size="$(stat -c%s "$INDEX_FILE")"
fi
validation_key="$PART_AA_SHA256:$PART_AB_SHA256"
index_needs_rebuild=false
if [ "$actual_index_size" -ne "$expected_index_size" ] \
    || [ ! -s "$INDEX_VALIDATION" ]; then
  index_needs_rebuild=true
else
  saved_source_key="$(sed -n 's/^source_key=//p' "$INDEX_VALIDATION")"
  saved_index_sha256="$(sed -n 's/^index_sha256=//p' "$INDEX_VALIDATION")"
  actual_index_sha256="$(sha256sum "$INDEX_FILE" | awk '{print $1}')"
  if [ "$saved_source_key" != "$validation_key" ] \
      || [ "$saved_index_sha256" != "$actual_index_sha256" ]; then
    index_needs_rebuild=true
  fi
fi

if [ "$index_needs_rebuild" = true ]; then
  rm -f "$INDEX_TMP"
  trap 'rm -f "$INDEX_TMP"' EXIT
  cat "$PART_AA" "$PART_AB" > "$INDEX_TMP"
  if [ "$(stat -c%s "$INDEX_TMP")" -ne "$expected_index_size" ]; then
    echo "Joined FAISS index size does not match its two downloaded parts." >&2
    exit 1
  fi
  mv "$INDEX_TMP" "$INDEX_FILE"
  trap - EXIT
  actual_index_sha256="$(sha256sum "$INDEX_FILE" | awk '{print $1}')"
  conda activate "$RETRIEVER_ENV"
  INDEX_FILE="$INDEX_FILE" python - <<'PY'
import os

import faiss

index_path = os.environ["INDEX_FILE"]
index = faiss.read_index(index_path)
if index.d != 768:
    raise SystemExit(f"Unexpected E5 index dimension: {index.d}")
if index.ntotal < 1:
    raise SystemExit("The E5 index contains no vectors.")
print(f"FAISS index verified: dimension={index.d}, vectors={index.ntotal}")
PY
  cat > "$INDEX_VALIDATION.tmp" <<EOF
source_key=$validation_key
index_sha256=$actual_index_sha256
EOF
  mv "$INDEX_VALIDATION.tmp" "$INDEX_VALIDATION"
  conda activate "$SEARCH_ENV"
fi

corpus_needs_rebuild=false
if [ ! -s "$CORPUS_FILE" ] || [ ! -s "$CORPUS_VALIDATION" ]; then
  corpus_needs_rebuild=true
else
  saved_corpus_source="$(sed -n 's/^source_gz_sha256=//p' "$CORPUS_VALIDATION")"
  saved_corpus_sha256="$(sed -n 's/^corpus_sha256=//p' "$CORPUS_VALIDATION")"
  actual_corpus_sha256="$(sha256sum "$CORPUS_FILE" | awk '{print $1}')"
  if [ "$saved_corpus_source" != "$CORPUS_GZ_SHA256" ] \
      || [ "$saved_corpus_sha256" != "$actual_corpus_sha256" ]; then
    corpus_needs_rebuild=true
  fi
fi

if [ "$corpus_needs_rebuild" = true ]; then
  rm -f "$CORPUS_TMP"
  trap 'rm -f "$CORPUS_TMP"' EXIT
  gzip -cd "$CORPUS_GZ" > "$CORPUS_TMP"
  if [ ! -s "$CORPUS_TMP" ]; then
    echo "Decompressed Wikipedia corpus is empty." >&2
    exit 1
  fi
  mv "$CORPUS_TMP" "$CORPUS_FILE"
  trap - EXIT
  actual_corpus_sha256="$(sha256sum "$CORPUS_FILE" | awk '{print $1}')"
  cat > "$CORPUS_VALIDATION.tmp" <<EOF
source_gz_sha256=$CORPUS_GZ_SHA256
corpus_sha256=$actual_corpus_sha256
EOF
  mv "$CORPUS_VALIDATION.tmp" "$CORPUS_VALIDATION"
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
