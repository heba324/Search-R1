#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SAVE_PATH="${SAVE_PATH:-$REPO_ROOT/data/wiki18}"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
PART_AA_SHA256=a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023
PART_AB_SHA256=b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11
CORPUS_GZ_SHA256=7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db

mkdir -p "$SAVE_PATH"
cd "$REPO_ROOT"
python scripts/download.py --save_path "$SAVE_PATH"
printf '%s  %s\n%s  %s\n%s  %s\n' \
  "$PART_AA_SHA256" part_aa "$PART_AB_SHA256" part_ab "$CORPUS_GZ_SHA256" wiki-18.jsonl.gz \
  > "$SAVE_PATH/paper-v1-downloads.sha256"
(cd "$SAVE_PATH" && sha256sum -c paper-v1-downloads.sha256)

cat "$SAVE_PATH/part_aa" "$SAVE_PATH/part_ab" > "$SAVE_PATH/e5_Flat.index.tmp"
mv "$SAVE_PATH/e5_Flat.index.tmp" "$SAVE_PATH/e5_Flat.index"
gzip -cd "$SAVE_PATH/wiki-18.jsonl.gz" > "$SAVE_PATH/wiki-18.jsonl.tmp"
mv "$SAVE_PATH/wiki-18.jsonl.tmp" "$SAVE_PATH/wiki-18.jsonl"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_ENV"
INDEX_FILE="$SAVE_PATH/e5_Flat.index" python - <<'PY'
import os
import faiss
index = faiss.read_index(os.environ["INDEX_FILE"])
assert index.d == 768 and index.ntotal > 0
print(f"Verified E5 index: d={index.d}, ntotal={index.ntotal}")
PY
