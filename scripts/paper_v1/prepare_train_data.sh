#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/nq_hotpotqa_train}"
DATASET_REPO="PeterJinGo/nq_hotpotqa_train"
DATASET_REVISION="b7d80abfee334a7a91cb377544f09180d58b34f6"
TRAIN_SIZE=355663891
TEST_SIZE=70370337
TRAIN_SHA256=c3cc21e862a8469105de666101578cbff23cdc77e91a803cef102622c89cc4f6
TEST_SHA256=30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc

verify_file() {
  local file="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local actual_sha
  [ -s "$file" ] || { echo "Missing dataset file: $file" >&2; exit 1; }
  [ "$(stat -c%s "$file")" = "$expected_size" ] || { echo "Wrong byte size: $file" >&2; exit 1; }
  actual_sha="$(sha256sum "$file" | awk '{print $1}')"
  [ "$actual_sha" = "$expected_sha" ] || { echo "SHA-256 mismatch: $file" >&2; exit 1; }
}

mkdir -p "$DATA_DIR"
huggingface-cli download \
  --repo-type dataset \
  --revision "$DATASET_REVISION" \
  --local-dir "$DATA_DIR" \
  "$DATASET_REPO"

verify_file "$DATA_DIR/train.parquet" "$TRAIN_SIZE" "$TRAIN_SHA256"
verify_file "$DATA_DIR/test.parquet" "$TEST_SIZE" "$TEST_SHA256"
printf '%s  %s\n%s  %s\n' \
  "$TRAIN_SHA256" train.parquet \
  "$TEST_SHA256" test.parquet > "$DATA_DIR/paper-v1.sha256"
echo "Paper v1 training data verified at $DATASET_REVISION"
