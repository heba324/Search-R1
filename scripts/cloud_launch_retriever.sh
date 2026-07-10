#!/usr/bin/env bash
set -euo pipefail

# Launch either the tiny smoke retriever or the full Wikipedia retriever.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
ASSET_PROFILE="${ASSET_PROFILE:-smoke}"
INDEX_FILE="${INDEX_FILE:-}"
CORPUS_FILE="${CORPUS_FILE:-}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
TOPK="${TOPK:-3}"

case "$ASSET_PROFILE" in
  smoke)
    INDEX_FILE="${INDEX_FILE:-$REPO_ROOT/data/smoke_retriever/e5_Flat.index}"
    CORPUS_FILE="${CORPUS_FILE:-$REPO_ROOT/example/corpus.jsonl}"
    ;;
  full)
    INDEX_FILE="${INDEX_FILE:-$REPO_ROOT/data/wiki18/e5_Flat.index}"
    CORPUS_FILE="${CORPUS_FILE:-$REPO_ROOT/data/wiki18/wiki-18.jsonl}"
    ;;
  *)
    echo "ASSET_PROFILE must be 'smoke' or 'full'; got: $ASSET_PROFILE" >&2
    exit 1
    ;;
esac

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

python - <<'PY'
import socket

with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", 8000)) == 0:
        raise SystemExit("Port 8000 is already in use. Stop the old retriever before continuing.")
PY

mkdir -p "$REPO_ROOT/artifacts"
printf '%s\n%s\n%s\n' \
  "$ASSET_PROFILE" "$INDEX_FILE" "$CORPUS_FILE" \
  > "$REPO_ROOT/artifacts/retriever_profile.txt"

echo "Starting $ASSET_PROFILE retriever on http://127.0.0.1:8000/retrieve"
exec python search_r1/search/retrieval_server.py \
  --index_path "$INDEX_FILE" \
  --corpus_path "$CORPUS_FILE" \
  --topk "$TOPK" \
  --retriever_name "$RETRIEVER_NAME" \
  --retriever_model "$RETRIEVER_MODEL" \
  --faiss_gpu
