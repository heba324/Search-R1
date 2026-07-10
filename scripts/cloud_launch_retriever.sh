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
RETRIEVER_READY_TIMEOUT="${RETRIEVER_READY_TIMEOUT:-1200}"
PROFILE_MARKER="$REPO_ROOT/artifacts/retriever_profile.txt"
MARKER_TMP="$REPO_ROOT/artifacts/retriever_profile.txt.tmp"
READY_LOG="$REPO_ROOT/artifacts/retriever_startup_check.log"

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

if [ ! -s "$INDEX_FILE" ]; then
  echo "Missing index file: $INDEX_FILE" >&2
  exit 1
fi

if [ ! -s "$CORPUS_FILE" ]; then
  echo "Missing corpus file: $CORPUS_FILE" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$RETRIEVER_ENV"

case "$RETRIEVER_READY_TIMEOUT" in
  ''|*[!0-9]*)
    echo "RETRIEVER_READY_TIMEOUT must be a positive integer number of seconds." >&2
    exit 1
    ;;
esac
if [ "$RETRIEVER_READY_TIMEOUT" -lt 1 ]; then
  echo "RETRIEVER_READY_TIMEOUT must be at least 1 second." >&2
  exit 1
fi

python - <<'PY'
import socket

with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", 8000)) == 0:
        raise SystemExit("Port 8000 is already in use. Stop the old retriever before continuing.")
PY

echo "Starting $ASSET_PROFILE retriever on http://127.0.0.1:8000/retrieve"
mkdir -p "$REPO_ROOT/artifacts"
rm -f "$PROFILE_MARKER" "$MARKER_TMP" "$READY_LOG"

python search_r1/search/retrieval_server.py \
  --index_path "$INDEX_FILE" \
  --corpus_path "$CORPUS_FILE" \
  --topk "$TOPK" \
  --retriever_name "$RETRIEVER_NAME" \
  --retriever_model "$RETRIEVER_MODEL" \
  --faiss_gpu &
SERVER_PID=$!

cleanup() {
  rm -f "$PROFILE_MARKER" "$MARKER_TMP"
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + RETRIEVER_READY_TIMEOUT))
ready=false
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    wait "$SERVER_PID" || true
    echo "Retriever process exited before becoming ready." >&2
    [ ! -s "$READY_LOG" ] || cat "$READY_LOG" >&2
    exit 1
  fi

  if RETRIEVER_URL="http://127.0.0.1:8000/retrieve" TOPK="$TOPK" \
      python scripts/cloud_check_retriever.py > "$READY_LOG" 2>&1; then
    ready=true
    break
  fi
  sleep 10
done

if [ "$ready" != true ]; then
  echo "Retriever did not become ready within $RETRIEVER_READY_TIMEOUT seconds." >&2
  [ ! -s "$READY_LOG" ] || cat "$READY_LOG" >&2
  exit 1
fi

git_commit="$(git rev-parse HEAD)"
cat > "$MARKER_TMP" <<EOF
profile=$ASSET_PROFILE
git_commit=$git_commit
pid=$SERVER_PID
index_file=$INDEX_FILE
corpus_file=$CORPUS_FILE
EOF
mv "$MARKER_TMP" "$PROFILE_MARKER"

cat "$READY_LOG"
echo "Retriever profile published after readiness: $PROFILE_MARKER"
wait "$SERVER_PID"
