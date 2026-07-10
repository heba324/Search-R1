#!/usr/bin/env bash
set -euo pipefail

# Collect reproducibility metadata without recording tokens or environment variables.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-}"
SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/nq_search}"

case "$MODE" in
  smoke|full) ;;
  *)
    echo "Usage: bash scripts/cloud_collect_evidence.sh smoke|full" >&2
    exit 1
    ;;
esac

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$REPO_ROOT/artifacts/reproduction-${MODE}-${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR/logs"

capture() {
  local output_file="$1"
  shift
  if ! "$@" > "$OUTPUT_DIR/$output_file" 2>&1; then
    printf 'Command failed while collecting this file: %s\n' "$*" \
      >> "$OUTPUT_DIR/$output_file"
  fi
}

git rev-parse HEAD > "$OUTPUT_DIR/git-commit.txt"
git status --short > "$OUTPUT_DIR/git-status.txt"
git remote -v > "$OUTPUT_DIR/git-remotes.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT_DIR/collected-at-utc.txt"

capture nvidia-smi.txt nvidia-smi
capture memory.txt free -h
capture disk.txt df -h
capture uname.txt uname -a
capture conda-info.txt conda info
capture conda-Search-R1-explicit.txt conda list -n "$SEARCH_ENV" --explicit
capture conda-Search-R1-retriever-explicit.txt \
  conda list -n "$RETRIEVER_ENV" --explicit

: > "$OUTPUT_DIR/data-sha256.txt"
for parquet in "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet"; do
  if [ -f "$parquet" ]; then
    sha256sum "$parquet" >> "$OUTPUT_DIR/data-sha256.txt"
  else
    echo "missing: $parquet" >> "$OUTPUT_DIR/data-sha256.txt"
  fi
done

for log_file in "$REPO_ROOT"/*.log; do
  if [ -f "$log_file" ]; then
    cp "$log_file" "$OUTPUT_DIR/logs/"
  fi
done

if [ -f "$REPO_ROOT/artifacts/retriever_profile.txt" ]; then
  cp "$REPO_ROOT/artifacts/retriever_profile.txt" "$OUTPUT_DIR/"
fi

if [ -d "$REPO_ROOT/verl_checkpoints" ]; then
  find "$REPO_ROOT/verl_checkpoints" -maxdepth 4 -type f -printf '%p\t%s bytes\n' \
    > "$OUTPUT_DIR/checkpoint-files.txt"
else
  echo "No verl_checkpoints directory was present." \
    > "$OUTPUT_DIR/checkpoint-files.txt"
fi

cat > "$OUTPUT_DIR/README.txt" <<EOF
Search-R1 reproduction evidence
mode: $MODE
git commit: $(git rev-parse HEAD)

Smoke evidence proves only that the startup pipeline was exercised.
Full reproduction claims additionally require completed training logs,
checkpoints, evaluation metrics, and comparison with the paper.
EOF

echo "Evidence collected at: $OUTPUT_DIR"
