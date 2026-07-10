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
  smoke)
    EXPECTED_ATTESTATION="$REPO_ROOT/artifacts/smoke_passed.txt"
    EXPECTED_STATUS=passed
    DEFAULT_EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-smoke
    ;;
  full)
    EXPECTED_ATTESTATION="$REPO_ROOT/artifacts/full_completed.txt"
    EXPECTED_STATUS=completed
    DEFAULT_EXPERIMENT_NAME=nq-search-r1-grpo-qwen2.5-3b-em
    ;;
  *)
    echo "Usage: bash scripts/cloud_collect_evidence.sh smoke|full" >&2
    exit 1
    ;;
esac

RUN_STATUS=not_completed
EXPERIMENT_NAME="$DEFAULT_EXPERIMENT_NAME"
current_commit="$(git rev-parse HEAD)"

if [ -s "$EXPECTED_ATTESTATION" ] \
    && grep -Fqx "status=$EXPECTED_STATUS" "$EXPECTED_ATTESTATION" \
    && grep -Fqx "mode=$MODE" "$EXPECTED_ATTESTATION" \
    && grep -Fqx "git_commit=$current_commit" "$EXPECTED_ATTESTATION"; then
  attested_experiment="$(sed -n 's/^experiment_name=//p' "$EXPECTED_ATTESTATION" | head -n 1)"
  case "$attested_experiment" in
    ''|*[!A-Za-z0-9._-]*) ;;
    *)
      EXPERIMENT_NAME="$attested_experiment"
      RUN_STATUS="$EXPECTED_STATUS"
      ;;
  esac
fi
EXPECTED_LOG="$REPO_ROOT/$EXPERIMENT_NAME.log"

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
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT_DIR/collected-at-utc.txt"

capture nvidia-smi.txt nvidia-smi
capture memory.txt free -h
capture disk.txt df -h
capture uname.txt uname -a
capture conda-info.txt conda info
capture conda-Search-R1-explicit.txt conda list -n "$SEARCH_ENV" --explicit
capture conda-Search-R1-retriever-explicit.txt \
  conda list -n "$RETRIEVER_ENV" --explicit
capture pip-Search-R1-freeze.txt \
  conda run -n "$SEARCH_ENV" python -m pip freeze
capture pip-Search-R1-retriever-freeze.txt \
  conda run -n "$RETRIEVER_ENV" python -m pip freeze
capture huggingface-cache.txt \
  conda run -n "$SEARCH_ENV" huggingface-cli scan-cache

: > "$OUTPUT_DIR/data-sha256.txt"
for parquet in "$DATA_DIR/train.parquet" "$DATA_DIR/test.parquet"; do
  if [ -f "$parquet" ]; then
    sha256sum "$parquet" >> "$OUTPUT_DIR/data-sha256.txt"
  else
    echo "missing: $parquet" >> "$OUTPUT_DIR/data-sha256.txt"
  fi
done

if [ -f "$EXPECTED_LOG" ]; then
  cp "$EXPECTED_LOG" "$OUTPUT_DIR/logs/"
else
  echo "missing: $EXPECTED_LOG" > "$OUTPUT_DIR/logs/missing-training-log.txt"
fi

for supporting_log in \
  "$REPO_ROOT/setup-Search-R1.log" \
  "$REPO_ROOT/setup-retriever.log" \
  "$REPO_ROOT/prepare-${MODE}.log" \
  "$REPO_ROOT/retriever-${MODE}.log"; do
  if [ -f "$supporting_log" ]; then
    cp "$supporting_log" "$OUTPUT_DIR/logs/"
  fi
done

if [ -f "$REPO_ROOT/artifacts/retriever_profile.txt" ]; then
  cp "$REPO_ROOT/artifacts/retriever_profile.txt" "$OUTPUT_DIR/"
fi
if [ -f "$EXPECTED_ATTESTATION" ]; then
  cp "$EXPECTED_ATTESTATION" "$OUTPUT_DIR/"
fi
if [ -f "$REPO_ROOT/data/wiki18/downloads.sha256" ]; then
  cp "$REPO_ROOT/data/wiki18/downloads.sha256" "$OUTPUT_DIR/"
fi
for validation_file in \
  "$REPO_ROOT/data/wiki18/e5_Flat.index.validated" \
  "$REPO_ROOT/data/wiki18/wiki-18.jsonl.validated"; do
  if [ -f "$validation_file" ]; then
    cp "$validation_file" "$OUTPUT_DIR/"
  fi
done

EXPERIMENT_CHECKPOINTS="$REPO_ROOT/verl_checkpoints/$EXPERIMENT_NAME"
if [ -d "$EXPERIMENT_CHECKPOINTS" ]; then
  find "$EXPERIMENT_CHECKPOINTS" -maxdepth 4 -type f -printf '%p\t%s bytes\n' \
    > "$OUTPUT_DIR/checkpoint-files.txt"
else
  echo "No checkpoint directory was present for $EXPERIMENT_NAME." \
    > "$OUTPUT_DIR/checkpoint-files.txt"
fi

cat > "$OUTPUT_DIR/README.txt" <<EOF
Search-R1 reproduction evidence
mode: $MODE
run status: $RUN_STATUS
experiment: $EXPERIMENT_NAME
expected training log: $EXPECTED_LOG
git commit: $current_commit

Smoke evidence proves only that the startup pipeline was exercised.
Full reproduction claims additionally require completed training logs,
checkpoints, evaluation metrics, and comparison with the paper.
EOF

echo "Evidence collected at: $OUTPUT_DIR"
