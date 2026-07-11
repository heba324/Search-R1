#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$REPO_ROOT/artifacts/paper-v1/evidence-$STAMP"
mkdir -p "$OUT"
cd "$REPO_ROOT"

git rev-parse HEAD > "$OUT/wrapper-commit.txt"
printf '%s\n' 118c6e7 > "$OUT/frozen-author-commit.txt"
git status --short > "$OUT/git-status.txt"
git diff --exit-code 118c6e7 -- search_r1 verl scripts/nq_hotpotqa train_ppo.sh train_grpo.sh retrieval_launch.sh \
  > "$OUT/author-core-diff.txt" || true
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || true
free -h > "$OUT/memory.txt" 2>&1 || true
df -h > "$OUT/disk.txt" 2>&1 || true
conda list -n "$SEARCH_ENV" --explicit > "$OUT/Search-R1-conda-explicit.txt" 2>&1 || true
conda list -n "$RETRIEVER_ENV" --explicit > "$OUT/retriever-conda-explicit.txt" 2>&1 || true
conda run -n "$SEARCH_ENV" python -m pip freeze > "$OUT/Search-R1-pip-freeze.txt" 2>&1 || true
conda run -n "$RETRIEVER_ENV" python -m pip freeze > "$OUT/retriever-pip-freeze.txt" 2>&1 || true

for file in \
  data/nq_hotpotqa_train/paper-v1.sha256 \
  data/wiki18/paper-v1-downloads.sha256 \
  data/models/paper-v1-model-revisions.txt \
  artifacts/paper-v1/training_completed.txt \
  artifacts/paper-v1/evaluation_completed.txt \
  artifacts/paper-v1/search-r1-v1-qwen2.5-3b-it-ppo-em.log \
  artifacts/paper-v1/evaluation-seven-datasets.log; do
  [ ! -f "$file" ] || cp "$file" "$OUT/"
done
find verl_checkpoints/search-r1-v1-qwen2.5-3b-it-ppo-em -type f -printf '%p %s bytes\n' \
  > "$OUT/checkpoint-files.txt" 2>/dev/null || true
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "Evidence archive: $OUT.tar.gz"
