#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$REPO_ROOT/artifacts/improvement/evidence"
mkdir -p "$OUT"

git -C "$REPO_ROOT" rev-parse HEAD > "$OUT/git-head.txt"
git -C "$REPO_ROOT" status --short --branch > "$OUT/git-status.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
free -h > "$OUT/memory.txt"
df -h "$REPO_ROOT" > "$OUT/disk.txt"

find "$REPO_ROOT/artifacts/improvement" -path "$OUT" -prune -o -type f -print0 | \
  sort -z | xargs -0 sha256sum > "$OUT/artifacts.sha256"

rm -f "$OUT/search-r1-cegr-evidence.tar.gz"
TMP_ARCHIVE="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$TMP_ARCHIVE"' EXIT
tar -czf "$TMP_ARCHIVE" -C "$REPO_ROOT" \
  artifacts/improvement \
  artifacts/course-reproduction/evaluation/baseline-paired \
  artifacts/course-reproduction/evaluation/cegr-post-rl \
  docs/improvement_experiment_zh.md \
  docs/research/search_rl_reward_improvement.md
mv "$TMP_ARCHIVE" "$OUT/search-r1-cegr-evidence.tar.gz"
trap - EXIT
echo "Improvement evidence: $OUT/search-r1-cegr-evidence.tar.gz"
