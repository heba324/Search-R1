#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$REPO_ROOT/artifacts/improvement-v2/evidence"
mkdir -p "$OUT"

git -C "$REPO_ROOT" rev-parse HEAD > "$OUT/git-head.txt"
git -C "$REPO_ROOT" status --short --branch > "$OUT/git-status.txt"
git -C "$REPO_ROOT" diff --name-only \
  8672aad0f4089f0fca388601cd9ce20fc9b8b776 -- search_r1 verl \
  > "$OUT/core-diff.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
free -h > "$OUT/memory.txt"
df -h "$REPO_ROOT" > "$OUT/disk.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SEARCH_ENV:-Search-R1}"
conda list > "$OUT/search-env-conda-list.txt"
python3 -m pip freeze > "$OUT/search-env-pip-freeze.txt"

find "$REPO_ROOT/artifacts/improvement-v2" -path "$OUT" -prune -o \
  -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/artifacts.sha256"

ARCHIVE="$OUT/search-r1-cegr-v2-evidence.tar.gz"
rm -f "$ARCHIVE"
TMP_ARCHIVE="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$TMP_ARCHIVE"' EXIT
tar --exclude='artifacts/improvement-v2/evidence/search-r1-cegr-v2-evidence.tar.gz*' \
  -czf "$TMP_ARCHIVE" -C "$REPO_ROOT" \
  artifacts/improvement-v2 \
  data/improvement_v2/pilot_manifest.json \
  docs/cegr_v2_experiment_zh.md \
  docs/research/cegr_v2_literature_review.md
mv "$TMP_ARCHIVE" "$ARCHIVE"
trap - EXIT
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo "CEGR V2 evidence archive: $ARCHIVE"
