#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$REPO_ROOT/artifacts/course-reproduction/evidence"
mkdir -p "$OUT"

git -C "$REPO_ROOT" rev-parse HEAD > "$OUT/git-head.txt"
git -C "$REPO_ROOT" status --short --branch > "$OUT/git-status.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi topo -m > "$OUT/nvidia-topology.txt"
free -h > "$OUT/memory.txt"
df -h "$REPO_ROOT" > "$OUT/disk.txt"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SEARCH_ENV:-Search-R1}"
conda list > "$OUT/search-env-conda-list.txt"
python -m pip freeze > "$OUT/search-env-pip-freeze.txt"
conda activate "${RETRIEVER_ENV:-Search-R1-retriever}"
conda list > "$OUT/retriever-env-conda-list.txt"
java -version 2> "$OUT/java-version.txt"

find "$REPO_ROOT/artifacts/course-reproduction" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/artifacts.sha256"
tar --exclude='evidence/course-reproduction-evidence.tar.gz' -czf "$OUT/course-reproduction-evidence.tar.gz" \
  -C "$REPO_ROOT" artifacts/course-reproduction docs/course_reproduction_zh.md docs/research/resource_limited_reproduction.md
echo "Evidence archive: $OUT/course-reproduction-evidence.tar.gz"
