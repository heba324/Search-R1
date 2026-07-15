#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
eval "$(python3 "$SCRIPT_DIR/config.py" --shell)"
OUT="$REPO_ROOT/artifacts/improvement-v2/evidence"
ARCHIVE="$OUT/search-r1-cegr-v2-evidence.tar.gz"
REQUIRE_FINAL_CHECKPOINT="${REQUIRE_FINAL_CHECKPOINT:-false}"
FINAL_CHECKPOINT_REL="verl_checkpoints/$CEGR_V2_TRAINING_RUN/actor/global_step_$CEGR_V2_TRAINING_STEPS"
FINAL_CHECKPOINT="$REPO_ROOT/$FINAL_CHECKPOINT_REL"
mkdir -p "$OUT"
rm -f "$ARCHIVE" "$ARCHIVE.sha256"
rm -f "$OUT/final-checkpoint.sha256" \
  "$OUT/final-checkpoint-not-included.txt"

case "$REQUIRE_FINAL_CHECKPOINT" in
  true|false) ;;
  *) echo "REQUIRE_FINAL_CHECKPOINT must be true or false" >&2; exit 1 ;;
esac
if [ "$REQUIRE_FINAL_CHECKPOINT" = true ] && \
   [ ! -s "$FINAL_CHECKPOINT/config.json" ]; then
  echo "Missing required CEGR V2 checkpoint: $FINAL_CHECKPOINT" >&2
  exit 1
fi

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

TAR_INPUTS=(
  artifacts/improvement-v2
  data/improvement_v2/pilot_manifest.json
  scripts/improvement_v2
  docs/cegr_v2_experiment_zh.md
  docs/research/cegr_v2_literature_review.md
  docs/research/cegr_v2_results_zh.md
)
if [ -s "$FINAL_CHECKPOINT/config.json" ]; then
  (
    cd "$REPO_ROOT"
    find "$FINAL_CHECKPOINT_REL" -type f -print0 | \
      sort -z | xargs -0 sha256sum
  ) > "$OUT/final-checkpoint.sha256"
  TAR_INPUTS+=("$FINAL_CHECKPOINT_REL")
  echo "CEGR V2 checkpoint will be included: $FINAL_CHECKPOINT_REL"
else
  printf 'CEGR V2 checkpoint was not present; evidence excludes model weights.\n' \
    > "$OUT/final-checkpoint-not-included.txt"
fi

(
  cd "$REPO_ROOT"
  find artifacts/improvement-v2 -path artifacts/improvement-v2/evidence \
    -prune -o -type f -print0 | sort -z | xargs -0 sha256sum
) > "$OUT/artifacts.sha256"

TMP_ARCHIVE="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "$TMP_ARCHIVE"' EXIT
tar --exclude='artifacts/improvement-v2/evidence/search-r1-cegr-v2-evidence.tar.gz*' \
  -czf "$TMP_ARCHIVE" -C "$REPO_ROOT" "${TAR_INPUTS[@]}"
mv "$TMP_ARCHIVE" "$ARCHIVE"
trap - EXIT
(
  cd "$OUT"
  sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256"
)
echo "CEGR V2 evidence archive: $ARCHIVE"
