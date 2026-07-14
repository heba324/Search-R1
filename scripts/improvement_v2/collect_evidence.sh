#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
eval "$(python3 "$SCRIPT_DIR/direct120_contract.py" --shell)"
OUT="$REPO_ROOT/artifacts/improvement-v2/evidence"
ARCHIVE="$OUT/search-r1-cegr-v2-evidence.tar.gz"
REQUIRE_DIRECT120_CHECKPOINT="${REQUIRE_DIRECT120_CHECKPOINT:-false}"
DIRECT120_CHECKPOINT_REL="verl_checkpoints/$DIRECT120_TRAINING_RUN/actor/global_step_$DIRECT120_TRAINING_STEPS"
DIRECT120_CHECKPOINT="$REPO_ROOT/$DIRECT120_CHECKPOINT_REL"
mkdir -p "$OUT"
rm -f "$ARCHIVE" "$ARCHIVE.sha256"
rm -f "$OUT/direct120-checkpoint.sha256" \
  "$OUT/direct120-checkpoint-not-included.txt"

case "$REQUIRE_DIRECT120_CHECKPOINT" in
  true|false) ;;
  *) echo "REQUIRE_DIRECT120_CHECKPOINT must be true or false" >&2; exit 1 ;;
esac
if [ "$REQUIRE_DIRECT120_CHECKPOINT" = true ] && \
   [ ! -s "$DIRECT120_CHECKPOINT/config.json" ]; then
  echo "Missing required Direct120 checkpoint: $DIRECT120_CHECKPOINT" >&2
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
  docs/cegr_v2_experiment_zh.md
  docs/cegr_v2_direct120_urgent_zh.md
  docs/research/cegr_v2_literature_review.md
)
if [ -s "$DIRECT120_CHECKPOINT/config.json" ]; then
  (
    cd "$REPO_ROOT"
    find "$DIRECT120_CHECKPOINT_REL" -type f -print0 | \
      sort -z | xargs -0 sha256sum
  ) > "$OUT/direct120-checkpoint.sha256"
  TAR_INPUTS+=("$DIRECT120_CHECKPOINT_REL")
  echo "Direct120 checkpoint will be included: $DIRECT120_CHECKPOINT_REL"
else
  printf 'Direct120 checkpoint was not present; standard V2 evidence only.\n' \
    > "$OUT/direct120-checkpoint-not-included.txt"
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
