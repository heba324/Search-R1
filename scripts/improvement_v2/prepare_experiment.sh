#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

mkdir -p artifacts/improvement-v2/preflight
git rev-parse HEAD > artifacts/improvement-v2/preflight/git-head.txt
git status --short --branch > artifacts/improvement-v2/preflight/git-status.txt

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SEARCH_ENV:-Search-R1}"

python3 -m scripts.improvement_v2.freeze_v1 --repo-root "$REPO_ROOT" --initialize
python3 -m scripts.improvement_v2.freeze_v1 --repo-root "$REPO_ROOT"
python3 -m scripts.improvement_v2.prepare_pilot_data
python3 -m scripts.improvement_v2.verify_pilot_data
python3 -m scripts.improvement_v2.verify_ray_colocation
python3 -m unittest discover -s tests -v 2>&1 | \
  tee artifacts/improvement-v2/preflight/unittest.log
python3 -m compileall -q scripts/improvement_v2

for script in scripts/improvement_v2/*.sh; do
  bash -n "$script"
done

git diff --name-only 8672aad0f4089f0fca388601cd9ce20fc9b8b776 -- \
  search_r1 verl > artifacts/improvement-v2/preflight/core-diff.txt
[ ! -s artifacts/improvement-v2/preflight/core-diff.txt ] || {
  echo "V2 unexpectedly changes search_r1/ or verl/." >&2
  exit 1
}
git diff --name-only 8672aad0f4089f0fca388601cd9ce20fc9b8b776 -- scripts/improvement docs/improvement_experiment_zh.md \
  > artifacts/improvement-v2/preflight/v1-code-diff.txt
[ ! -s artifacts/improvement-v2/preflight/v1-code-diff.txt ] || {
  echo "V2 unexpectedly changes frozen CEGR V1 code or documentation." >&2
  exit 1
}

echo "CEGR V2 offline preparation passed. Next: bash scripts/improvement_v2/run_smoke.sh"
