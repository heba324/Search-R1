#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
V1_ROOT="${V1_ROOT:-${1:-}}"
V1_COMMIT="8672aad0f4089f0fca388601cd9ce20fc9b8b776"

[ -n "$V1_ROOT" ] || {
  echo "Usage: V1_ROOT=/path/to/v1-repo bash $0" >&2
  exit 1
}
V1_ROOT="$(cd "$V1_ROOT" && pwd)"
[ "$V1_ROOT" != "$REPO_ROOT" ] || {
  echo "V1_ROOT must be the completed V1 checkout, not the V2 worktree." >&2
  exit 1
}
V1_HEAD="$(git -C "$V1_ROOT" rev-parse HEAD)" || {
  echo "V1_ROOT is not a Git checkout: $V1_ROOT" >&2
  exit 1
}
[ "$V1_HEAD" = "$V1_COMMIT" ] || {
  echo "V1_ROOT must be frozen at $V1_COMMIT, found $V1_HEAD" >&2
  exit 1
}
git -C "$V1_ROOT" diff --quiet || {
  echo "V1_ROOT has modified tracked files; archive or restore them first." >&2
  exit 1
}
git -C "$V1_ROOT" diff --cached --quiet || {
  echo "V1_ROOT has staged tracked changes; archive or restore them first." >&2
  exit 1
}

link_once() {
  local source="$1"
  local target="$2"
  [ -e "$source" ] || { echo "Missing V1 asset: $source" >&2; exit 1; }
  if [ -L "$target" ]; then
    [ "$(readlink -f "$target")" = "$(readlink -f "$source")" ] || {
      echo "Existing symlink points elsewhere: $target" >&2
      exit 1
    }
    return
  fi
  [ ! -e "$target" ] || {
    echo "Refusing to replace an existing V2 path: $target" >&2
    exit 1
  }
  ln -s "$source" "$target"
}

mkdir -p "$REPO_ROOT/data" "$REPO_ROOT/verl_checkpoints" "$REPO_ROOT/artifacts"
mkdir -p "$REPO_ROOT/artifacts/course-reproduction/evaluation"
for data_path in nq_hotpotqa_train models course_eval wiki18_bm25; do
  link_once "$V1_ROOT/data/$data_path" "$REPO_ROOT/data/$data_path"
done

for run_name in \
  search-r1-course-qwen2.5-1.5b-grpo-bm25 \
  search-r1-cegr-qwen2.5-1.5b-grpo-bm25
do
  link_once \
    "$V1_ROOT/verl_checkpoints/$run_name" \
    "$REPO_ROOT/verl_checkpoints/$run_name"
done

link_once \
  "$V1_ROOT/artifacts/improvement" \
  "$REPO_ROOT/artifacts/improvement"

for evaluation_name in baseline-paired cegr-post-rl; do
  link_once \
    "$V1_ROOT/artifacts/course-reproduction/evaluation/$evaluation_name" \
    "$REPO_ROOT/artifacts/course-reproduction/evaluation/$evaluation_name"
done

echo "V1 assets linked under an immutable hash contract; SHA verification is required."
echo "V2 worktree: $REPO_ROOT"
echo "V1 source:    $V1_ROOT"
