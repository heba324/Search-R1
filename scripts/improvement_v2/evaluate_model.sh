#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
SEED="${SEED:-42}"
ROLLOUT_ENGINE_SEED="${ROLLOUT_ENGINE_SEED:-42}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:?EVAL_RUN_NAME is required}"
EVAL_DATA="${EVAL_DATA:?EVAL_DATA is required}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:?EVAL_BATCH_SIZE is required}"
TRAJECTORIES_PATH="${TRAJECTORIES_PATH:?TRAJECTORIES_PATH is required}"
ARTIFACT_DIR="$REPO_ROOT/artifacts/improvement-v2/evaluation/$EVAL_RUN_NAME"
LOG_FILE="$ARTIFACT_DIR/evaluation.log"
MARKER="$ARTIFACT_DIR/evaluation_completed.json"
PARITY="$ARTIFACT_DIR/record-parity.json"

[ -s "$EVAL_DATA" ] || { echo "Missing evaluation parquet: $EVAL_DATA" >&2; exit 1; }
[ -s "$MODEL_PATH/config.json" ] || { echo "Missing actor checkpoint: $MODEL_PATH" >&2; exit 1; }
if [ -s "$MARKER" ] && [ -s "$TRAJECTORIES_PATH" ] && [ -s "$PARITY" ]; then
  python3 - "$MARKER" "$EVAL_DATA" "$MODEL_PATH" "$SEED" "$ROLLOUT_ENGINE_SEED" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker_path, eval_data, model_path = map(Path, sys.argv[1:4])
driver_seed, engine_seed = map(int, sys.argv[4:6])
marker = json.loads(marker_path.read_text(encoding="utf-8"))
digest = hashlib.sha256(eval_data.read_bytes()).hexdigest()
if Path(marker["evaluation_data_path"]) != eval_data.resolve():
    raise SystemExit("Completed evaluation points to different data")
if Path(marker["model_path"]) != model_path.resolve():
    raise SystemExit("Completed evaluation points to a different model")
if marker["evaluation_data_sha256"] != digest:
    raise SystemExit("Completed evaluation data SHA-256 no longer matches")
if marker.get("driver_seed") != driver_seed:
    raise SystemExit("Completed evaluation uses a different driver seed")
if marker.get("rollout_engine_seed") != engine_seed:
    raise SystemExit("Completed evaluation uses a different vLLM engine seed")
PY
  TEMP_PARITY="$(mktemp)"
  trap 'rm -f "$TEMP_PARITY"' EXIT
  python3 scripts/improvement_v2/verify_evaluation_records.py \
    "$MARKER" "$TRAJECTORIES_PATH" "$TEMP_PARITY"
  rm -f "$TEMP_PARITY"
  trap - EXIT
  echo "Already completed; preserving evaluation: $EVAL_RUN_NAME"
  exit 0
fi
if [ -e "$ARTIFACT_DIR" ] || [ -e "$TRAJECTORIES_PATH" ]; then
  echo "Refusing to overwrite a partial evaluation: $EVAL_RUN_NAME" >&2
  exit 1
fi
mkdir -p "$ARTIFACT_DIR" "$(dirname "$TRAJECTORIES_PATH")"
python3 scripts/course_reproduction/preflight.py --repo-root "$REPO_ROOT" --require-assets
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SEARCH_ENV"
python scripts/course_reproduction/check_retriever.py

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"
export SEARCH_R1_EVAL_TRAJECTORIES="$TRAJECTORIES_PATH"
START_TIME="$(date +%s)"

python3 -m scripts.improvement_v2.main_ppo_refinement \
  data.train_files="$EVAL_DATA" data.val_files="$EVAL_DATA" \
  data.train_data_num=null data.val_data_num=null \
  data.train_batch_size="$EVAL_BATCH_SIZE" data.val_batch_size="$EVAL_BATCH_SIZE" \
  data.max_prompt_length=4096 data.max_response_length=500 \
  data.max_start_length=2048 data.max_obs_length=500 \
  algorithm.adv_estimator=grpo actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$EVAL_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  +actor_rollout_ref.rollout.engine_seed="$ROLLOUT_ENGINE_SEED" \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true algorithm.no_think_rl=false \
  +reward_strategy.name=grouped_em +reward_strategy.group_size=5 \
  +reward_strategy.seed="$SEED" \
  "trainer.logger=['console']" +trainer.val_only=true +trainer.val_before_train=true \
  trainer.default_hdfs_dir=null trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.project_name=Search-R1-course trainer.experiment_name="cegr-v2-eval-$EVAL_RUN_NAME" \
  max_turns=4 retriever.url="http://127.0.0.1:8000/retrieve" retriever.topk=3 \
  2>&1 | tee "$LOG_FILE"

ELAPSED="$(( $(date +%s) - START_TIME ))"
python3 scripts/course_reproduction/parse_eval_metrics.py \
  "$LOG_FILE" "$MARKER.tmp" --run-name "$EVAL_RUN_NAME" \
  --elapsed-seconds "$ELAPSED" --eval-data "$EVAL_DATA" --model-path "$MODEL_PATH"
python3 - "$MARKER.tmp" "$SEED" "$ROLLOUT_ENGINE_SEED" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["driver_seed"] = int(sys.argv[2])
payload["rollout_engine_seed"] = int(sys.argv[3])
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
[ -s "$TRAJECTORIES_PATH" ] || { echo "Evaluation produced no trajectory records" >&2; exit 1; }
python3 scripts/improvement_v2/verify_evaluation_records.py \
  "$MARKER.tmp" "$TRAJECTORIES_PATH" "$PARITY"
mv "$MARKER.tmp" "$MARKER"
echo "CEGR V2 evaluation completed: $MARKER"
