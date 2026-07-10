#!/usr/bin/env bash
set -euo pipefail

# Run a tiny GRPO smoke test before spending money on a full run.
# This verifies model loading, Ray/FSDP/vLLM wiring, retrieval calls, reward code,
# and checkpoint/log directory creation with only a few training steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/nq_search}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-nq-search-r1-grpo-qwen2.5-3b-smoke}"
WAND_PROJECT="${WAND_PROJECT:-Search-R1-smoke}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE_MARKER="$REPO_ROOT/artifacts/retriever_profile.txt"
SMOKE_ATTESTATION="$REPO_ROOT/artifacts/smoke_passed.txt"
ATTESTATION_TMP="$REPO_ROOT/artifacts/smoke_passed.txt.tmp"

mkdir -p "$REPO_ROOT/artifacts"
rm -f "$SMOKE_ATTESTATION" "$ATTESTATION_TMP"
"$PYTHON_BIN" scripts/cloud_preflight.py --profile smoke

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "Missing train parquet: $DATA_DIR/train.parquet" >&2
  exit 1
fi

if [ ! -f "$DATA_DIR/test.parquet" ]; then
  echo "Missing test parquet: $DATA_DIR/test.parquet" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$SEARCH_ENV"

current_commit="$(git rev-parse HEAD)"
if [ ! -s "$PROFILE_MARKER" ] \
    || ! grep -Fqx "profile=smoke" "$PROFILE_MARKER" \
    || ! grep -Fqx "git_commit=$current_commit" "$PROFILE_MARKER" \
    || ! grep -Fqx "index_file=$REPO_ROOT/data/smoke_retriever/e5_Flat.index" "$PROFILE_MARKER" \
    || ! grep -Fqx "corpus_file=$REPO_ROOT/example/corpus.jsonl" "$PROFILE_MARKER"; then
  echo "Smoke training requires a ready smoke retriever from the same Git commit." >&2
  echo "Start it with: bash scripts/cloud_launch_retriever.sh" >&2
  exit 1
fi

retriever_pid="$(sed -n 's/^pid=//p' "$PROFILE_MARKER" | head -n 1)"
case "$retriever_pid" in
  ''|*[!0-9]*)
    echo "Retriever profile contains an invalid PID." >&2
    exit 1
    ;;
esac
if ! kill -0 "$retriever_pid" >/dev/null 2>&1; then
  echo "Retriever profile PID is no longer running: $retriever_pid" >&2
  exit 1
fi

python scripts/cloud_check_retriever.py

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/test.parquet" \
  data.train_data_num=8 \
  data.val_data_num=8 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  data.max_prompt_length=2048 \
  data.max_response_length=128 \
  data.max_start_length=1024 \
  data.max_obs_length=256 \
  data.shuffle_train_dataloader=True \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false \
  actor_rollout_ref.rollout.n_agent=2 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.actor.state_masking=true \
  "trainer.logger=['console']" \
  +trainer.val_only=false \
  +trainer.val_before_train=false \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node="$GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.project_name="$WAND_PROJECT" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
  max_turns=2 \
  retriever.url="http://127.0.0.1:8000/retrieve" \
  retriever.topk=3 \
  2>&1 | tee "$EXPERIMENT_NAME.log"

cat > "$ATTESTATION_TMP" <<EOF
status=passed
mode=smoke
git_commit=$current_commit
experiment_name=$EXPERIMENT_NAME
training_steps=2
completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
retriever_profile_sha256=$(sha256sum "$PROFILE_MARKER" | awk '{print $1}')
EOF
mv "$ATTESTATION_TMP" "$SMOKE_ATTESTATION"
echo "Smoke success attestation: $SMOKE_ATTESTATION"
