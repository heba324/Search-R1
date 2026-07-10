#!/usr/bin/env bash
set -euo pipefail

# Create the training environment used by Search-R1 on a rented Linux GPU host.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-Search-R1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Choose a cloud image with Anaconda/Miniconda installed." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists; reusing it."
else
  conda create -n "$ENV_NAME" python=3.9 -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install ninja packaging
python -m pip install vllm==0.6.3
python -m pip install -e .
python -m pip install "numpy<2" "tensordict<0.6" "transformers<4.48" "datasets<4" requests
python -m pip install flash-attn --no-build-isolation
python -m pip install wandb huggingface_hub
python -m pip check

python - <<'PY'
import torch
import transformers
import verl

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("verl:", verl.__version__)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the Search-R1 environment.")
PY

echo "Search-R1 training environment is ready: $ENV_NAME"
