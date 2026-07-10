#!/usr/bin/env bash
set -euo pipefail

# Create the retrieval environment used by the local Search-R1 retriever server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-Search-R1-retriever}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available. Choose a cloud image with Anaconda/Miniconda installed." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists; reusing it."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
python -m pip install --upgrade pip
python -m pip install "numpy<2" "transformers<4.48" "datasets<4" requests uvicorn fastapi huggingface_hub tqdm
python -m pip check

python - <<'PY'
import torch
import faiss
import datasets
import fastapi

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("faiss gpu resources:", hasattr(faiss, "StandardGpuResources"))
print("datasets:", datasets.__version__)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the retriever environment.")
if not hasattr(faiss, "StandardGpuResources"):
    raise SystemExit("The installed FAISS build does not provide GPU support.")
PY

echo "Retriever environment is ready: $ENV_NAME"
