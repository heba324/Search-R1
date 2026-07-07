#!/usr/bin/env bash
set -euo pipefail

# Create the retrieval environment used by the local Search-R1 retriever server.
# Run from the repository root:
#   bash scripts/cloud_setup_retriever.sh

ENV_NAME="${ENV_NAME:-retriever}"

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
python -m pip install transformers datasets pyserini uvicorn fastapi huggingface_hub tqdm

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
PY

echo "Retriever environment is ready: $ENV_NAME"
