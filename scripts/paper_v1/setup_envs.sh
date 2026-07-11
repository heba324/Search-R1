#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SEARCH_ENV="${SEARCH_ENV:-Search-R1}"
RETRIEVER_ENV="${RETRIEVER_ENV:-Search-R1-retriever}"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fqx "$SEARCH_ENV"; then
  conda create -y -n "$SEARCH_ENV" python=3.9
fi
conda activate "$SEARCH_ENV"
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install vllm==0.6.3
python -m pip install -e "$REPO_ROOT"
python -m pip install --no-build-isolation flash-attn
python -m pip install 'numpy<2' 'tensordict<0.6' 'transformers<4.48' datasets huggingface_hub wandb requests
python -m pip check
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"

if ! conda env list | awk '{print $1}' | grep -Fqx "$RETRIEVER_ENV"; then
  conda create -y -n "$RETRIEVER_ENV" python=3.10
fi
conda activate "$RETRIEVER_ENV"
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y faiss-gpu=1.8.0 -c pytorch -c nvidia
python -m pip install 'numpy<2' 'transformers<4.48' datasets huggingface_hub uvicorn fastapi requests
python -m pip check
python -c "import faiss, torch; assert hasattr(faiss, 'StandardGpuResources'); assert torch.cuda.is_available()"
