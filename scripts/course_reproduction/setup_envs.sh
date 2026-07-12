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
conda install -y -c conda-forge openjdk=21
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install pyserini==0.25.0 faiss-cpu==1.8.0 'numpy<2' 'transformers<4.48' datasets uvicorn fastapi requests
python -m pip check
java -version
python -c "from pyserini.search.lucene import LuceneSearcher; print('Pyserini BM25 runtime ready')"
