# Search-R1 Cloud Reproduction Runbook

This runbook is designed to reduce paid GPU time. Prepare the repository locally first, then rent a Linux GPU host and execute the commands below.

## Recommended Rental Target

- OS: Ubuntu 20.04 or 22.04
- GPU: 8 x A100 40GB or 8 x A100 80GB
- CPU: 32 cores or more
- RAM: 128GB or more
- Disk: 500GB minimum, 1TB preferred
- CUDA: 12.1-compatible image
- Base image: PyTorch 2.4 / CUDA 12.1 if available

For a cheaper first attempt, rent by the hour and run only the smoke test. Stop the machine immediately after collecting the log if the smoke test fails.

## Cost-Saving Workflow

1. Rent the machine only after this repository contains the cloud scripts.
2. Start a by-hour 8 x A100 instance.
3. Run environment installation once.
4. Download data and index once.
5. Run retriever check.
6. Run the 2-step GRPO smoke test.
7. Only if smoke test passes, decide whether to run the full training script.

## SSH Into The Machine

```bash
ssh root@YOUR_SERVER_IP
nvidia-smi
df -h
```

## Clone Repository

```bash
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1
```

If you are using this prepared local copy instead of the public repository, upload it with `scp` or push it to your own private GitHub repository first.

## Install Training Environment

```bash
bash scripts/cloud_setup_searchr1.sh
```

This creates a conda environment named `searchr1` by default.

## Install Retriever Environment

```bash
bash scripts/cloud_setup_retriever.sh
```

This creates a conda environment named `retriever` by default.

## Download Data And Index

```bash
bash scripts/cloud_prepare_data_and_index.sh
```

Expected outputs:

```text
data/wiki18/e5_Flat.index
data/wiki18/wiki-18.jsonl
data/nq_search/train.parquet
data/nq_search/test.parquet
```

This step can take a long time because it downloads the Wikipedia corpus and dense index from Hugging Face.

## Launch Retriever

Use `tmux` so the retriever keeps running:

```bash
tmux new -s retriever
bash scripts/cloud_launch_retriever.sh
```

Leave it running. In another terminal:

```bash
tmux new -s train
conda activate searchr1
python scripts/cloud_check_retriever.py
```

The check should print `retriever ok` and three retrieved documents.

## Run 2-Step Smoke Test

```bash
bash scripts/cloud_train_grpo_smoke.sh
```

The smoke script first calls `scripts/cloud_check_retriever.py`. If the retriever is not alive, it fails before starting expensive training components.

Success criteria:

- Model downloads and loads.
- Ray starts.
- vLLM rollout starts.
- The retriever receives requests.
- The script reaches training steps and writes `nq-search-r1-grpo-qwen2.5-3b-smoke.log`.

If this fails, save the log and stop the paid instance. Do not start full training until the smoke test is fixed.

## Run Full GRPO Training

Only after smoke succeeds:

```bash
wandb login
bash scripts/cloud_train_grpo_full.sh
```

Default model:

```text
Qwen/Qwen2.5-3B
```

Default training scale:

```text
8 GPUs
1005 total training steps
NQ search data
local e5 dense retriever over wiki-18
```

## What To Save For The Report

- `nvidia-smi` screenshot or text output
- conda package versions
- retriever check output
- smoke test log
- full training log if run
- WandB link if enabled
- checkpoint directory under `verl_checkpoints/`

## Stop The Instance

After the smoke test or full run:

```bash
sudo shutdown now
```

Also stop or destroy the instance in the cloud provider console. Some platforms keep charging if the instance is merely disconnected.
