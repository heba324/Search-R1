# Search-R1 Cloud Reproduction Runbook

This runbook separates a low-cost startup validation from the official-scale run. Local tests, a smoke run, and a completed full experiment are different evidence levels.

## Provenance

- Prepared repository: `https://github.com/heba324/Search-R1`
- Upstream repository: `https://github.com/PeterGriffinJin/Search-R1`
- Recorded upstream baseline: `598e61b`
- Training environment: `Search-R1`
- Retriever environment: `Search-R1-retriever`

Cloud helpers are kept in `scripts/cloud_*`. Upstream training, reward, retrieval, and data-processing implementations are left unchanged.

## Rental Profiles

Smoke profile:

- Ubuntu 20.04 or 22.04
- 1 x A100 80GB
- 64GB RAM minimum
- 150GB total disk minimum, 200GB recommended; preflight requires 100 GiB free
- Hourly billing

Full profile:

- Ubuntu 20.04 or 22.04
- 8 x A100 40GB or 8 x A100 80GB
- 128GB RAM minimum, 256GB recommended
- 500GB disk minimum, 1TB recommended
- Hourly billing

The smoke stage builds a tiny E5 index from `example/corpus.jsonl`. It does not download the full Wikipedia index and cannot produce paper accuracy results.

## Connect And Clone

```bash
ssh root@YOUR_SERVER_IP
git clone https://github.com/heba324/Search-R1.git
cd Search-R1
git rev-parse HEAD
```

## Stage A: Single-GPU Smoke

Check the host before installing the Search-R1 project environments. The selected cloud image must already provide Conda:

```bash
python3 scripts/cloud_preflight.py --profile smoke
```

Install the two environments:

```bash
bash scripts/cloud_setup_searchr1.sh 2>&1 | tee setup-Search-R1.log
bash scripts/cloud_setup_retriever.sh 2>&1 | tee setup-retriever.log
```

Prepare NQ and a tiny retriever index:

```bash
bash scripts/cloud_prepare_smoke_assets.sh 2>&1 | tee prepare-smoke.log
```

Start the smoke retriever in tmux:

```bash
tmux new -s retriever
cd ~/Search-R1
bash scripts/cloud_launch_retriever.sh 2>&1 | tee retriever-smoke.log
```

Detach with `Ctrl+B`, then `D`. In another shell:

```bash
cd ~/Search-R1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python scripts/cloud_check_retriever.py
bash scripts/cloud_train_grpo_smoke.sh
bash scripts/cloud_collect_evidence.sh smoke
```

Smoke success means that the 2-step training path ran and created `artifacts/smoke_passed.txt`. It is not a paper reproduction result. Download that attestation with the evidence bundle. If smoke fails, collect evidence and stop the paid instance.

## Stage B: Eight-GPU Full Run

Start a new full-profile instance and repeat clone and environment setup. Upload the smoke attestation from Stage A before continuing:

```bash
mkdir -p ~/Search-R1/artifacts
cp /path/to/uploaded/smoke_passed.txt ~/Search-R1/artifacts/smoke_passed.txt
```

The attestation Git commit must match the full instance. Then run:

```bash
python3 scripts/cloud_preflight.py --profile full
bash scripts/cloud_prepare_data_and_index.sh 2>&1 | tee prepare-full.log
```

Start the full Wikipedia retriever:

```bash
tmux new -s retriever
cd ~/Search-R1
ASSET_PROFILE=full bash scripts/cloud_launch_retriever.sh 2>&1 | tee retriever-full.log
```

Detach and verify the API:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
python scripts/cloud_check_retriever.py
grep '^profile=' artifacts/retriever_profile.txt
```

The output must be `profile=full`. The official single-node configuration intentionally makes the FAISS retriever and training share all eight GPUs: the float16 index is sharded while training still uses GPUs 0-7. Preserve this for official-scale reproduction, monitor memory, and record any deviation. Start the expensive run only with explicit confirmation:

```bash
CONFIRM_FULL_RUN=YES bash scripts/cloud_train_grpo_full.sh
```

WandB is optional and disabled by default. To enable it:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Search-R1
wandb login
CONFIRM_FULL_RUN=YES TRAINER_LOGGER=wandb bash scripts/cloud_train_grpo_full.sh
```

Collect final evidence:

```bash
bash scripts/cloud_collect_evidence.sh full
```

## Full-Run Defaults

```text
Model: Qwen/Qwen2.5-3B base
Algorithm: GRPO
Dataset: NQ
Retriever: E5 dense retrieval over Wikipedia 2018
GPUs: 8
Training steps: 1005
Checkpoint interval: 100 steps
Validation interval: 50 steps
```

## Evidence Boundary

Local readiness requires automated tests and syntax checks. Cloud smoke evidence requires retriever output and a completed 2-step log. A full reproduction claim additionally requires the hardware inventory, exact Git commit, explicit Conda package lists, data hashes, full logs, checkpoints, evaluation metrics, and comparison with the paper.

## Stop Billing

After a failure or completion, download the evidence and stop or destroy the instance in the provider console. Disconnecting SSH does not stop billing.
