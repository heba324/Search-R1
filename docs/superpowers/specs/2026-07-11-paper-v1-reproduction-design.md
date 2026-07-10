# Search-R1 Paper v1 Reproduction Design

## Objective

Reproduce the primary Search-R1 result from arXiv:2503.09516v1 using the author code snapshot `118c6e7361bb68e33c525b50d62f83b63462799e`. This is the smallest post-paper-script correction: it enables EM evaluation for all seven datasets without changing the training recipe. The primary target is Qwen2.5-3B-Instruct with PPO, the merged NQ and HotpotQA training set, outcome-only exact-match reward, four search turns, top-3 E5 retrieval, and 305 training steps.

## Scope

The implementation may add reproducibility wrappers, verification scripts, evidence collection, tests, and Chinese runbooks. It must not modify the v1 Search-R1 training core, reward implementation, retrieval logic, or paper launcher hyperparameters. Later `v0.2` and `v0.3` settings, including 1005 training steps and format rewards, are explicitly out of scope.

## Approach Options

1. Freeze v1 code and add wrappers around it. Recommended because the public commit was added one day after the v1 preprint and contains the paper scripts.
2. Reconstruct the final arXiv v5 setting manually. Rejected for the primary reproduction because the public repository has no matching frozen commit.
3. Use the current repository's latest scripts. Rejected because their later research settings change the scientific claim.

## Architecture

`scripts/paper_v1/` owns only reproducibility infrastructure:

- `preflight.py` verifies the immutable author commit, Linux/GPU resources, required tools, and required local artifacts.
- `prepare_data.sh` fetches the author-published merged training data and validates that its expected parquet files are present before training.
- `train_qwen25_3b_instruct_ppo.sh` invokes `verl.trainer.main_ppo` with the v1 launcher settings while exposing only paths, logging, and the model identifier as shell variables.
- `evaluate_paper_v1.sh` runs the author evaluation entry point against the trained checkpoint and writes an unambiguous experiment record.
- `collect_evidence.sh` stores immutable Git, environment, data, hardware, launcher, logs, checkpoint, and metric evidence.

The wrappers always reject a repository that is not the exact author commit. They record the target table row: average EM `0.327` for Qwen2.5-3B-Instruct with PPO in arXiv v1 Table 2. A successful training process is not treated as a successful paper reproduction until the seven-dataset evaluation record exists.

## Data Flow

1. Clone this branch and verify `118c6e7`.
2. Build the Search-R1 and retriever Conda environments.
3. Download the author merged NQ+HotpotQA parquet data, then build/download the E5 Wikipedia retrieval resources.
4. Start the v1 retriever with the full Wikipedia corpus and top-k 3.
5. Train Qwen2.5-3B-Instruct using PPO for 305 steps and max turns 4.
6. Evaluate the seven benchmark datasets with EM and collect evidence.

## Failure Handling

Preflight failure, missing artifacts, a stopped retriever, wrong Git commit, a wrong launcher setting, or a nonzero train/evaluate command prevents a completion marker. Evidence is collected after both failure and success. The runbook tells the user to stop the paid instance immediately after evidence collection.

## Verification

Static `unittest` tests assert the exact commit, model, PPO algorithm, 305 steps, EM-only reward, four turns, top-k 3, NQ+HotpotQA data identifier, and seven-dataset target. Shell syntax and Python compilation checks cover all added wrappers. A paid GPU run remains an external validation step and must be reported separately from local checks.
