# CEGR V2 Code Layout

This directory contains the single CEGR V2 workflow that produced the final experiment.

## Run order

```text
link_assets.sh
prepare.sh
run_smoke.sh
run_train.sh
run_evaluation.sh
collect_evidence.sh
```

`run_smoke.sh` and `run_train.sh` both call `train.sh`. The smoke run checks the mechanism for two updates; the formal run starts again from the original Qwen model and trains for 120 updates.

## Core method

- `config.py`: frozen model, optimization, seed, evaluation, and success settings.
- `reward.py`: EM-first, F1-fallback group reward.
- `grouping.py`: stable prompt identity for same-question rollouts.
- `manager.py`: group validation, reward assignment, and training metrics.
- `main.py`: veRL/Ray entrypoint without modifying `verl/`.
- `worker.py`, `vllm_seed.py`: deterministic rollout-engine setup.

## Evaluation and analysis

- `evaluate_model.sh`: common model evaluation wrapper.
- `evaluation_record.py`: strict answer parsing and search diagnostics.
- `run_evaluation.sh`: paired 700-example baseline/V2 evaluation.
- `analysis.py`: paired effects, bootstrap intervals, McNemar test, and claim gate.
- `rescore_baseline.py`: strict offline audit of historical baseline records.

## Safety and reproducibility

- `freeze_v1.py`, `link_assets.sh`: preserve and link baseline/CEGR V1 inputs.
- `prepare_pilot_data.py`, `verify_pilot_data.py`: disjoint smoke-validation data.
- `audit_reward_safety.py`: check mixed-group EM preservation and fallback signal.
- `parse_metrics.py`, `verify_training.py`: reject incomplete or non-finite runs.
- `verify_evaluation.py`: enforce trajectory/metric parity.
- `verify_ray_colocation.py`, `token_id_compat.py`: runtime compatibility checks.
- `collect_evidence.sh`: archive code identity, environment, results, and checkpoint hashes.

Names containing `direct120` remain only inside immutable artifact identifiers from the completed server run. They are retained so checkpoints, logs, result JSON, and the evidence archive continue to refer to the same experiment.
