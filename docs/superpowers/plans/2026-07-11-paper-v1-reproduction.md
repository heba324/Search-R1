# Search-R1 Paper v1 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the author-compatible Search-R1 v1 PPO reproduction runnable and auditable before renting a GPU host.

**Architecture:** Keep commit `118c6e7` as the immutable Search-R1 core. Add a small Python contract module for pure checks, shell wrappers that call the unchanged v1 training entry point with fixed values, and a Chinese runbook that distinguishes local validation, paid smoke validation, and the final paper evaluation.

**Tech Stack:** Python 3.9 standard library, Bash, `unittest`, Conda, Hugging Face CLI, veRL, FAISS, vLLM.

---

### Task 1: Lock the scientific contract

**Files:**
- Create: `scripts/paper_v1/contract.py`
- Create: `tests/test_paper_v1_contract.py`

- [ ] **Step 1: Write failing tests for the immutable v1 constants**

```python
from scripts.paper_v1.contract import PAPER_V1

self.assertEqual(PAPER_V1.git_commit, "118c6e7361bb68e33c525b50d62f83b63462799e")
self.assertEqual(PAPER_V1.training_steps, 305)
self.assertEqual(PAPER_V1.max_turns, 4)
self.assertEqual(PAPER_V1.target_average_em, 0.327)
```

- [ ] **Step 2: Verify the test fails because the module is absent**

Run: `python -m unittest tests.test_paper_v1_contract -v`

Expected: import error for `scripts.paper_v1.contract`.

- [ ] **Step 3: Implement the frozen constants and pure validation functions**

The implementation must reject an incorrect Git commit, missing `train.parquet` or `test.parquet`, and an incomplete seven-dataset result dictionary.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m unittest tests.test_paper_v1_contract -v`

### Task 2: Build a pre-rental host and artifact gate

**Files:**
- Create: `scripts/paper_v1/preflight.py`
- Modify: `tests/test_paper_v1_contract.py`

- [ ] **Step 1: Write failing tests for wrong-commit and incomplete-evaluation messages**

- [ ] **Step 2: Implement a standard-library CLI that checks Linux, exact Git commit, 8 visible GPUs, 128 GiB RAM, 500 GiB free disk, required commands, train/test parquet, full E5 index, and Wikipedia corpus**

- [ ] **Step 3: Run `python scripts/paper_v1/preflight.py --help` and the focused tests**

### Task 3: Add exact v1 training and evaluation wrappers

**Files:**
- Create: `scripts/paper_v1/prepare_train_data.sh`
- Create: `scripts/paper_v1/train_qwen25_3b_instruct_ppo.sh`
- Create: `scripts/paper_v1/evaluate_qwen25_3b_instruct_ppo.sh`
- Modify: `tests/test_paper_v1_contract.py`

- [ ] **Step 1: Write failing shell-contract tests**

Assert the training wrapper contains `algorithm.adv_estimator=gae`, `Qwen/Qwen2.5-3B-Instruct`, `trainer.total_training_steps=305`, `max_turns=4`, `retriever.topk=3`, and no format reward setting.

- [ ] **Step 2: Implement wrappers that preserve v1 training values**

The data wrapper downloads `PeterJinGo/nq_hotpotqa_train` at revision `b7d80abfee334a7a91cb377544f09180d58b34f6`, then verifies the published parquet byte sizes and SHA-256 values. The training wrapper writes a completion marker only after a zero exit status. The evaluation wrapper runs the unchanged `verl.trainer.main_ppo` in val-only mode against a supplied seven-dataset parquet and records per-dataset EM lines.

- [ ] **Step 3: Run Bash syntax checks and focused tests**

Run: `bash -n scripts/paper_v1/*.sh` and `python -m unittest tests.test_paper_v1_contract -v`.

### Task 4: Document the exact paper-v1 workflow and evidence

**Files:**
- Create: `docs/paper_v1_reproduction_zh.md`
- Create: `scripts/paper_v1/collect_evidence.sh`
- Modify: `tests/test_paper_v1_contract.py`

- [ ] **Step 1: Write failing documentation/evidence contract tests**

Assert the runbook names the v1 paper, `118c6e7`, Qwen2.5-3B-Instruct, PPO, 305 steps, EM-only reward, the seven datasets, and the target average EM of 0.327.

- [ ] **Step 2: Implement the Chinese beginner runbook and evidence collector**

The runbook must instruct the user to stop the cloud instance after collecting evidence and must not call a smoke run a paper result.

- [ ] **Step 3: Run all verification commands**

Run: `python -m unittest discover -s tests -v`, `bash -n scripts/paper_v1/*.sh`, `python -m compileall -q scripts tests search_r1 verl`, and `git diff --check`.

### Task 5: Review and integrate

**Files:**
- Verify: `docs/superpowers/specs/2026-07-11-paper-v1-reproduction-design.md`
- Verify: `docs/superpowers/plans/2026-07-11-paper-v1-reproduction.md`
- Verify: `scripts/paper_v1/`
- Verify: `tests/test_paper_v1_contract.py`

- [ ] **Step 1: Inspect all diffs against `118c6e7` and verify no Search-R1 core file changed**

Run: `git diff --exit-code 118c6e7 -- search_r1 verl scripts/nq_hotpotqa`.

- [ ] **Step 2: Commit the wrappers, tests, and runbook only after all checks pass**

```bash
git add docs scripts/paper_v1 tests/test_paper_v1_contract.py
git commit -m "feat: add paper v1 reproduction workflow"
```
