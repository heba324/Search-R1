# Search-R1 Cloud Reproduction Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally verify a cost-controlled cloud reproduction workflow before renting Search-R1 GPU instances.

**Architecture:** Keep upstream Search-R1 source unchanged and place all rental-specific behavior in `scripts/cloud_*`. Use a testable Python preflight core for hardware decisions, thin strict-mode shell launchers for cloud operations, and standard-library tests that run on Windows without CUDA.

**Tech Stack:** Python 3.9+, `unittest`, Bash, Conda, PyTorch 2.4.0, vLLM 0.6.3, FAISS-GPU 1.8.0, Hugging Face Hub.

---

### Task 1: Testable Cloud Hardware Preflight

**Files:**
- Create: `scripts/cloud_preflight.py`
- Create: `tests/test_cloud_preflight.py`

- [ ] **Step 1: Write failing hardware-policy tests**

```python
import unittest

from scripts.cloud_preflight import HardwareInfo, assess_hardware, parse_gpu_memory


class CloudPreflightTests(unittest.TestCase):
    def test_parse_gpu_memory(self):
        self.assertEqual(parse_gpu_memory("81920\n81920\n"), (81920, 81920))

    def test_smoke_accepts_one_a100_80gb(self):
        info = HardwareInfo((81920,), 128, 300)
        self.assertEqual(assess_hardware("smoke", info), [])

    def test_smoke_rejects_small_gpu(self):
        info = HardwareInfo((24576,), 128, 300)
        self.assertIn("at least 75 GiB VRAM", "\n".join(assess_hardware("smoke", info)))

    def test_full_requires_eight_40gb_gpus(self):
        info = HardwareInfo((40960,) * 4, 256, 800)
        self.assertIn("at least 8 NVIDIA GPUs", "\n".join(assess_hardware("full", info)))

    def test_full_checks_ram_and_disk(self):
        info = HardwareInfo((40960,) * 8, 100, 400)
        errors = "\n".join(assess_hardware("full", info))
        self.assertIn("128 GiB RAM", errors)
        self.assertIn("500 GiB free disk", errors)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_cloud_preflight -v`

Expected: import failure because `scripts.cloud_preflight` does not exist.

- [ ] **Step 3: Implement the preflight core and CLI**

Create an immutable `HardwareInfo` dataclass, `parse_gpu_memory()`, and `assess_hardware()` with these policies:

```python
POLICIES = {
    "smoke": {"gpus": 1, "vram_mib": 75 * 1024, "ram_gib": 64, "disk_gib": 100},
    "full": {"gpus": 8, "vram_mib": 40 * 1024, "ram_gib": 128, "disk_gib": 500},
}
```

The CLI must collect GPU memory through `nvidia-smi`, RAM through `/proc/meminfo`, disk through `shutil.disk_usage`, verify Linux, Conda, Git, the repository root, and optional HTTPS connectivity to `github.com`, `huggingface.co`, and `download.pytorch.org`. It exits `0` only when all hard checks pass and prints that smoke approval is not paper reproduction approval.

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `python -m unittest tests.test_cloud_preflight -v`

Expected: 5 tests pass.

- [ ] **Step 5: Commit the preflight unit**

```bash
git add scripts/cloud_preflight.py tests/test_cloud_preflight.py
git commit -m "feat: add cloud hardware preflight"
```

### Task 2: Guard Environment, Data, And Training Scripts

**Files:**
- Create: `tests/test_cloud_scripts.py`
- Modify: `scripts/cloud_setup_searchr1.sh`
- Modify: `scripts/cloud_setup_retriever.sh`
- Create: `scripts/cloud_prepare_smoke_assets.sh`
- Modify: `scripts/cloud_prepare_data_and_index.sh`
- Modify: `scripts/cloud_launch_retriever.sh`
- Modify: `scripts/cloud_train_grpo_smoke.sh`
- Modify: `scripts/cloud_train_grpo_full.sh`

- [ ] **Step 1: Write failing script-contract tests**

The test reads scripts as text and asserts these exact contracts:

```python
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class CloudScriptContractTests(unittest.TestCase):
    def read(self, name):
        return (ROOT / "scripts" / name).read_text(encoding="utf-8")

    def test_requested_conda_environment_names_are_defaults(self):
        self.assertIn('ENV_NAME="${ENV_NAME:-Search-R1}"', self.read("cloud_setup_searchr1.sh"))
        self.assertIn('ENV_NAME="${ENV_NAME:-Search-R1-retriever}"', self.read("cloud_setup_retriever.sh"))

    def test_all_launchers_resolve_repository_root(self):
        for name in (
            "cloud_prepare_data_and_index.sh",
            "cloud_launch_retriever.sh",
            "cloud_train_grpo_smoke.sh",
            "cloud_train_grpo_full.sh",
        ):
            self.assertIn('REPO_ROOT=', self.read(name), name)
            self.assertIn('cd "$REPO_ROOT"', self.read(name), name)

    def test_retriever_has_smoke_and_full_asset_profiles(self):
        script = self.read("cloud_launch_retriever.sh")
        self.assertIn('ASSET_PROFILE="${ASSET_PROFILE:-smoke}"', script)
        self.assertIn('data/smoke_retriever/e5_Flat.index', script)
        self.assertIn('data/wiki18/e5_Flat.index', script)

    def test_full_run_has_hardware_and_confirmation_gates(self):
        script = self.read("cloud_train_grpo_full.sh")
        self.assertIn('cloud_preflight.py --profile full', script)
        self.assertIn('CONFIRM_FULL_RUN', script)

    def test_console_logging_is_default(self):
        script = self.read("cloud_train_grpo_full.sh")
        self.assertIn('TRAINER_LOGGER="${TRAINER_LOGGER:-console}"', script)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_cloud_scripts -v`

Expected: failures for old environment names, missing repository-root handling, missing asset profiles, and absent full-run gates.

- [ ] **Step 3: Implement minimal shell changes**

Every launcher calculates and enters the repository root:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
```

Use `Search-R1` and `Search-R1-retriever` as defaults. Add explicit file-size checks after data preparation. `cloud_prepare_smoke_assets.sh` processes NQ and builds `data/smoke_retriever/e5_Flat.index` from the tracked `example/corpus.jsonl` with `search_r1/search/index_builder.py`. The retrieval launcher selects smoke assets by default and full Wikipedia assets only when `ASSET_PROFILE=full` is explicit. Run the smoke preflight before the smoke training command. The full script must run the full preflight and require `CONFIRM_FULL_RUN=YES`; its logger override is assembled as `trainer.logger=['console']` or `trainer.logger=['wandb']` based on `TRAINER_LOGGER`.

- [ ] **Step 4: Verify official source provenance remains unchanged**

Run:

```bash
git diff --exit-code 598e61b -- search_r1 verl scripts/data_process
```

Expected: exit `0`; all rental compatibility behavior stays in new cloud helpers.

- [ ] **Step 5: Run tests and shell syntax checks**

Run:

```bash
python -m unittest tests.test_cloud_scripts -v
bash -n scripts/cloud_setup_searchr1.sh scripts/cloud_setup_retriever.sh scripts/cloud_prepare_smoke_assets.sh scripts/cloud_prepare_data_and_index.sh scripts/cloud_launch_retriever.sh scripts/cloud_train_grpo_smoke.sh scripts/cloud_train_grpo_full.sh
```

Expected: all contract tests pass and `bash -n` exits `0`.

- [ ] **Step 6: Commit guarded launchers**

```bash
git add tests/test_cloud_scripts.py scripts/cloud_*.sh
git commit -m "fix: guard cloud reproduction launchers"
```

### Task 3: Validate Retriever API Responses

**Files:**
- Create: `tests/test_cloud_check_retriever.py`
- Modify: `scripts/cloud_check_retriever.py`

- [ ] **Step 1: Write failing response-schema tests**

```python
import unittest

from scripts.cloud_check_retriever import validate_response


class RetrieverResponseTests(unittest.TestCase):
    def test_accepts_expected_response(self):
        payload = {"result": [[{"document": {"contents": '"Hamlet"\nText'}, "score": 1.0}]]}
        self.assertEqual(validate_response(payload, expected_topk=1)[0]["score"], 1.0)

    def test_rejects_missing_result(self):
        with self.assertRaisesRegex(ValueError, "result"):
            validate_response({}, expected_topk=1)

    def test_rejects_document_without_contents(self):
        payload = {"result": [[{"document": {}, "score": 1.0}]]}
        with self.assertRaisesRegex(ValueError, "contents"):
            validate_response(payload, expected_topk=1)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_cloud_check_retriever -v`

Expected: import failure for missing `validate_response`.

- [ ] **Step 3: Implement schema validation and configurable endpoint**

Add a pure `validate_response(data, expected_topk)` function. Move HTTP execution under `main()`, read `RETRIEVER_URL` and `TOPK` from environment variables, reject malformed or short result lists with actionable `ValueError` messages, and keep the existing concise document preview.

- [ ] **Step 4: Run tests and compile Python files**

Run:

```bash
python -m unittest tests.test_cloud_check_retriever -v
python -m py_compile scripts/cloud_preflight.py scripts/cloud_check_retriever.py search_r1/search/retrieval_server.py
```

Expected: 3 tests pass and compilation exits `0`.

- [ ] **Step 5: Commit retriever validation**

```bash
git add tests/test_cloud_check_retriever.py scripts/cloud_check_retriever.py
git commit -m "test: validate retriever preflight response"
```

### Task 4: Evidence Manifest And Beginner Documentation

**Files:**
- Create: `scripts/cloud_collect_evidence.sh`
- Modify: `tests/test_cloud_scripts.py`
- Modify: `docs/beginner_cloud_reproduction_zh.md`
- Modify: `docs/rental_reproduction_runbook.md`

- [ ] **Step 1: Add failing evidence-script contract tests**

Assert that `cloud_collect_evidence.sh` records `git rev-parse HEAD`, `nvidia-smi`, both Conda environment exports, disk/RAM information, log filenames, and SHA-256 hashes for parquet files.

- [ ] **Step 2: Run the contract test and verify RED**

Run: `python -m unittest tests.test_cloud_scripts.CloudScriptContractTests.test_evidence_script_records_reproduction_metadata -v`

Expected: failure because `scripts/cloud_collect_evidence.sh` does not exist.

- [ ] **Step 3: Implement evidence collection**

Create a strict-mode shell script that writes into `artifacts/reproduction-<UTC timestamp>/`, captures system and Git metadata, exports explicit Conda package lists, hashes prepared parquet files, copies available training logs, and creates `README.txt` stating whether evidence is from smoke or full mode.

- [ ] **Step 4: Rewrite the beginner path around exact commands**

The Chinese document must use the real repository URL `https://github.com/heba324/Search-R1`, exact environment names, this sequence, and explicit stop conditions:

```bash
python3 scripts/cloud_preflight.py --profile smoke
bash scripts/cloud_setup_searchr1.sh
bash scripts/cloud_setup_retriever.sh
bash scripts/cloud_prepare_smoke_assets.sh
bash scripts/cloud_launch_retriever.sh
bash scripts/cloud_train_grpo_smoke.sh
bash scripts/cloud_collect_evidence.sh smoke
bash scripts/cloud_prepare_data_and_index.sh
ASSET_PROFILE=full bash scripts/cloud_launch_retriever.sh
CONFIRM_FULL_RUN=YES bash scripts/cloud_train_grpo_full.sh
bash scripts/cloud_collect_evidence.sh full
```

It must state that local tests are not reproduction, smoke is not paper reproduction, and full success requires logs, checkpoints, evaluation metrics, and package/hardware evidence.

- [ ] **Step 5: Run tests and documentation consistency checks**

Run:

```bash
python -m unittest tests.test_cloud_scripts -v
rg -n "searchr1|conda activate retriever|你的用户名|Search-R1-reproduce" docs/beginner_cloud_reproduction_zh.md docs/rental_reproduction_runbook.md scripts/cloud_*.sh
```

Expected: tests pass; the search returns no obsolete environment names or placeholder repository names.

- [ ] **Step 6: Commit evidence and documentation**

```bash
git add scripts/cloud_collect_evidence.sh tests/test_cloud_scripts.py docs/beginner_cloud_reproduction_zh.md docs/rental_reproduction_runbook.md
git commit -m "docs: add reproducible cloud evidence workflow"
```

### Task 5: Full Local Readiness Verification

**Files:**
- Modify only files required to fix failures found by the commands below.

- [ ] **Step 1: Run the complete standard-library test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 2: Run syntax verification**

Run:

```bash
bash -n scripts/cloud_*.sh
python -m compileall -q scripts tests search_r1/search/retrieval_server.py
```

Expected: both commands exit `0`.

- [ ] **Step 3: Verify upstream provenance and clean diffs**

Run:

```bash
git diff --check origin/main...HEAD
git diff --name-only origin/main...HEAD
git status --short
```

Expected: no whitespace errors; only cloud helpers, tests, and docs differ from upstream; no uncommitted files remain.

- [ ] **Step 4: Push the verified branch**

```bash
git push heba HEAD
```

Expected: the remote branch advances to the verified local commit.

- [ ] **Step 5: Record the honest readiness boundary**

Report exactly these two states separately:

```text
Local readiness: verified by automated tests and syntax checks.
Cloud reproduction: not yet claimed; requires running the smoke and full gates on rented Linux GPU hardware.
```
