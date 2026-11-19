# Host Runtime Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect cloud images that cannot build or reliably run Search-R1 before model and dataset downloads begin.

**Architecture:** Extend the standard-library `scripts/cloud_preflight.py` with a small, pure runtime assessment function. It receives driver version, shared-memory capacity, build-tool availability, and file-descriptor limit, returning blocking errors and non-blocking warnings. `main()` collects the Linux-host facts only when the existing base checks permit it, prints the facts and remediation text, and keeps the existing hardware and network gates intact.

**Tech Stack:** Python 3.9 standard library, `unittest`, Bash, Markdown.

---

### Task 1: Specify runtime preflight behavior with failing tests

**Files:**
- Modify: `tests/test_cloud_preflight.py`

- [ ] **Step 1: Write failing tests for supported CUDA driver, shared memory, and build tools**

```python
runtime = RuntimeInfo("520.00", 32, ("nvcc", "gcc", "g++", "make"), 65535)
errors, warnings = assess_runtime("smoke", runtime)
self.assertIn("CUDA 12.1 requires NVIDIA driver", "\n".join(errors))
```

- [ ] **Step 2: Run the focused test and verify it fails because `RuntimeInfo` and `assess_runtime` do not exist**

Run: `python -m unittest tests.test_cloud_preflight.CloudPreflightTests.test_smoke_rejects_old_cuda_driver -v`

Expected: import error for the new runtime assessment API.

### Task 2: Implement runtime collection and assessment

**Files:**
- Modify: `scripts/cloud_preflight.py`
- Test: `tests/test_cloud_preflight.py`

- [ ] **Step 1: Add `RuntimeInfo` and `assess_runtime`**

```python
@dataclass(frozen=True)
class RuntimeInfo:
    driver_version: str
    shared_memory_gib: int
    build_tools: Tuple[str, ...]
    open_file_limit: int
```

`assess_runtime` must block driver versions below `525.60`, missing `nvcc`, `gcc`, `g++`, or `make`, and shared memory below 8 GiB for smoke or 16 GiB for full. It must warn, rather than fail, when the soft open-file recommendation of 65535 is not met, because the user can raise it in the current shell.

- [ ] **Step 2: Collect host values in `main()` after the existing command and hardware gates**

Use `nvidia-smi --query-gpu=driver_version`, `shutil.disk_usage("/dev/shm")`, `resource.getrlimit(resource.RLIMIT_NOFILE)`, and `shutil.which`. Print the collected values, errors, and warnings without changing existing network checks.

- [ ] **Step 3: Run focused tests and the complete suite**

Run: `python -m unittest tests.test_cloud_preflight -v` and `python -m unittest discover -s tests -v`.

Expected: all tests pass.

### Task 3: Document cloud-image requirements and remediation

**Files:**
- Modify: `docs/beginner_cloud_reproduction_zh.md`
- Modify: `docs/rental_reproduction_runbook.md`
- Test: `tests/test_cloud_scripts.py`

- [ ] **Step 1: Add a concise preflight requirement list before the first smoke command**

Document the minimum NVIDIA driver, CUDA compilation tools, `/dev/shm` capacity, and the `ulimit -n 65535` remedy. State that an image lacking `nvcc` should be replaced before beginning the paid setup.

- [ ] **Step 2: Add a text contract test for the documented runtime requirements**

Run: `python -m unittest tests.test_cloud_scripts.CloudScriptContractTests.test_beginner_docs_use_current_commands -v`.

- [ ] **Step 3: Run syntax and compilation checks**

Run: `bash -n scripts/cloud_*.sh` and `python -m compileall -q scripts tests search_r1 verl`.

### Task 4: Verify, review, integrate, and publish

**Files:**
- Verify: `scripts/cloud_preflight.py`
- Verify: `tests/test_cloud_preflight.py`
- Verify: `docs/beginner_cloud_reproduction_zh.md`
- Verify: `docs/rental_reproduction_runbook.md`

- [ ] **Step 1: Run the full automated verification suite and inspect the diff**

Run: `python -m unittest discover -s tests -v`, `bash -n scripts/cloud_*.sh`, `python -m compileall -q scripts tests search_r1 verl`, and `git diff --check`.

- [ ] **Step 2: Commit the focused changes and fast-forward `main` only after all checks pass**

```bash
git add scripts/cloud_preflight.py tests/test_cloud_preflight.py tests/test_cloud_scripts.py docs/beginner_cloud_reproduction_zh.md docs/rental_reproduction_runbook.md docs/superpowers/plans/2026-07-11-host-runtime-preflight.md
git commit -m "feat: harden cloud host preflight"
```
