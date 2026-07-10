# Search-R1 Cloud Reproduction Readiness Design

## Goal

Prepare a beginner-friendly, cost-controlled cloud workflow that validates as much
as possible before renting GPUs, then separates a cheap single-GPU startup check
from the official eight-GPU reproduction run.

## Baseline And Scope

- The upstream baseline is `PeterGriffinJin/Search-R1` commit `598e61b`.
- Upstream training, retrieval, reward, and data-processing source files remain
  unchanged so experiment provenance stays clear.
- Reproduction helpers live under `scripts/cloud_*`; beginner instructions live
  under `docs/`.
- This phase prepares and verifies reproduction only. It does not implement a
  research improvement or claim paper-level results.

## Considered Approaches

1. Rent eight A100 GPUs immediately and follow upstream commands manually. This
   is closest to the original setup but spends money while discovering basic
   installation, path, data, and login errors.
2. Use only one GPU and reduce the full experiment. This is cheaper, but it
   changes the hardware and training configuration too much to count as an
   official-scale reproduction.
3. Use a gated two-stage workflow. A single A100 validates installation, data,
   models, retrieval, configuration composition, and startup behavior; eight
   A100 GPUs are rented only after those checks pass. This is the selected design.

## Workflow

### Local Gate

A repository test checks shell syntax, Python syntax, required upstream files,
Hydra override names, environment-name consistency, and accidental placeholders.
It requires no CUDA runtime and can be rerun before every push.

### Cloud Preflight Gate

A read-only preflight script reports Linux distribution, GPU count and memory,
CUDA driver visibility, RAM, free disk, Conda availability, repository revision,
and network access to required model/data hosts. It fails with actionable messages
when a hard requirement is missing.

Two profiles are supported:

- `smoke`: at least one NVIDIA GPU, at least 70 GiB RAM, and at least 250 GiB
  free disk. Passing this profile authorizes preparation and startup checks only.
- `full`: exactly the requested official single-node scale of at least eight
  NVIDIA GPUs, at least 40 GiB VRAM per GPU, at least 128 GiB RAM, and at least
  500 GiB free disk.

### Environment And Data Gate

The main Conda environment is named `Search-R1`, as requested. The retriever uses
the separate `Search-R1-retriever` environment because its FAISS and Python
requirements differ from the training runtime.

Environment scripts are idempotent. Data preparation supports resumed Hugging
Face downloads and verifies that both index parts, the joined FAISS index, the
decompressed corpus, and NQ parquet files exist and are non-empty before success.

### Retrieval Gate

The retrieval launcher validates its files and accepts an explicit port. The API
check validates the response schema as well as HTTP status, ensuring that training
will receive documents in the shape expected by Search-R1.

### Training Gates

- `smoke` performs a very short configuration/startup run and never presents its
  output as a paper reproduction result.
- `full` retains the official Qwen2.5-3B GRPO scale and refuses to launch unless
  the full hardware preflight passes.
- Console logging is the default so a missing WandB login cannot waste rental
  time. WandB remains opt-in through an environment variable.

## Error Handling And Cost Controls

- Shell scripts use strict mode and resolve the repository root independently of
  the caller's current directory.
- Expensive stages validate prerequisites before loading models or starting Ray.
- The full launcher requires an explicit confirmation variable in addition to a
  passing hardware preflight.
- Every failure prints the failed requirement and the command needed for the next
  diagnostic step.

## Verification

Automated tests exercise preflight decisions with fixture command outputs rather
than requiring real GPUs. Static checks parse all added shell scripts and compile
all added Python files. Repository checks compare the upstream tree and confirm
that official source files are not modified.

The final local verdict can only mean "ready to rent for cloud verification."
Actual reproduction success requires fresh cloud evidence: retriever response,
training logs, checkpoints, evaluation metrics, hardware inventory, and package
versions.

## Deliverables

- Corrected cloud setup, preparation, retrieval, smoke, and full-run scripts.
- A cloud hardware/repository preflight command.
- Local automated readiness tests.
- Updated Chinese beginner instructions with exact commands and result meanings.
- A manifest for recording commit, environment, hardware, logs, and checkpoints
  used in the eventual experimental report.
