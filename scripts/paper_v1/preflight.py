#!/usr/bin/env python3
"""Validate a host before a Search-R1 arXiv v1 paper reproduction run."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from scripts.paper_v1.contract import PAPER_V1, assess_required_assets

AUTHOR_CORE_PATHS = (
    "search_r1",
    "verl",
    "scripts/nq_hotpotqa",
    "train_ppo.sh",
    "train_grpo.sh",
    "retrieval_launch.sh",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
)


@dataclass(frozen=True)
class HostInfo:
    gpu_count: int
    ram_gib: int
    disk_gib: int


def assess_host(info: HostInfo) -> List[str]:
    """Return paper-v1 host requirement failures."""
    errors: List[str] = []
    if info.gpu_count < 8:
        errors.append(f"Paper v1 requires at least 8 NVIDIA GPUs; found {info.gpu_count}.")
    if info.ram_gib < 128:
        errors.append(f"Paper v1 requires at least 128 GiB RAM; found {info.ram_gib} GiB.")
    if info.disk_gib < 500:
        errors.append(f"Paper v1 requires at least 500 GiB free disk; found {info.disk_gib} GiB.")
    return errors


def assess_author_source(repo_root: Path) -> List[str]:
    """Verify that the wrapper branch contains the unchanged paper-v1 core."""
    ancestor = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "merge-base", "--is-ancestor", PAPER_V1.git_commit, "HEAD"],
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        return [f"Author paper commit {PAPER_V1.git_commit[:7]} is not an ancestor of HEAD."]
    core_diff = subprocess.run(
        ["git", "-C", os.fspath(repo_root), "diff", "--quiet", PAPER_V1.git_commit, "--", *AUTHOR_CORE_PATHS],
        capture_output=True,
        text=True,
    )
    if core_diff.returncode == 1:
        return ["Author Search-R1 training core differs from paper commit 118c6e7."]
    if core_diff.returncode != 0:
        return [f"Unable to compare author training core: {core_diff.stderr.strip()}"]
    return []


def run_command(command: Sequence[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def collect_host_info(repo_root: Path) -> HostInfo:
    gpu_lines = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).splitlines()
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    mem_total_kib = int(next(line for line in meminfo.splitlines() if line.startswith("MemTotal:")).split()[1])
    return HostInfo(
        gpu_count=len([line for line in gpu_lines if line.strip()]),
        ram_gib=mem_total_kib // (1024 * 1024),
        disk_gib=shutil.disk_usage(repo_root).free // (1024**3),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--require-assets", action="store_true", help="Also require downloaded data and retrieval assets.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    errors: List[str] = []
    print("Search-R1 paper v1 preflight")
    print(f"Repository: {repo_root}")

    if platform.system() != "Linux":
        errors.append("Paper v1 reproduction requires Linux.")
    for command in ("conda", "git", "nvidia-smi"):
        if shutil.which(command) is None:
            errors.append(f"Required command is unavailable: {command}")

    try:
        head_commit = run_command(["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"])
        errors.extend(assess_author_source(repo_root))
        print(f"Wrapper commit: {head_commit}")
        print(f"Frozen author core: {PAPER_V1.git_commit}")
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(str(exc))

    if platform.system() == "Linux" and shutil.which("nvidia-smi") is not None:
        try:
            info = collect_host_info(repo_root)
            print(f"Visible GPUs: {info.gpu_count}")
            print(f"RAM: {info.ram_gib} GiB")
            print(f"Free disk: {info.disk_gib} GiB")
            errors.extend(assess_host(info))
        except (OSError, StopIteration, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"Unable to collect host information: {exc}")

    if args.require_assets:
        errors.extend(assess_required_assets(repo_root))

    if errors:
        print("Preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Paper v1 preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
