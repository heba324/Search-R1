#!/usr/bin/env python3
"""Check a host for the single-GPU Search-R1 course reproduction."""

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

from scripts.course_reproduction.contract import assess_assets


@dataclass(frozen=True)
class HostInfo:
    gpu_count: int
    gpu_memory_gib: int
    ram_gib: int
    disk_gib: int


def assess_host(info: HostInfo) -> List[str]:
    errors = []
    if info.gpu_count < 1:
        errors.append("Course reproduction requires at least one NVIDIA GPU.")
    if info.gpu_memory_gib < 79:
        errors.append(f"Course reproduction requires an 80 GiB-class GPU; found {info.gpu_memory_gib} GiB.")
    if info.ram_gib < 120:
        errors.append(f"Course reproduction requires at least 120 GiB RAM; found {info.ram_gib} GiB.")
    if info.disk_gib < 500:
        errors.append(f"Course reproduction requires at least 500 GiB free disk; found {info.disk_gib} GiB.")
    return errors


def run(command: Sequence[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def collect_host_info(repo_root: Path) -> HostInfo:
    memory_lines = run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).splitlines()
    memories_mib = [int(line.strip()) for line in memory_lines if line.strip()]
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    ram_kib = int(next(line for line in meminfo.splitlines() if line.startswith("MemTotal:")).split()[1])
    return HostInfo(
        gpu_count=len(memories_mib),
        gpu_memory_gib=max(memories_mib, default=0) // 1024,
        ram_gib=ram_kib // (1024 * 1024),
        disk_gib=shutil.disk_usage(repo_root).free // (1024**3),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--require-assets", action="store_true")
    args = parser.parse_args(argv)
    errors = []
    if platform.system() != "Linux":
        errors.append("Course reproduction requires Linux.")
    for command in ("conda", "git", "nvidia-smi"):
        if shutil.which(command) is None:
            errors.append(f"Required command is unavailable: {command}")
    if platform.system() == "Linux" and shutil.which("nvidia-smi"):
        try:
            info = collect_host_info(args.repo_root)
            print(f"GPUs: {info.gpu_count}; largest GPU: {info.gpu_memory_gib} GiB")
            print(f"RAM: {info.ram_gib} GiB; free disk: {info.disk_gib} GiB")
            errors.extend(assess_host(info))
        except (OSError, StopIteration, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"Unable to collect host information: {exc}")
    if args.require_assets:
        errors.extend(assess_assets(args.repo_root.resolve()))
    if errors:
        print("Course reproduction preflight failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Course reproduction preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
