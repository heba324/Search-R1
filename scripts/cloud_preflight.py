#!/usr/bin/env python3
"""Check a rented host before starting a Search-R1 cloud run."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


MIB_PER_GIB = 1024
POLICIES: Dict[str, Dict[str, int]] = {
    "smoke": {"gpus": 1, "vram_mib": 75 * MIB_PER_GIB, "ram_gib": 64, "disk_gib": 100},
    "full": {"gpus": 8, "vram_mib": 40 * MIB_PER_GIB, "ram_gib": 128, "disk_gib": 500},
}
NETWORK_URLS = (
    "https://github.com",
    "https://huggingface.co",
    "https://download.pytorch.org",
)
REQUIRED_REPOSITORY_PATHS = (
    "search_r1",
    "verl",
    "scripts/data_process/nq_search.py",
    "train_grpo.sh",
)


@dataclass(frozen=True)
class HardwareInfo:
    gpu_memory_mib: Tuple[int, ...]
    ram_gib: int
    disk_gib: int


def parse_gpu_memory(output: str) -> Tuple[int, ...]:
    """Parse one MiB value per line from nvidia-smi query output."""
    values: List[int] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError as exc:
            raise ValueError(f"Invalid nvidia-smi memory value: {line!r}") from exc
    return tuple(values)


def assess_hardware(profile: str, info: HardwareInfo) -> List[str]:
    """Return user-facing requirement failures for a hardware profile."""
    if profile not in POLICIES:
        raise ValueError(f"Unknown profile: {profile}")

    policy = POLICIES[profile]
    errors: List[str] = []
    required_gpus = policy["gpus"]
    required_vram_mib = policy["vram_mib"]

    if len(info.gpu_memory_mib) < required_gpus:
        errors.append(
            f"{profile} requires at least {required_gpus} NVIDIA GPUs; "
            f"found {len(info.gpu_memory_mib)}."
        )
    else:
        selected = sorted(info.gpu_memory_mib, reverse=True)[:required_gpus]
        if min(selected) < required_vram_mib:
            errors.append(
                f"{profile} requires at least {required_vram_mib // MIB_PER_GIB} GiB VRAM "
                f"on each selected GPU; found {min(selected) / MIB_PER_GIB:.1f} GiB."
            )

    if info.ram_gib < policy["ram_gib"]:
        errors.append(
            f"{profile} requires at least {policy['ram_gib']} GiB RAM; "
            f"found {info.ram_gib} GiB."
        )
    if info.disk_gib < policy["disk_gib"]:
        errors.append(
            f"{profile} requires at least {policy['disk_gib']} GiB free disk; "
            f"found {info.disk_gib} GiB."
        )
    return errors


def run_command(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def collect_gpu_memory() -> Tuple[int, ...]:
    output = run_command(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    )
    return parse_gpu_memory(output)


def collect_ram_gib() -> int:
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    first_line = next(line for line in meminfo.splitlines() if line.startswith("MemTotal:"))
    total_kib = int(first_line.split()[1])
    return total_kib // (MIB_PER_GIB * MIB_PER_GIB)


def collect_hardware(repo_root: Path) -> HardwareInfo:
    free_disk_gib = shutil.disk_usage(repo_root).free // (MIB_PER_GIB**3)
    return HardwareInfo(
        gpu_memory_mib=collect_gpu_memory(),
        ram_gib=collect_ram_gib(),
        disk_gib=free_disk_gib,
    )


def check_repository(repo_root: Path) -> List[str]:
    errors: List[str] = []
    for relative_path in REQUIRED_REPOSITORY_PATHS:
        if not (repo_root / relative_path).exists():
            errors.append(f"Missing repository path: {relative_path}")
    return errors


def check_commands() -> List[str]:
    errors: List[str] = []
    for command in ("conda", "git", "nvidia-smi"):
        if shutil.which(command) is None:
            errors.append(f"Required command is unavailable: {command}")
    return errors


def check_network(urls: Sequence[str] = NETWORK_URLS, timeout: int = 10) -> List[str]:
    errors: List[str] = []
    for url in urls:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Search-R1-preflight"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status >= 400:
                    errors.append(f"Network check failed for {url}: HTTP {response.status}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"Network check failed for {url}: {exc}")
    return errors


def print_hardware(info: HardwareInfo) -> None:
    gpu_values = ", ".join(f"{value / MIB_PER_GIB:.1f} GiB" for value in info.gpu_memory_mib)
    print(f"GPU count: {len(info.gpu_memory_mib)}")
    print(f"GPU memory: {gpu_values or 'none'}")
    print(f"RAM: {info.ram_gib} GiB")
    print(f"Free disk: {info.disk_gib} GiB")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(POLICIES), required=True)
    parser.add_argument("--skip-network", action="store_true", help="Skip external HTTPS checks.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Search-R1 repository root.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    errors: List[str] = []

    print(f"Search-R1 cloud preflight profile: {args.profile}")
    print(f"Repository: {repo_root}")
    print(f"Platform: {platform.platform()}")

    if platform.system() != "Linux":
        errors.append("Cloud reproduction requires Linux (Ubuntu 20.04 or 22.04 recommended).")
    errors.extend(check_commands())
    errors.extend(check_repository(repo_root))

    hardware: Optional[HardwareInfo] = None
    if shutil.which("nvidia-smi") is not None and Path("/proc/meminfo").exists():
        try:
            hardware = collect_hardware(repo_root)
            print_hardware(hardware)
            errors.extend(assess_hardware(args.profile, hardware))
        except (OSError, ValueError, subprocess.SubprocessError, StopIteration) as exc:
            errors.append(f"Unable to collect hardware information: {exc}")

    if not args.skip_network:
        errors.extend(check_network())

    if errors:
        print("\nPreflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    commit = run_command(["git", "-C", os.fspath(repo_root), "rev-parse", "HEAD"])
    print(f"Git commit: {commit}")
    print(f"Preflight passed for profile: {args.profile}")
    if args.profile == "smoke":
        print("This authorizes a startup smoke test only; it is not a paper reproduction result.")
    else:
        print("Hardware is eligible to start the full run; logs and metrics are still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
