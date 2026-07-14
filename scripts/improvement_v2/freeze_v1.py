#!/usr/bin/env python3
"""Freeze and verify the completed CEGR V1 checkpoints and evidence by SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


V1_EVIDENCE_PATHS = (
    "verl_checkpoints/search-r1-course-qwen2.5-1.5b-grpo-bm25/actor/global_step_120",
    "verl_checkpoints/search-r1-cegr-qwen2.5-1.5b-grpo-bm25/actor/global_step_120",
    "artifacts/improvement/baseline-vs-cegr.json",
    "artifacts/improvement/paired-statistical-analysis.json",
    "artifacts/improvement/paired-evaluation",
    "artifacts/course-reproduction/evaluation/baseline-paired",
    "artifacts/course-reproduction/evaluation/cegr-post-rl",
    "scripts/improvement",
    "docs/improvement_experiment_zh.md",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_files(root, relative_paths):
    files = {}
    for relative in relative_paths:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing V1 evidence: {relative}")
        scan_root = path.resolve() if path.is_dir() else path
        candidates = [scan_root] if scan_root.is_file() else sorted(
            candidate for candidate in scan_root.rglob("*") if candidate.is_file()
        )
        if not candidates:
            raise FileNotFoundError(f"Missing V1 evidence files under: {relative}")
        for candidate in candidates:
            if scan_root.is_file():
                manifest_path = Path(relative).as_posix()
            else:
                manifest_path = (
                    Path(relative) / candidate.relative_to(scan_root)
                ).as_posix()
            files[manifest_path] = candidate
    return files


def create_manifest(root, relative_paths=V1_EVIDENCE_PATHS):
    candidates = _collect_files(root, relative_paths)
    files = {
        manifest_path: {
            "bytes": candidate.stat().st_size,
            "sha256": _sha256(candidate),
        }
        for manifest_path, candidate in candidates.items()
    }
    return {
        "contract": "CEGR V1 code, checkpoints, and evidence are immutable after V2 begins",
        "relative_paths": list(relative_paths),
        "files": files,
    }


def verify_manifest(root, manifest):
    errors = []
    relative_paths = manifest.get("relative_paths")
    if not relative_paths:
        return ["Frozen V1 manifest does not declare its scanned roots"]
    try:
        current_files = _collect_files(root, relative_paths)
    except FileNotFoundError as error:
        return [str(error)]
    expected_names = set(manifest["files"])
    current_names = set(current_files)
    for relative in sorted(current_names - expected_names):
        errors.append(f"Unexpected frozen V1 file: {relative}")
    for relative, expected in manifest["files"].items():
        path = current_files.get(relative)
        if path is None or not path.is_file():
            errors.append(f"Missing frozen V1 file: {relative}")
            continue
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"File size changed for frozen V1 file: {relative}")
            continue
        if _sha256(path) != expected["sha256"]:
            errors.append(f"SHA-256 changed for frozen V1 file: {relative}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/improvement-v2/v1-frozen-manifest.json"),
    )
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    if args.initialize and not manifest_path.exists():
        manifest = create_manifest(root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Frozen CEGR V1 manifest created: {manifest_path}")
        return
    if not manifest_path.is_file():
        raise SystemExit(
            "Missing V1 freeze manifest; run freeze_v1.py --initialize before V2"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = verify_manifest(root, manifest)
    if errors:
        raise SystemExit("CEGR V1 freeze verification failed:\n- " + "\n- ".join(errors))
    print(f"CEGR V1 freeze verified: {manifest_path}")


if __name__ == "__main__":
    main()
