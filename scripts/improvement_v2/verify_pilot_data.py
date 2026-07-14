#!/usr/bin/env python3
"""Verify that the disjoint pilot and frozen final set still match their manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pilot_files(pilot, final_eval, manifest_path):
    errors = []
    for label, path in (
        ("pilot", pilot),
        ("final evaluation", final_eval),
        ("pilot manifest", manifest_path),
    ):
        if not path.is_file():
            errors.append(f"Missing {label} file: {path}")
    if errors:
        return errors

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if pilot.stat().st_size != manifest.get("pilot_bytes"):
        errors.append("pilot file size changed")
    elif _sha256(pilot) != manifest.get("pilot_sha256"):
        errors.append("pilot SHA-256 changed")
    if _sha256(final_eval) != manifest.get("excluded_final_eval_sha256"):
        errors.append("final evaluation SHA-256 changed after pilot selection")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot", type=Path, default=Path("data/improvement_v2/pilot.parquet")
    )
    parser.add_argument(
        "--final-eval", type=Path, default=Path("data/course_eval/test.parquet")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/improvement_v2/pilot_manifest.json"),
    )
    args = parser.parse_args()
    errors = verify_pilot_files(args.pilot, args.final_eval, args.manifest)
    if errors:
        raise SystemExit("Pilot data verification failed:\n- " + "\n- ".join(errors))
    print(f"Pilot data verified: {args.pilot}")


if __name__ == "__main__":
    main()
