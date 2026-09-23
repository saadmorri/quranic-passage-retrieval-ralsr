#!/usr/bin/env python3
"""
Verify local official Qur'an QA 2023 Task A resource files against immutable
Git blob IDs from the organizers' repository.

This does not download or alter files.
"""

from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

EXPECTED = {
    "QQA23_TaskA_ayatec_v1.2_train.tsv": "7d4400c3abb1c2c893731d6b89686ec1e776837a",
    "QQA23_TaskA_ayatec_v1.2_dev.tsv": "af8db5f1a498fdea22c2cd2305d8076a397eb200",
    "QQA23_TaskA_ayatec_v1.2_test.tsv": "39ce1ef1a5e3c00e54d60be29bf56c1af1094adc",
    "QQA23_TaskA_ayatec_v1.2_qrels_train.gold": "119837f85dd1a2ee08f846b3792b21b787ef3a57",
    "QQA23_TaskA_ayatec_v1.2_qrels_dev.gold": "8c0758a2b75394e98f8a0f568b2d77690a9a96d3",
    "QQA23_TaskA_ayatec_v1.2_qrels_test.gold": "9c2c403d563a8ccae4d65b3e06382b2fc4f2d7a9",
    "QQA23_TaskA_QPC_v1.1.tsv": "1876a5284561800f91539b14ee321dcec56d613b",
}

def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("resource_dir", help="Directory containing the official files")
    args = ap.parse_args()

    root = Path(args.resource_dir)
    failures = 0
    for name, expected in EXPECTED.items():
        candidates = list(root.rglob(name))
        if not candidates:
            print(f"MISSING  {name}")
            failures += 1
            continue
        if len(candidates) > 1:
            print(f"WARNING  multiple copies found for {name}; checking all")
        for path in candidates:
            actual = git_blob_sha1(path)
            status = "OK" if actual == expected else "MISMATCH"
            print(f"{status:8s} {path}  blob={actual}")
            if actual != expected:
                failures += 1

    raise SystemExit(1 if failures else 0)

if __name__ == "__main__":
    main()
