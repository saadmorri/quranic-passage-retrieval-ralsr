#!/usr/bin/env python3
"""
Run the preserved organizers' official Task A checker and scorer.

Recommended order:
1) quranqa_taskA_strict_validator.py
2) this wrapper (official checker + official scorer)
3) quranqa_taskA_independent_eval.py
4) compare official and independent map_cut_10 / recip_rank

This wrapper never modifies the official scripts.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--official-dir", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()

    official = Path(args.official_dir)
    checker = official / "QQA23_TaskA_submission_checker.py"
    scorer = official / "QQA23_TaskA_eval.py"
    for p in (checker, scorer):
        if not p.exists():
            raise SystemExit(f"Missing preserved official file: {p}")

    print("=== Official submission checker ===")
    check = subprocess.run(
        [sys.executable, str(checker), "--model-prediction", str(Path(args.run).resolve())],
        cwd=str(official.resolve()),
        capture_output=True,
        text=True,
    )
    if check.stdout:
        print(check.stdout, end="")
    if check.stderr:
        print(check.stderr, end="", file=sys.stderr)

    # The released organizer checker prints failure but its CLI does not
    # propagate check_run() as a nonzero process exit code. Require the
    # organizer's explicit success message in addition to return status.
    if check.returncode != 0 or "Format check: Passed" not in check.stdout:
        raise SystemExit(
            "Official submission checker did not confirm 'Format check: Passed'."
        )

    print("\n=== Official Task A scorer ===")
    cmd = [
        sys.executable, str(scorer),
        "--run", str(Path(args.run).resolve()),
        "--qrels", str(Path(args.qrels).resolve()),
    ]
    if args.output:
        cmd += ["--output", str(Path(args.output).resolve())]
    score = subprocess.run(cmd, cwd=str(official.resolve()))
    raise SystemExit(score.returncode)

if __name__ == "__main__":
    main()
