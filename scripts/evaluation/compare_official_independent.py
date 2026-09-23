#!/usr/bin/env python3
"""Compare organizer official metrics TSV with independent verification JSON."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True, help="TSV produced by official scorer")
    ap.add_argument("--independent", required=True, help="JSON from independent evaluator")
    ap.add_argument("--tolerance", type=float, default=1e-10)
    args = ap.parse_args()

    off = pd.read_csv(args.official, sep="\t")
    if len(off) != 1:
        raise SystemExit("Official metric file must contain exactly one data row.")
    ind = json.loads(Path(args.independent).read_text(encoding="utf-8"))["overall"]

    failures = []
    for metric in ("map_cut_10", "recip_rank"):
        ov = float(off.iloc[0][metric])
        iv = float(ind[metric])
        diff = abs(ov - iv)
        print(f"{metric}: official={ov:.12f} independent={iv:.12f} diff={diff:.3e}")
        if diff > args.tolerance:
            failures.append(metric)

    if failures:
        raise SystemExit("Metric mismatch beyond tolerance: " + ", ".join(failures))
    print("MATCH: official and independent metrics agree within tolerance.")

if __name__ == "__main__":
    main()
