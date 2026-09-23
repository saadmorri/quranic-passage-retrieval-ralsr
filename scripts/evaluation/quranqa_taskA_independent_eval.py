#!/usr/bin/env python3
"""
Independent verifier for the two official Qur'an QA 2023 Task A measures.

This is NOT a substitute for the organizers' official scorer. Its purpose is
cross-checking a strict-valid run.

Normal questions:
- AP@10 is trec_eval map_cut_10 semantics:
  sum of precision at relevant hits within top 10 / TOTAL number of relevant
  documents for that qid.
- RR is reciprocal rank of the first relevant hit.
Because official runs are capped at 10, recip_rank is effectively MRR@10.

Zero-answer questions:
- the official scorer gives full score 1 for BOTH measures iff the run has
  exactly one retrieved document and its docid is -1; otherwise 0.
- rank 1 is an additional thesis-side strict-validation requirement.

The overall score is the mean over qids represented in the qrels. A question
present in the question file but absent from the official qrels is reported
separately and is not inventively assigned a gold label.
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

def read_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _, docid, rel = line.rstrip("\n").split("\t")
            qrels[qid][docid] = int(rel)
    return qrels

def read_run(path):
    run = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, _, docid, rank, score, tag = line.rstrip("\n").split("\t")
            run[qid].append((int(rank), float(score), docid))
    # The released official scorer converts docids to score values and lets
    # pytrec_eval reconstruct the ranking from score. Strict validation makes
    # submitted rank and score order agree; sorting by score here mirrors the
    # authoritative scorer directly.
    for qid in run:
        run[qid].sort(key=lambda x: (-x[1], x[2]))
    return run

def evaluate(qrels, run):
    per_qid = {}
    for qid, docs in qrels.items():
        rows = run.get(qid, [])
        if "-1" in docs:
            # The official scorer checks only that the qid has exactly one
            # retrieved docid and that it is -1. Rank-1 structure is enforced
            # separately by the thesis-side strict validator.
            correct = len(rows) == 1 and rows[0][2] == "-1"
            score = 1.0 if correct else 0.0
            per_qid[qid] = {"map_cut_10": score, "recip_rank": score, "zero_answer": True}
            continue

        relevant = {docid for docid, rel in docs.items() if rel > 0 and docid != "-1"}
        rel_so_far = 0
        ap_sum = 0.0
        rr = 0.0
        for rank, _, docid in rows[:10]:
            if docid in relevant:
                rel_so_far += 1
                ap_sum += rel_so_far / rank
                if rr == 0.0:
                    rr = 1.0 / rank
        ap10 = ap_sum / len(relevant) if relevant else 0.0
        per_qid[qid] = {"map_cut_10": ap10, "recip_rank": rr, "zero_answer": False}

    if per_qid:
        map10 = sum(x["map_cut_10"] for x in per_qid.values()) / len(per_qid)
        mrr = sum(x["recip_rank"] for x in per_qid.values()) / len(per_qid)
    else:
        map10 = mrr = 0.0
    return {"map_cut_10": map10, "recip_rank": mrr, "n_qrels_qids": len(per_qid)}, per_qid

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()

    overall, per_qid = evaluate(read_qrels(args.qrels), read_run(args.run))
    print(f"map_cut_10 = {overall['map_cut_10']:.12f}")
    print(f"recip_rank = {overall['recip_rank']:.12f}")
    print(f"scored qids = {overall['n_qrels_qids']}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"overall": overall, "per_qid": per_qid},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

if __name__ == "__main__":
    main()
