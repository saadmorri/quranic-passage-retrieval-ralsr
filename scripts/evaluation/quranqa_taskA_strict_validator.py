#!/usr/bin/env python3
"""
Strict pre-validator for Qur'an QA 2023 Task A runs.

This intentionally checks more than the organizers' official submission checker.
It does NOT replace the official checker; run both before scoring.

Checks include:
- official filename and six-column tab-separated structure
- complete split question coverage
- known question IDs and known QPC passage IDs
- Q0, integer contiguous ranks 1..N, max 10
- finite scores, strictly decreasing by rank
- duplicate qid/docid and duplicate rank detection
- exact no-answer row structure (-1 only, single row, rank 1)
- optional qrels audit without using qrels to alter predictions
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict

def read_ids_tsv(path):
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ids.append(line.rstrip("\n").split("\t", 1)[0])
    return ids

def read_qpc_ids(path):
    return set(read_ids_tsv(path))

def read_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"Malformed qrels line {n}: expected 4 tab columns")
            qid, _, docid, rel = parts
            qrels[qid][docid] = int(rel)
    return qrels

def validate(run_path, questions_path, qpc_path, qrels_path=None):
    errors, warnings = [], []
    run_path = Path(run_path)

    if not re.fullmatch(r"[A-Za-z0-9]{3,9}_[A-Za-z0-9]{2,9}\.tsv", run_path.name):
        errors.append("Filename does not match official TeamID_RunID.tsv pattern.")

    question_ids = read_ids_tsv(questions_path)
    qset = set(question_ids)
    if len(question_ids) != len(qset):
        errors.append("Question file itself contains duplicate qids.")

    qpc_ids = read_qpc_ids(qpc_path)
    by_qid = defaultdict(list)
    seen_pairs, seen_ranks = set(), set()
    tags = set()

    with open(run_path, encoding="utf-8", newline="") as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                errors.append(f"Blank line at {n}.")
                continue
            if "\t" not in raw:
                errors.append(f"Line {n} is not tab-separated.")
                continue
            parts = raw.rstrip("\n").split("\t")
            if len(parts) != 6:
                errors.append(f"Line {n}: expected 6 columns, got {len(parts)}.")
                continue

            qid, q0, docid, rank_s, score_s, tag = parts
            if qid not in qset:
                errors.append(f"Line {n}: unknown qid {qid}.")
            if q0 != "Q0":
                errors.append(f"Line {n}: Q0 column must be literal Q0.")
            if not tag:
                errors.append(f"Line {n}: empty tag.")
            tags.add(tag)

            try:
                rank = int(rank_s)
                if rank < 1:
                    raise ValueError
            except Exception:
                errors.append(f"Line {n}: rank must be a positive integer.")
                continue

            try:
                score = float(score_s)
                if not math.isfinite(score):
                    raise ValueError
            except Exception:
                errors.append(f"Line {n}: score must be finite numeric.")
                continue

            pair = (qid, docid)
            if pair in seen_pairs:
                errors.append(f"Line {n}: duplicate (qid, docid) pair {pair}.")
            seen_pairs.add(pair)

            rk = (qid, rank)
            if rk in seen_ranks:
                errors.append(f"Line {n}: duplicate rank {rank} for qid {qid}.")
            seen_ranks.add(rk)

            if docid != "-1" and docid not in qpc_ids:
                errors.append(f"Line {n}: passage-id {docid} is not in the QPC.")

            by_qid[qid].append((rank, score, docid, n))

    if len(tags) > 1:
        errors.append(f"Multiple run tags found: {sorted(tags)}")

    missing = [qid for qid in question_ids if qid not in by_qid]
    if missing:
        errors.append(f"Missing questions in run: {missing}")

    for qid, rows in by_qid.items():
        rows = sorted(rows)
        if len(rows) > 10:
            errors.append(f"qid {qid}: {len(rows)} rows; official maximum is 10.")

        ranks = [r[0] for r in rows]
        expected = list(range(1, len(rows) + 1))
        if ranks != expected:
            errors.append(f"qid {qid}: ranks are {ranks}, expected contiguous {expected}.")

        minus_one = [r for r in rows if r[2] == "-1"]
        if minus_one:
            if len(rows) != 1 or len(minus_one) != 1 or rows[0][0] != 1:
                errors.append(
                    f"qid {qid}: -1 must be the only row and must be at rank 1."
                )
        else:
            scores = [r[1] for r in rows]
            for i in range(1, len(scores)):
                if not scores[i-1] > scores[i]:
                    errors.append(
                        f"qid {qid}: scores must be strictly decreasing by rank; "
                        f"ranks {i} and {i+1} are not."
                    )
                    break

    qrels_audit = None
    if qrels_path:
        qrels = read_qrels(qrels_path)
        qrel_qids = set(qrels)
        missing_qrels = sorted(qset - qrel_qids)
        extra_qrels = sorted(qrel_qids - qset)
        bad_qrel_docs = []
        zero_qids = []
        for qid, docs in qrels.items():
            if "-1" in docs:
                zero_qids.append(qid)
            for docid in docs:
                if docid != "-1" and docid not in qpc_ids:
                    bad_qrel_docs.append((qid, docid))
        if missing_qrels:
            warnings.append(
                "Questions present in the official question file but absent from qrels: "
                + ", ".join(missing_qrels)
            )
        if extra_qrels:
            errors.append(f"Qrels contain qids absent from question file: {extra_qrels}")
        if bad_qrel_docs:
            errors.append(f"Qrels reference passage IDs absent from QPC: {bad_qrel_docs[:20]}")
        qrels_audit = {
            "qrels_qids": len(qrel_qids),
            "zero_answer_qids": sorted(zero_qids),
            "questions_without_qrels": missing_qrels,
        }

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "question_count": len(question_ids),
        "run_question_count": len(by_qid),
        "run_row_count": sum(len(v) for v in by_qid.values()),
        "qpc_passage_count": len(qpc_ids),
        "tags": sorted(tags),
        "qrels_audit": qrels_audit,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--qpc", required=True)
    ap.add_argument("--qrels")
    ap.add_argument("--json-report")
    args = ap.parse_args()

    report = validate(args.run, args.questions, args.qpc, args.qrels)
    print("STRICT VALIDATION:", "PASSED" if report["passed"] else "FAILED")
    print(f"Questions: run={report['run_question_count']} expected={report['question_count']}")
    print(f"Rows: {report['run_row_count']}; QPC passages: {report['qpc_passage_count']}")
    for w in report["warnings"]:
        print("WARNING:", w)
    for e in report["errors"]:
        print("ERROR:", e)

    if args.json_report:
        Path(args.json_report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    raise SystemExit(0 if report["passed"] else 1)

if __name__ == "__main__":
    main()
