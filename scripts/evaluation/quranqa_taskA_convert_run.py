#!/usr/bin/env python3
"""
Convert an internal retrieval CSV/TSV into a strict Qur'an QA 2023 Task A
TREC-style TSV run.

Important:
- Gold qrels are NOT accepted by this converter.
- No-answer qids, if supplied, must come from the SYSTEM's own abstention rule,
  never from gold qrels.
- The output score is rank-derived by default so the official evaluator's
  score-based ordering is guaranteed to match the intended rank ordering.
"""

from __future__ import annotations
import argparse
import csv
import math
import re
from pathlib import Path
import pandas as pd

QID_CANDIDATES = ["qid", "question_id", "Question_ID", "Query_ID", "query_id"]
DOCID_CANDIDATES = ["docid", "passage_id", "Passage_ID", "Document_ID", "doc_id"]
SCORE_CANDIDATES = [
    "score", "Score", "BM25_Score", "Dense_Score", "RALSR_Score",
    "Final_Score", "CrossEncoder_Score", "CSR_Score"
]
RANK_CANDIDATES = ["rank", "Rank", "Original_Rank"]

def detect_col(df, explicit, candidates, label):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"{label} column '{explicit}' not found. Columns: {list(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not detect {label} column. Specify it explicitly.")

def read_table(path: Path):
    return pd.read_csv(path, sep=None, engine="python", dtype=str)

def read_question_ids(path: Path):
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    return df.iloc[:, 0].astype(str).tolist()

def read_no_answer_ids(path: Path | None):
    if path is None:
        return set()
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            ids.add(value)
    return ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True,
                    help="Must be a valid TeamID_RunID.tsv filename")
    ap.add_argument("--questions", required=True,
                    help="Official split question TSV; used only for completeness/order")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--qid-col")
    ap.add_argument("--docid-col")
    ap.add_argument("--score-col")
    ap.add_argument("--rank-col")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-answer-qids",
                    help="One SYSTEM-predicted no-answer qid per line. Never use gold qrels.")
    ap.add_argument("--score-mode", choices=["rank-derived", "preserve"], default="rank-derived")
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    if args.top_k < 1 or args.top_k > 10:
        raise SystemExit("--top-k must be between 1 and 10 for Task A.")
    if not re.fullmatch(r"[A-Za-z0-9]{3,9}_[A-Za-z0-9]{2,9}\.tsv", out.name):
        raise SystemExit(
            "Output filename must follow official TeamID_RunID.tsv naming, "
            "e.g. saad_r01.tsv"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise SystemExit("Tag must be a simple non-space identifier.")

    df = read_table(inp)
    qid_col = detect_col(df, args.qid_col, QID_CANDIDATES, "qid")
    docid_col = detect_col(df, args.docid_col, DOCID_CANDIDATES, "docid")
    score_col = detect_col(df, args.score_col, SCORE_CANDIDATES, "score")
    rank_col = None
    if args.rank_col:
        rank_col = detect_col(df, args.rank_col, RANK_CANDIDATES, "rank")
    else:
        for c in RANK_CANDIDATES:
            if c in df.columns:
                rank_col = c
                break

    work = df[[qid_col, docid_col, score_col] + ([rank_col] if rank_col else [])].copy()
    work.columns = ["qid", "docid", "score"] + (["source_rank"] if rank_col else [])
    work["qid"] = work["qid"].astype(str)
    work["docid"] = work["docid"].astype(str)
    work["score"] = pd.to_numeric(work["score"], errors="raise")
    if not work["score"].map(math.isfinite).all():
        raise SystemExit("Input contains non-finite scores.")

    dup = work.duplicated(["qid", "docid"], keep=False)
    if dup.any():
        raise SystemExit(
            "Duplicate (qid, docid) pairs exist in the internal run. "
            "Resolve them before conversion."
        )

    if (work["docid"] == "-1").any():
        raise SystemExit(
            "Internal retrieval input contains passage id -1. Supply system-generated "
            "no-answer qids explicitly with --no-answer-qids instead of encoding "
            "implicit no-answer rows in retrieval output."
        )

    if "source_rank" in work:
        work["source_rank"] = pd.to_numeric(work["source_rank"], errors="raise").astype(int)
        work = work.sort_values(
            ["qid", "source_rank", "score", "docid"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
    else:
        work = work.sort_values(
            ["qid", "score", "docid"],
            ascending=[True, False, True],
            kind="mergesort",
        )

    question_ids = read_question_ids(Path(args.questions))
    qset = set(question_ids)
    unknown = set(work["qid"]) - qset
    if unknown:
        raise SystemExit(f"Internal run contains unknown question IDs: {sorted(unknown)[:20]}")

    no_answer = read_no_answer_ids(Path(args.no_answer_qids) if args.no_answer_qids else None)
    unknown_na = no_answer - qset
    if unknown_na:
        raise SystemExit(f"No-answer file contains unknown question IDs: {sorted(unknown_na)}")

    rows = []
    for qid in question_ids:
        if qid in no_answer:
            rows.append([qid, "Q0", "-1", 1, "1", args.tag])
            continue

        qrows = work[work["qid"] == qid].head(args.top_k)
        if qrows.empty:
            raise SystemExit(
                f"Question {qid} has no retrieved passage and was not explicitly "
                "predicted as no-answer."
            )

        if args.score_mode == "preserve":
            preserved_scores = qrows["score"].astype(float).tolist()
            if any(
                preserved_scores[i - 1] <= preserved_scores[i]
                for i in range(1, len(preserved_scores))
            ):
                raise SystemExit(
                    f"Question {qid} has preserved scores that are not strictly "
                    "decreasing in intended rank order. Use rank-derived scores or "
                    "repair the internal ranking before conversion."
                )

        for rank, (_, r) in enumerate(qrows.iterrows(), start=1):
            if args.score_mode == "rank-derived":
                # Strictly decreasing and ranking-equivalent. Model scores remain
                # preserved in the original internal run.
                official_score = f"{args.top_k - rank + 1:.6f}"
            else:
                official_score = f"{float(r['score']):.12g}"
            rows.append([qid, "Q0", r["docid"], rank, official_score, args.tag])

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows for {len(question_ids)} questions: {out}")
    if no_answer:
        print(f"System-predicted no-answer questions: {len(no_answer)}")
    else:
        print(
            "No system no-answer predictions supplied. This is benchmark-valid, "
            "but gold zero-answer questions will receive zero credit."
        )

if __name__ == "__main__":
    main()
