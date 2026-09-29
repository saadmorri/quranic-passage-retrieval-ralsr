from __future__ import annotations

import json

import pandas as pd

from config import RUN_KEYS
from data_loader import ResourceStore, parse_json_list


class BenchmarkStore:
    def __init__(self, resources: ResourceStore):
        self.resources = resources
        self.questions = resources.questions.copy()
        self.qpc = resources.qpc.set_index("Passage_ID")["Passage_Text"].to_dict()
        fixed = resources.fixed_ralsr_full.copy()
        fixed["Question_ID"] = pd.to_numeric(fixed["Question_ID"], errors="raise").astype(int)
        fixed["Passage_ID"] = fixed["Passage_ID"].astype(str)
        self.fixed_lookup = {
            (int(row.Question_ID), str(row.Passage_ID)): row
            for row in fixed.itertuples(index=False)
            if str(row.Passage_ID) != "-1"
        }

    def list_questions(self):
        return [
            {"qid": int(row.QID), "question": row.Question, "split": row.Split}
            for row in self.questions.sort_values(["Split", "QID"]).itertuples(index=False)
        ]

    def _feature_payload(self, qid, pid):
        row = self.fixed_lookup.get((int(qid), str(pid)))
        if row is None:
            return None
        return {
            "query_terms": parse_json_list(row.Query_Terms),
            "query_roots": parse_json_list(row.Query_Roots),
            "query_root_sources": parse_json_list(row.Query_Root_Sources),
            "shared_roots": parse_json_list(row.Shared_Roots),
            "unresolved_terms": [
                item.get("Query_Term")
                for item in parse_json_list(row.Query_Root_Sources)
                if not item.get("Root_AR")
            ],
            "features": {
                "Root_Coverage": float(row.Root_Coverage),
                "Root_Overlap": float(row.Root_Overlap),
                "Semantic_Overlap": float(row.Semantic_Overlap),
                "Jaccard_Root_Similarity": float(row.Jaccard_Root_Similarity),
                "Passage_Root_Count": float(row.Passage_Root_Count),
                "Semantic_Count": float(row.Semantic_Count),
            },
            "ralsr_score": float(row.RALSR_Score),
        }

    def results(self, qid: int, system: str, top_n: int):
        question_row = self.questions.loc[self.questions["QID"] == int(qid)]
        if question_row.empty:
            raise KeyError(f"Unknown official qid: {qid}")
        question = question_row.iloc[0]
        run_key = RUN_KEYS[system]
        run = self.resources.runs[run_key][question["Split"]]
        subset = run.loc[run["QID"] == int(qid)].sort_values("Rank").head(top_n)
        rows = []
        for item in subset.itertuples(index=False):
            pid = str(item.Passage_ID)
            structural = pid == "-1"
            row = {
                "rank": int(item.Rank),
                "passage_id": pid,
                "passage_text": "" if structural else self.qpc.get(pid, ""),
                "score": float(item.Score),
                "structural_no_answer": structural,
            }
            if system in {"Fixed RALSR", "RALSR + CrossEncoder"} and not structural:
                row["explanation"] = self._feature_payload(qid, pid)
            rows.append(row)
        return {
            "mode": "frozen_benchmark",
            "qid": int(qid),
            "split": question["Split"],
            "question": question["Question"],
            "system": system,
            "top_n": top_n,
            "unjudged_note": (
                "Unjudged in the published test qrels; excluded from official scoring."
                if int(qid) == 504 else None
            ),
            "results": rows,
        }
