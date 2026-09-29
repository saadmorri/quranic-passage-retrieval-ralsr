from __future__ import annotations

from collections import defaultdict

from data_loader import parse_json_list


WEIGHTS = {
    "Root_Coverage": 0.30,
    "Root_Overlap": 0.25,
    "Semantic_Overlap": 0.20,
    "Jaccard_Root_Similarity": 0.15,
    "Passage_Root_Count": 0.05,
    "Semantic_Count": 0.05,
}


class RALSRRetriever:
    def __init__(self, passage_representation, normalization):
        self.passages = {}
        self.root_index = defaultdict(set)
        for row in passage_representation.itertuples(index=False):
            roots = {str(value).strip() for value in parse_json_list(row.Roots) if str(value).strip()}
            semantics = {str(value).strip() for value in parse_json_list(row.Semantic_Meanings) if str(value).strip()}
            pid = str(row.Passage_ID)
            self.passages[pid] = {
                "passage_text": row.Passage_Text,
                "roots": roots,
                "semantics": semantics,
                "root_count": len(roots),
                "semantic_count": len(semantics),
            }
            for root in roots:
                self.root_index[root].add(pid)
        self.norm = {
            row.Feature: (float(row.Training_Minimum), float(row.Training_Maximum))
            for row in normalization.itertuples(index=False)
        }

    def _normalize(self, feature, value):
        minimum, maximum = self.norm[feature]
        return 0.0 if maximum == minimum else (value - minimum) / (maximum - minimum)

    def retrieve(self, query_info, top_n=10, candidate_depth=100):
        query_roots = set(query_info["accepted_roots"])
        query_semantics = set(query_info["semantics"])
        candidates = set()
        for root in query_roots:
            candidates.update(self.root_index.get(root, set()))
        if not candidates:
            return [], 0
        rows = []
        for pid in candidates:
            passage = self.passages[pid]
            shared_roots = sorted(query_roots & passage["roots"])
            query_only = sorted(query_roots - passage["roots"])
            shared_semantics = sorted(query_semantics & passage["semantics"])
            root_overlap = float(len(shared_roots))
            semantic_overlap = float(len(shared_semantics))
            root_coverage = round(len(shared_roots) / len(query_roots), 4) if query_roots else 0.0
            union_size = len(query_only) + passage["root_count"]
            jaccard = round(root_overlap / union_size, 4) if union_size else 0.0
            raw = {
                "Root_Coverage": root_coverage,
                "Root_Overlap": root_overlap,
                "Semantic_Overlap": semantic_overlap,
                "Jaccard_Root_Similarity": jaccard,
                "Passage_Root_Count": float(passage["root_count"]),
                "Semantic_Count": float(passage["semantic_count"]),
            }
            normalized = {name: self._normalize(name, value) for name, value in raw.items()}
            score = round(sum(WEIGHTS[name] * normalized[name] for name in WEIGHTS), 6)
            rows.append({
                "passage_id": pid,
                "passage_text": passage["passage_text"],
                "score": score,
                "shared_roots": shared_roots,
                "features": raw,
                "normalized_features": normalized,
            })
        rows.sort(key=lambda row: (-row["score"], row["passage_id"]))
        rows = rows[:candidate_depth]
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows[:top_n] if top_n else rows, len(candidates)

    def candidates(self, query_info, depth=100):
        return self.retrieve(query_info, top_n=0, candidate_depth=depth)
