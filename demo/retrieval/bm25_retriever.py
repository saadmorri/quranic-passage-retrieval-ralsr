from __future__ import annotations

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, processed_qpc):
        self.passages = processed_qpc[["Passage_ID", "Passage_Text"]].copy()
        self.passages["Passage_ID"] = self.passages["Passage_ID"].astype(str)
        corpus = self.passages["Passage_Text"].fillna("").str.split().tolist()
        self.index = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)

    def retrieve(self, query_terms, top_n=10):
        scores = self.index.get_scores(query_terms)
        rows = []
        for passage, score in zip(self.passages.itertuples(index=False), scores):
            rows.append({"passage_id": passage.Passage_ID, "passage_text": passage.Passage_Text, "score": float(score)})
        rows.sort(key=lambda row: (-row["score"], row["passage_id"]))
        for rank, row in enumerate(rows[:top_n], start=1):
            row["rank"] = rank
        return rows[:top_n]
