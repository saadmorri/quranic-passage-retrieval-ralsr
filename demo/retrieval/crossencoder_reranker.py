from __future__ import annotations

import re


def passage_key(passage_id):
    numbers = tuple(int(value) for value in re.findall(r"\d+", str(passage_id)))
    return numbers or (10**9,)


class CrossEncoderReranker:
    def __init__(self, model_dir):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(model_dir), device="cpu")

    def rerank(self, question, candidates, top_n=10):
        if not candidates:
            return []
        scores = self.model.predict(
            [[question, row["passage_text"]] for row in candidates],
            batch_size=16,
            show_progress_bar=False,
        )
        rows = []
        for candidate, score in zip(candidates, scores):
            row = dict(candidate)
            row["original_ralsr_rank"] = int(candidate["rank"])
            row["original_ralsr_score"] = float(candidate["score"])
            row["score"] = float(score)
            rows.append(row)
        rows.sort(key=lambda row: (-row["score"], passage_key(row["passage_id"])))
        for rank, row in enumerate(rows[:top_n], start=1):
            row["rank"] = rank
        return rows[:top_n]
