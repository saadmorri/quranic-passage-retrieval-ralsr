from __future__ import annotations

import numpy as np
import pandas as pd


class DenseE5Retriever:
    def __init__(self, qpc, model_dir, embedding_cache, metadata_cache):
        from sentence_transformers import SentenceTransformer

        self.qpc = qpc[["Passage_ID", "Passage_Text"]].copy()
        self.qpc["Passage_ID"] = self.qpc["Passage_ID"].astype(str)
        self.model = SentenceTransformer(str(model_dir), device="cpu")
        self.embedding_cache = embedding_cache
        self.metadata_cache = metadata_cache
        if embedding_cache.exists() and metadata_cache.exists():
            self.passage_embeddings = np.load(embedding_cache)
            metadata = pd.read_csv(metadata_cache, dtype=str, keep_default_na=False)
            if metadata["Passage_ID"].tolist() != self.qpc["Passage_ID"].tolist():
                raise ValueError("Dense cache passage order does not match official QPC")
        else:
            self.passage_embeddings = self.model.encode(
                ["passage: " + text for text in self.qpc["Passage_Text"].tolist()],
                batch_size=16,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(embedding_cache, self.passage_embeddings)
            self.qpc.to_csv(metadata_cache, index=False, encoding="utf-8-sig")
        if self.passage_embeddings.shape != (1266, 768):
            raise ValueError(f"Unexpected E5 passage embedding shape: {self.passage_embeddings.shape}")

    def retrieve(self, question, top_n=10):
        query = self.model.encode(
            "query: " + question,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = self.passage_embeddings @ query
        rows = [
            {
                "passage_id": str(row.Passage_ID),
                "passage_text": row.Passage_Text,
                "score": float(score),
            }
            for row, score in zip(self.qpc.itertuples(index=False), scores)
        ]
        rows.sort(key=lambda row: (-row["score"], row["passage_id"]))
        for rank, row in enumerate(rows[:top_n], start=1):
            row["rank"] = rank
        return rows[:top_n]
