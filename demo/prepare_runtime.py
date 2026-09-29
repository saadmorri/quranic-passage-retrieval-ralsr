import os

import torch

from config import load_config
from data_loader import ResourceStore
from retrieval.dense_e5_retriever import DenseE5Retriever


if __name__ == "__main__":
    # Runtime-only CPU allocation; this does not change model inputs, pooling,
    # normalization, scoring, or any frozen thesis artifact.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = load_config()
    resources = ResourceStore.load(config)
    DenseE5Retriever(
        resources.qpc,
        config.e5_model_dir,
        config.dense_embedding_cache,
        config.dense_metadata_cache,
    )
    print(config.dense_embedding_cache)
