"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
# ============================================================
# Import Required Libraries
# Notebook 10B – Baseline 2: Dense Retrieval
# ============================================================

import ast
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path.cwd().parents[2]

CODE_DIR = PROJECT_ROOT / "04_Code"

MODELS_DIR = CODE_DIR / "Models"

MODEL_PATH = MODELS_DIR / "multilingual-e5-base"

EXPERIMENTS_DIR = PROJECT_ROOT / "05_Experiments"

PREPROCESSING_DIR = (
    EXPERIMENTS_DIR /
    "Preprocessing"
)

PASSAGE_PREPARATION_DIR = (
    EXPERIMENTS_DIR /
    "Passage_Preparation"
)

DENSE_RETRIEVAL_DIR = (
    EXPERIMENTS_DIR /
    "Dense_Retrieval"
)

# Create output directory if it does not already exist
DENSE_RETRIEVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Device Configuration
# ------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("PROJECT PATHS INITIALIZED")
print("=" * 70)

print(f"Project Root       : {PROJECT_ROOT}")
print(f"Models Folder      : {MODELS_DIR}")
print(f"Embedding Model    : {MODEL_PATH.name}")
print(f"Experiments Folder : {EXPERIMENTS_DIR}")
print(f"Output Folder      : {DENSE_RETRIEVAL_DIR}")
print(f"Processing Device  : {DEVICE}")

# %% [notebook cell 4]
# ============================================================
# Load Query and Passage Collections
# ============================================================

import ast

# ------------------------------------------------------------
# Input Files
# ------------------------------------------------------------

QUERY_FILE = (
    PREPROCESSING_DIR /
    "QuranQA_Preprocessed_Queries.csv"
)

PASSAGE_FILE = (
    PASSAGE_PREPARATION_DIR /
    "QuranQA_Processed_Passage_Collection.csv"
)

# ------------------------------------------------------------
# Load Datasets
# ------------------------------------------------------------

queries_df = pd.read_csv(QUERY_FILE)

passages_df = pd.read_csv(PASSAGE_FILE)

# ------------------------------------------------------------
# Convert Query Terms to Python Lists
# ------------------------------------------------------------

queries_df["Query_Terms"] = queries_df["Query_Terms"].apply(ast.literal_eval)

# ------------------------------------------------------------
# Validate Required Columns
# ------------------------------------------------------------

required_query_columns = {
    "Question_ID",
    "Question",
    "Query_Terms"
}

required_passage_columns = {
    "Passage_ID",
    "Passage_Text"
}

missing_query = required_query_columns - set(queries_df.columns)
missing_passage = required_passage_columns - set(passages_df.columns)

if missing_query:
    raise ValueError(
        f"Missing query columns: {sorted(missing_query)}"
    )

if missing_passage:
    raise ValueError(
        f"Missing passage columns: {sorted(missing_passage)}"
    )

# ------------------------------------------------------------
# Display Summary
# ------------------------------------------------------------

print("=" * 70)
print("DENSE RETRIEVAL RESOURCES LOADED")
print("=" * 70)

print(f"Questions : {len(queries_df):,}")
print(f"Passages  : {len(passages_df):,}")

print("\nQuery Dataset Columns")

print(list(queries_df.columns))

print("\nPassage Dataset Columns")

print(list(passages_df.columns))

print("\nSample Query")

display(
    queries_df[
        [
            "Question_ID",
            "Question",
            "Query_Terms"
        ]
    ].head(1)
)

print("\nSample Passage")

display(
    passages_df[
        [
            "Passage_ID",
            "Passage_Text"
        ]
    ].head(1)
)

# %% [notebook cell 6]
# ============================================================
# Load Local multilingual-e5-base Embedding Model
# ============================================================

print("=" * 70)
print("LOADING DENSE RETRIEVAL MODEL")
print("=" * 70)

print("Embedding Model : multilingual-e5-base (Local)")
print(f"Model Path      : {MODEL_PATH}")
print(f"Device          : {DEVICE}")

# ------------------------------------------------------------
# Load Sentence Transformer Model
# ------------------------------------------------------------

model = SentenceTransformer(
    str(MODEL_PATH),
    device=DEVICE
)

print("\nModel loaded successfully.")

# ------------------------------------------------------------
# Verify the Model
# ------------------------------------------------------------

test_query = "query: من هم قوم شعيب؟"

test_embedding = model.encode(
    test_query,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("\nModel Verification")
print("-" * 40)

print(f"Test Query         : {test_query}")
print(f"Embedding Shape    : {test_embedding.shape}")
print(f"Embedding Dimension: {len(test_embedding)}")
print(f"Vector Norm        : {np.linalg.norm(test_embedding):.4f}")

print("\nFirst 10 Dimensions")

print(np.round(test_embedding[:10], 4))

# %% [notebook cell 8]
# ============================================================
# Generate and Cache Passage Embeddings
# ============================================================

import os

EMBEDDINGS_FILE = (
    DENSE_RETRIEVAL_DIR /
    "Passage_Embeddings.npy"
)

METADATA_FILE = (
    DENSE_RETRIEVAL_DIR /
    "Passage_Metadata.csv"
)

# ------------------------------------------------------------
# Load Existing Embeddings (if available)
# ------------------------------------------------------------

if EMBEDDINGS_FILE.exists():

    print("=" * 70)
    print("LOADING EXISTING PASSAGE EMBEDDINGS")
    print("=" * 70)

    passage_embeddings = np.load(EMBEDDINGS_FILE)

    passage_metadata = pd.read_csv(METADATA_FILE)

# ------------------------------------------------------------
# Otherwise Generate Embeddings
# ------------------------------------------------------------

else:

    print("=" * 70)
    print("GENERATING PASSAGE EMBEDDINGS")
    print("=" * 70)

    # Prepare E5 Input Format

    passage_texts = [
        "passage: " + text
        for text in passages_df["Passage_Text"]
    ]

    # Generate Embeddings

    passage_embeddings = model.encode(
        passage_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save Embeddings

    np.save(
        EMBEDDINGS_FILE,
        passage_embeddings
    )

    # Save Metadata

    passage_metadata = passages_df[
        [
            "Passage_ID",
            "Passage_Text"
        ]
    ].copy()

    passage_metadata.to_csv(
        METADATA_FILE,
        index=False,
        encoding="utf-8-sig"
    )

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nEmbedding Summary")
print("-" * 40)

print(f"Passages           : {len(passage_metadata):,}")
print(f"Embedding Shape    : {passage_embeddings.shape}")
print(f"Embedding Dimension: {passage_embeddings.shape[1]}")

print("\nSample Passage")

display(
    passage_metadata.head(1)
)

print("\nFirst 10 Values of First Embedding")

print(
    np.round(
        passage_embeddings[0][:10],
        4
    )
)

# %% [notebook cell 10]
# ============================================================
# Generate and Cache Query Embeddings
# ============================================================

QUERY_EMBEDDINGS_FILE = (
    DENSE_RETRIEVAL_DIR /
    "Query_Embeddings.npy"
)

QUERY_METADATA_FILE = (
    DENSE_RETRIEVAL_DIR /
    "Query_Metadata.csv"
)

# ------------------------------------------------------------
# Load Existing Query Embeddings
# ------------------------------------------------------------

if QUERY_EMBEDDINGS_FILE.exists():

    print("=" * 70)
    print("LOADING EXISTING QUERY EMBEDDINGS")
    print("=" * 70)

    query_embeddings = np.load(QUERY_EMBEDDINGS_FILE)

    query_metadata = pd.read_csv(QUERY_METADATA_FILE)

# ------------------------------------------------------------
# Otherwise Generate Query Embeddings
# ------------------------------------------------------------

else:

    print("=" * 70)
    print("GENERATING QUERY EMBEDDINGS")
    print("=" * 70)

    # Prepare E5 Input Format

    query_texts = [
        "query: " + text
        for text in queries_df["Question"]
    ]

    # Generate Embeddings

    query_embeddings = model.encode(
        query_texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save Embeddings

    np.save(
        QUERY_EMBEDDINGS_FILE,
        query_embeddings
    )

    # Save Metadata

    query_metadata = queries_df[
        [
            "Question_ID",
            "Question"
        ]
    ].copy()

    query_metadata.to_csv(
        QUERY_METADATA_FILE,
        index=False,
        encoding="utf-8-sig"
    )

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nEmbedding Summary")
print("-" * 40)

print(f"Queries             : {len(query_metadata):,}")
print(f"Embedding Shape     : {query_embeddings.shape}")
print(f"Embedding Dimension : {query_embeddings.shape[1]}")

print("\nSample Query")

display(
    query_metadata.head(1)
)

print("\nFirst 10 Values of First Query Embedding")

print(
    np.round(
        query_embeddings[0][:10],
        4
    )
)

# %% [notebook cell 12]
# ============================================================
# Compute Query–Passage Similarity Matrix
# ============================================================

print("=" * 70)
print("COMPUTING QUERY–PASSAGE SIMILARITY")
print("=" * 70)

# ------------------------------------------------------------
# Compute Cosine Similarity
# (Embeddings are already normalized)
# ------------------------------------------------------------

similarity_matrix = np.matmul(
    query_embeddings,
    passage_embeddings.T
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nSimilarity Matrix Summary")
print("-" * 40)

print(f"Matrix Shape       : {similarity_matrix.shape}")
print(f"Total Comparisons  : {similarity_matrix.size:,}")

print(f"\nMinimum Similarity : {similarity_matrix.min():.4f}")
print(f"Maximum Similarity : {similarity_matrix.max():.4f}")
print(f"Average Similarity : {similarity_matrix.mean():.4f}")

print("\nSimilarity Scores for the First Query")
print("-" * 40)

print(
    np.round(
        similarity_matrix[0][:10],
        4
    )
)

# %% [notebook cell 14]
# ============================================================
# Build Dense Retrieval Results
# ============================================================

print("=" * 70)
print("BUILDING DENSE RETRIEVAL RESULTS")
print("=" * 70)

retrieval_results = []

# ------------------------------------------------------------
# Build Query–Passage Similarity Table
# ------------------------------------------------------------

for query_idx, query_row in queries_df.iterrows():

    similarities = similarity_matrix[query_idx]

    for passage_idx, similarity in enumerate(similarities):

        retrieval_results.append({

            "Question_ID": query_row["Question_ID"],
            "Question": query_row["Question"],
            "Passage_ID": passages_df.iloc[passage_idx]["Passage_ID"],
            "Similarity_Score": float(similarity)

        })

# ------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------

dense_df = pd.DataFrame(retrieval_results)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nDense Retrieval Completed")
print("-" * 40)

print(f"Query–Passage Pairs : {len(dense_df):,}")
print(f"Questions Processed : {dense_df['Question_ID'].nunique():,}")
print(f"Passages Compared   : {passages_df.shape[0]:,}")

print("\nSimilarity Statistics")

print(f"Minimum Score : {dense_df['Similarity_Score'].min():.4f}")
print(f"Maximum Score : {dense_df['Similarity_Score'].max():.4f}")
print(f"Average Score : {dense_df['Similarity_Score'].mean():.4f}")

print("\nSample Retrieval Scores")

display(
    dense_df.head(10)
)

# %% [notebook cell 16]
# ============================================================
# Rank Dense Retrieval Results
# ============================================================

print("=" * 70)
print("RANKING DENSE RETRIEVAL RESULTS")
print("=" * 70)

# ------------------------------------------------------------
# Sort by Similarity Score
# ------------------------------------------------------------

dense_df = dense_df.sort_values(
    by=["Question_ID", "Similarity_Score"],
    ascending=[True, False]
).reset_index(drop=True)

# ------------------------------------------------------------
# Assign Rank
# ------------------------------------------------------------

dense_df["Rank"] = (
    dense_df
    .groupby("Question_ID")
    .cumcount()
    + 1
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nRanking Completed")
print("-" * 40)

print(f"Total Ranked Pairs : {len(dense_df):,}")
print(f"Questions Ranked   : {dense_df['Question_ID'].nunique():,}")

print("\nTop-ranked passage for the first five questions")

display(
    dense_df
    .groupby("Question_ID")
    .head(1)
    .head(5)
)

print("\nRank Distribution")

print(
    dense_df["Rank"].describe()
)

# %% [notebook cell 18]
# ============================================================
# Keep Only Top-100 Ranked Passages
# ============================================================

TOP_K = 100

print("=" * 70)
print(f"SELECTING TOP-{TOP_K} RETRIEVED PASSAGES")
print("=" * 70)

# ------------------------------------------------------------
# Keep Only Top-K Passages per Question
# ------------------------------------------------------------

dense_topk_df = (
    dense_df
    .groupby("Question_ID", group_keys=False)
    .head(TOP_K)
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nTop-K Retrieval Summary")
print("-" * 40)

print(f"Questions              : {dense_topk_df['Question_ID'].nunique():,}")
print(f"Top-K per Question     : {TOP_K}")
print(f"Total Retrieved Pairs  : {len(dense_topk_df):,}")

print("\nTop-5 Retrieved Passages for the First Question")

display(
    dense_topk_df
    .groupby("Question_ID")
    .head(5)
    .head(5)
)

# %% [notebook cell 20]
# ============================================================
# Export Dense Retrieval Results
# ============================================================

ALL_RESULTS_FILE = (
    DENSE_RETRIEVAL_DIR /
    "Dense_Retrieval_All_Ranked.csv"
)

TOP100_RESULTS_FILE = (
    DENSE_RETRIEVAL_DIR /
    "Dense_Retrieval_Top100.csv"
)

print("=" * 70)
print("EXPORTING DENSE RETRIEVAL RESULTS")
print("=" * 70)

# ------------------------------------------------------------
# Export Complete Ranked Results
# ------------------------------------------------------------

dense_df.to_csv(
    ALL_RESULTS_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# Export Top-100 Results
# ------------------------------------------------------------

dense_topk_df.to_csv(
    TOP100_RESULTS_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# Export Summary
# ------------------------------------------------------------

print("\nExport Completed Successfully")
print("-" * 40)

print(f"Complete Ranked Results : {ALL_RESULTS_FILE.name}")
print(f"Top-100 Results         : {TOP100_RESULTS_FILE.name}")

print("\nFile Statistics")
print("-" * 40)

print(f"Complete Ranked Rows : {len(dense_df):,}")
print(f"Top-100 Rows         : {len(dense_topk_df):,}")

print("\nFiles Saved To")

print(DENSE_RETRIEVAL_DIR)
