"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
# ============================================================
# Import Required Libraries
# Notebook 10A – Baseline 1: BM25 Sparse Retrieval
# ============================================================

import ast
from pathlib import Path

import numpy as np
import pandas as pd


def find_workspace_root(start: Path) -> Path:
    """Locate the thesis writing workspace from the execution directory."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (
            (candidate / "00_Project_Control").is_dir()
            and (candidate / "02_Methodology_Evidence").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not locate the thesis workspace. Run this notebook from within "
        "the Master Thesis - Quranic Passage Retrieval - Writing Workspace tree."
    )


PROJECT_ROOT = find_workspace_root(Path.cwd())
METHODOLOGY_DIR = PROJECT_ROOT / "02_Methodology_Evidence"

QUERY_FILE = (
    METHODOLOGY_DIR
    / "QAC and Root Extraction"
    / "Corrected_Outputs"
    / "Point_2_Query_Root_Provenance"
    / "QuranQA_Preprocessed_Queries_Corrected.csv"
)

PASSAGE_FILE = (
    METHODOLOGY_DIR
    / "Data and Resources"
    / "Corrected_Outputs"
    / "QPC_Passage_Side_Validation"
    / "QuranQA_Processed_Passage_Collection.csv"
)

BM25_OUTPUT_DIR = (
    METHODOLOGY_DIR
    / "Baselines"
    / "Corrected_Outputs"
    / "Base_BM25_Corrected_Query"
)
BM25_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PROJECT PATHS INITIALIZED")
print("=" * 70)
print(f"Project Root        : {PROJECT_ROOT}")
print(f"Corrected Query File: {QUERY_FILE}")
print(f"Processed QPC File  : {PASSAGE_FILE}")
print(f"Output Folder       : {BM25_OUTPUT_DIR}")

# %% [notebook cell 3]
# ============================================================
# Load and Validate Authoritative BM25 Resources
# ============================================================

queries_df = pd.read_csv(QUERY_FILE, encoding="utf-8-sig")
passages_df = pd.read_csv(PASSAGE_FILE, encoding="utf-8-sig")

queries_df["Query_Terms"] = queries_df["Query_Terms"].apply(ast.literal_eval)
passages_df["Tokens"] = passages_df["Passage_Text"].fillna("").str.split()

selected_term_count = int(queries_df["Query_Terms"].map(len).sum())

if len(queries_df) != 251 or queries_df["Question_ID"].nunique() != 251:
    raise ValueError("Corrected query input must contain 251 unique question IDs.")
if selected_term_count != 1342:
    raise ValueError(f"Expected 1,342 corrected query terms, found {selected_term_count:,}.")
if 504 not in set(pd.to_numeric(queries_df["Question_ID"], errors="raise").astype(int)):
    raise ValueError("qid 504 is missing from the corrected query input.")
if len(passages_df) != 1266 or passages_df["Passage_ID"].nunique() != 1266:
    raise ValueError("Processed QPC must contain 1,266 unique passage IDs.")
if passages_df["Passage_ID"].isna().any() or passages_df["Passage_Text"].isna().any():
    raise ValueError("Processed QPC contains a blank passage ID or passage text.")

print("=" * 70)
print("AUTHORITATIVE BM25 RESOURCES LOADED")
print("=" * 70)
print(f"Questions             : {len(queries_df):,}")
print(f"Selected Query Terms  : {selected_term_count:,}")
print(f"Passages              : {len(passages_df):,}")
print(f"Unique Passage IDs    : {passages_df['Passage_ID'].nunique():,}")
print("Qrel data loaded      : No")

print("\nSample Query")
display(queries_df[["Question_ID", "Question", "Query_Terms"]].head(1))

print("\nSample Passage")
display(passages_df[["Passage_ID", "Passage_Text", "Tokens"]].head(1))

# %% [notebook cell 4]
# ============================================================
# Build BM25 Corpus and Index
# ============================================================

from rank_bm25 import BM25Okapi

BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

bm25_corpus = passages_df["Tokens"].tolist()
bm25_index = BM25Okapi(
    bm25_corpus,
    k1=BM25_K1,
    b=BM25_B,
    epsilon=BM25_EPSILON,
)

document_lengths = [len(document) for document in bm25_corpus]

print("=" * 70)
print("BM25 INDEX CREATED")
print("=" * 70)
print(f"Indexed Passages      : {len(bm25_corpus):,}")
print(f"k1 / b / epsilon      : {BM25_K1} / {BM25_B} / {BM25_EPSILON}")
print(f"Average Document Size : {np.mean(document_lengths):.2f} tokens")
print(f"Minimum Length        : {np.min(document_lengths)} tokens")
print(f"Maximum Length        : {np.max(document_lengths)} tokens")
print("\nSample Tokenized Passage\n")
print(bm25_corpus[0][:30])

# %% [notebook cell 6]
# ============================================================
# BM25 Passage Retrieval and Scoring
# ============================================================

bm25_results = []

# ------------------------------------------------------------
# Score Every Question Against Every Passage
# ------------------------------------------------------------

for _, query_row in queries_df.iterrows():

    query_terms = query_row["Query_Terms"]

    # Compute BM25 scores for the entire corpus
    scores = bm25_index.get_scores(query_terms)

    # Store one record per passage
    for passage_row, score in zip(
        passages_df.itertuples(index=False),
        scores
    ):

        bm25_results.append({

            "Question_ID": query_row["Question_ID"],
            "Question": query_row["Question"],
            "Passage_ID": passage_row.Passage_ID,
            "BM25_Score": float(score)

        })

# ------------------------------------------------------------
# Convert to DataFrame
# ------------------------------------------------------------

bm25_df = pd.DataFrame(bm25_results)

# ------------------------------------------------------------
# Display Summary
# ------------------------------------------------------------

print("=" * 70)
print("BM25 RETRIEVAL COMPLETED")
print("=" * 70)

print(f"Question–Passage Pairs : {len(bm25_df):,}")
print(f"Questions Processed    : {queries_df['Question_ID'].nunique():,}")
print(f"Indexed Passages       : {passages_df['Passage_ID'].nunique():,}")

print(f"\nMinimum Score : {bm25_df['BM25_Score'].min():.4f}")
print(f"Maximum Score : {bm25_df['BM25_Score'].max():.4f}")
print(f"Average Score : {bm25_df['BM25_Score'].mean():.4f}")

print("\nSample Retrieval Scores")

display(
    bm25_df.head(10)
)

# %% [notebook cell 8]
# ============================================================
# Rank BM25 Retrieval Results Deterministically
# ============================================================

# Descending BM25 score; equal scores use ascending official Passage_ID.
bm25_df = bm25_df.sort_values(
    by=["Question_ID", "BM25_Score", "Passage_ID"],
    ascending=[True, False, True],
    kind="mergesort",
).reset_index(drop=True)

bm25_df["Rank"] = bm25_df.groupby("Question_ID").cumcount() + 1

print("=" * 70)
print("BM25 RANKING COMPLETED")
print("=" * 70)
print(f"Total Ranked Pairs : {len(bm25_df):,}")
print(f"Questions Ranked   : {bm25_df['Question_ID'].nunique():,}")
print("Tie Ordering       : Ascending Passage_ID")

print("\nTop-ranked passage for the first five questions\n")
display(bm25_df.groupby("Question_ID").head(1).head(5))

print("\nRank Distribution\n")
display(bm25_df["Rank"].describe())

# %% [notebook cell 10]
# ============================================================
# Retrieve Top-100 BM25 Passages
# ============================================================

TOP_K = 100

# ------------------------------------------------------------
# Keep Only Top-K Ranked Passages per Question
# ------------------------------------------------------------

bm25_topk_df = (
    bm25_df
    .groupby("Question_ID", group_keys=False)
    .head(TOP_K)
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# Display Summary
# ------------------------------------------------------------

print("=" * 70)
print("TOP-100 BM25 RETRIEVAL COMPLETED")
print("=" * 70)

print(f"Questions Retrieved : {bm25_topk_df['Question_ID'].nunique():,}")
print(f"Top-K               : {TOP_K}")
print(f"Retrieved Records   : {len(bm25_topk_df):,}")

print("\nSample Top-10 Retrieved Passages")

display(
    bm25_topk_df.head(10)
)

# %% [notebook cell 12]
# ============================================================
# Export Corrected BM25 Retrieval Results
# ============================================================

ALL_RESULTS_FILE = BM25_OUTPUT_DIR / "BM25_Retrieval_All_Ranked_Corrected.csv"
TOP100_RESULTS_FILE = BM25_OUTPUT_DIR / "BM25_Retrieval_Top100_Corrected.csv"

query_lookup = queries_df[["Question_ID", "Query_Terms"]]
bm25_df = bm25_df.merge(query_lookup, on="Question_ID", how="left")
bm25_topk_df = bm25_topk_df.merge(query_lookup, on="Question_ID", how="left")

column_order = [
    "Question_ID",
    "Question",
    "Query_Terms",
    "Passage_ID",
    "BM25_Score",
    "Rank",
]
bm25_df = bm25_df[column_order]
bm25_topk_df = bm25_topk_df[column_order]

bm25_df.to_csv(ALL_RESULTS_FILE, index=False, encoding="utf-8-sig")
bm25_topk_df.to_csv(TOP100_RESULTS_FILE, index=False, encoding="utf-8-sig")

print("=" * 70)
print("CORRECTED BM25 RETRIEVAL RESULTS EXPORTED")
print("=" * 70)
print(f"Complete Ranked Results : {ALL_RESULTS_FILE.name}")
print(f"Top-100 Results         : {TOP100_RESULTS_FILE.name}")
print(f"Complete Ranked Rows    : {len(bm25_df):,}")
print(f"Top-100 Rows            : {len(bm25_topk_df):,}")
print(f"Questions               : {bm25_topk_df['Question_ID'].nunique():,}")
print(f"Top-K                   : {TOP_K}")
print(f"Files Saved To          : {BM25_OUTPUT_DIR}")

# %% [notebook cell 13]
# ============================================================
# Display Top-10 BM25 Retrieval Results
# ============================================================

SAMPLE_QUESTION_ID = 101

sample_results = (
    bm25_topk_df[
        bm25_topk_df["Question_ID"] == SAMPLE_QUESTION_ID
    ]
    .head(10)
    .copy()
)

print("=" * 70)
print("TOP 10 BM25 RETRIEVAL RESULTS")
print("=" * 70)

print(f"Question ID : {SAMPLE_QUESTION_ID}")

print(
    f"Question    : "
    f"{sample_results.iloc[0]['Question']}"
)

print("\nTop-10 Retrieved Passages\n")

display(
    sample_results[
        [
            "Rank",
            "Passage_ID",
            "BM25_Score"
        ]
    ]
)

# %% [notebook cell 15]
# ============================================================
# Qualitative Inspection of BM25 Retrieval Results
# ============================================================

SAMPLE_QUESTION_ID = 101

# ------------------------------------------------------------
# Retrieve Top-10 Results
# ------------------------------------------------------------

sample_results = (
    bm25_topk_df[
        bm25_topk_df["Question_ID"] == SAMPLE_QUESTION_ID
    ]
    .head(10)
    .copy()
)

# ------------------------------------------------------------
# Attach Passage Text
# ------------------------------------------------------------

sample_results = sample_results.merge(
    passages_df[
        [
            "Passage_ID",
            "Passage_Text"
        ]
    ],
    on="Passage_ID",
    how="left"
)

# ------------------------------------------------------------
# Display Results
# ------------------------------------------------------------

print("=" * 70)
print("TOP 10 BM25 RETRIEVAL RESULTS")
print("=" * 70)

print(f"Question ID : {SAMPLE_QUESTION_ID}")
print(f"Question    : {sample_results.iloc[0]['Question']}")
print(f"Query Terms : {sample_results.iloc[0]['Query_Terms']}")

print("\nTop-10 Retrieved Passages\n")

display(
    sample_results[
        [
            "Rank",
            "Passage_ID",
            "BM25_Score",
            "Passage_Text"
        ]
    ]
)
