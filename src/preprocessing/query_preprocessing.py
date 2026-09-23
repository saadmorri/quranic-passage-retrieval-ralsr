"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
# ============================================================
# Configure Project Path
# ============================================================

import os
import sys
from pathlib import Path


def find_workspace_root(start=None):
    """
    Locate the current thesis workspace without hard-coding a user-specific
    absolute Windows path.

    An optional THESIS_WORKSPACE_ROOT environment variable can be used when
    the notebook server is started outside the thesis workspace.
    """

    configured = os.environ.get("THESIS_WORKSPACE_ROOT")

    if configured:
        configured_path = Path(configured).expanduser().resolve()

        if (configured_path / "02_Methodology_Evidence").is_dir():
            return configured_path

        raise FileNotFoundError(
            "THESIS_WORKSPACE_ROOT is set, but it does not contain "
            "'02_Methodology_Evidence'."
        )

    start = Path(start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "02_Methodology_Evidence").is_dir():
            return candidate

        expected_child = (
            candidate /
            "Master Thesis - Quranic Passage Retrieval - Writing Workspace"
        )

        if (expected_child / "02_Methodology_Evidence").is_dir():
            return expected_child

    raise FileNotFoundError(
        "Could not locate the thesis workspace. Start Jupyter from the "
        "workspace (or one of its subfolders), or set THESIS_WORKSPACE_ROOT."
    )


PROJECT_ROOT = find_workspace_root()

METHODOLOGY_DIR = PROJECT_ROOT / "02_Methodology_Evidence"

QAC_ROOT_EXTRACTION_DIR = (
    METHODOLOGY_DIR /
    "QAC and Root Extraction"
)

# The reusable preprocessing module is stored beside this notebook
if str(QAC_ROOT_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(QAC_ROOT_EXTRACTION_DIR))

print("Project Root:")
print(PROJECT_ROOT)

print("\nQAC / Root Extraction Directory:")
print(QAC_ROOT_EXTRACTION_DIR)

# %% [notebook cell 3]
# ============================================================
# Standard Library
# ============================================================

from pathlib import Path
from collections import Counter
import re

# ============================================================
# Third-Party Libraries
# ============================================================

import pandas as pd

# %% [notebook cell 4]
# ============================================================
# Import Arabic Preprocessing Module
# ============================================================

import arabic_preprocessing as anlp

print("✅ Arabic preprocessing module imported successfully.")

# %% [notebook cell 5]
# ============================================================
# Configure Display Settings
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 120)

print("✅ Display settings configured successfully.")

# %% [notebook cell 6]
# ============================================================
# Define Project Paths
# ============================================================

# QuranQA benchmark resources
QURANQA_DIR = (
    METHODOLOGY_DIR /
    "Data and Resources" /
    "QuranQA"
)

# Input files
TRAIN_FILE = QURANQA_DIR / "QQA23_TaskA_ayatec_v1.2_train.tsv"
DEV_FILE   = QURANQA_DIR / "QQA23_TaskA_ayatec_v1.2_dev.tsv"
TEST_FILE  = QURANQA_DIR / "QQA23_TaskA_ayatec_v1.2_test.tsv"

# Point-2 corrected outputs are kept separate from preserved historical outputs
OUTPUT_DIR = (
    QAC_ROOT_EXTRACTION_DIR /
    "Corrected_Outputs" /
    "Point_2_Query_Root_Provenance"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Display paths
print("=" * 70)
print("PROJECT PATHS")
print("=" * 70)
print(f"Project Root       : {PROJECT_ROOT}")
print(f"QuranQA Directory  : {QURANQA_DIR}")
print(f"Output Directory   : {OUTPUT_DIR}")

print("\nInput Validation")
print(f"Train exists       : {TRAIN_FILE.exists()}")
print(f"Development exists : {DEV_FILE.exists()}")
print(f"Test exists        : {TEST_FILE.exists()}")
print("=" * 70)

# %% [notebook cell 7]
# ============================================================
# Load the QuranQA Datasets
# ============================================================

# Load each dataset (files do not contain column headers)
train_df = pd.read_csv(
    TRAIN_FILE,
    sep="\t",
    header=None,
    names=["Question_ID", "Question"]
)

dev_df = pd.read_csv(
    DEV_FILE,
    sep="\t",
    header=None,
    names=["Question_ID", "Question"]
)

test_df = pd.read_csv(
    TEST_FILE,
    sep="\t",
    header=None,
    names=["Question_ID", "Question"]
)

# Display dataset sizes
dataset_summary = pd.DataFrame({
    "Dataset": ["Training", "Development", "Test"],
    "Questions": [
        len(train_df),
        len(dev_df),
        len(test_df)
    ]
})

print("=" * 60)
print("QURANQA DATASETS LOADED")
print("=" * 60)

display(dataset_summary)

print(f"\nTotal Questions: {dataset_summary['Questions'].sum()}")

# %% [notebook cell 8]
# ============================================================
# Combine the QuranQA Datasets
# ============================================================

# Add dataset labels
train_df["Dataset"] = "Train"
dev_df["Dataset"] = "Development"
test_df["Dataset"] = "Test"

# Combine all questions
questions_df = pd.concat(
    [train_df, dev_df, test_df],
    ignore_index=True
)

# Reorder columns
questions_df = questions_df[
    ["Dataset", "Question_ID", "Question"]
]

print("=" * 60)
print("COMBINED QUESTION DATASET")
print("=" * 60)

print(f"Total Questions : {len(questions_df)}")

display(questions_df.head(10))

# %% [notebook cell 11]
# ============================================================
# Apply DD-001: Arabic Text Normalization
# ============================================================

# Create a copy to preserve the original dataset
normalized_df = questions_df.copy()

# Apply normalization
normalized_df["Normalized_Question"] = (
    normalized_df["Question"]
    .apply(anlp.normalize_arabic)
)

print("=" * 70)
print("DD-001 COMPLETED: ARABIC TEXT NORMALIZATION")
print("=" * 70)

print(f"Total Questions Processed : {len(normalized_df)}")

display(
    normalized_df[
        [
            "Question_ID",
            "Question",
            "Normalized_Question"
        ]
    ].head(10)
)

# %% [notebook cell 12]
# ============================================================
# Evaluate the Effect of Arabic Text Normalization
# ============================================================

# Identify questions modified by normalization
changed_questions = normalized_df[
    normalized_df["Question"] != normalized_df["Normalized_Question"]
].copy()

# Calculate statistics
total_questions = len(normalized_df)
changed_count = len(changed_questions)
unchanged_count = total_questions - changed_count
changed_percentage = round((changed_count / total_questions) * 100, 2)

# ------------------------------------------------------------
# Create summary table
# ------------------------------------------------------------

summary_df = pd.DataFrame({
    "Metric": [
        "Total Questions",
        "Changed Questions",
        "Unchanged Questions",
        "Percentage Changed"
    ],
    "Value": [
        total_questions,
        changed_count,
        unchanged_count,
        f"{changed_percentage}%"
    ]
})

print("=" * 70)
print("DD-001 VALIDATION: ARABIC TEXT NORMALIZATION")
print("=" * 70)

display(summary_df)

# ------------------------------------------------------------
# Display representative examples
# ------------------------------------------------------------

print("\nRepresentative Examples of Normalized Questions\n")

display(
    changed_questions[
        ["Question_ID", "Question", "Normalized_Question"]
    ].head(10)
)

# ------------------------------------------------------------
# Interpretation
# ------------------------------------------------------------

print("\nInterpretation:")

if changed_count == 0:
    print("- No questions required normalization.")
else:
    print(
        f"- Arabic text normalization modified "
        f"{changed_count} out of {total_questions} questions "
        f"({changed_percentage}%)."
    )
    print("- Most changes correspond to orthographic normalization "
          "such as Alif variants and Alif Maqsura.")
    print("- Canonical spellings (e.g., Ta Marbuta) were preserved "
          "to maintain compatibility with the Quranic Arabic Corpus (QAC) "
          "and Maqāyīs al-Lughah.")

# %% [notebook cell 13]
# ============================================================
# Apply DD-002: Punctuation Removal
# ============================================================

# Create a copy
punctuation_df = normalized_df.copy()

# Preserve a surface-form version of the original question for
# term-level provenance. The retrieval text itself continues to use
# the normalized question exactly as in the original methodology.
punctuation_df["Original_Clean_Question"] = (
    punctuation_df["Question"]
    .apply(anlp.remove_punctuation)
)

# Apply punctuation removal to the normalized retrieval text
punctuation_df["Clean_Question"] = (
    punctuation_df["Normalized_Question"]
    .apply(anlp.remove_punctuation)
)

print("=" * 70)
print("DD-002 COMPLETED: PUNCTUATION REMOVAL")
print("=" * 70)

display(
    punctuation_df[
        [
            "Question_ID",
            "Original_Clean_Question",
            "Normalized_Question",
            "Clean_Question"
        ]
    ].head(10)
)

# %% [notebook cell 15]
# ============================================================
# Validate DD-002: Punctuation Removal
# ============================================================

# Identify modified questions
changed_questions = punctuation_df[
    punctuation_df["Normalized_Question"] != punctuation_df["Clean_Question"]
].copy()

# Calculate statistics
total_questions = len(punctuation_df)
changed_count = len(changed_questions)
unchanged_count = total_questions - changed_count
changed_percentage = round((changed_count / total_questions) * 100, 2)

# Summary table
summary_df = pd.DataFrame({
    "Metric": [
        "Total Questions",
        "Modified Questions",
        "Unchanged Questions",
        "Percentage Modified"
    ],
    "Value": [
        total_questions,
        changed_count,
        unchanged_count,
        f"{changed_percentage}%"
    ]
})

print("=" * 70)
print("DD-002 VALIDATION: PUNCTUATION REMOVAL")
print("=" * 70)

display(summary_df)

print("\nRepresentative Examples\n")

display(
    changed_questions[
        [
            "Question_ID",
            "Normalized_Question",
            "Clean_Question"
        ]
    ].head(10)
)

print("\nInterpretation:")

print("- Arabic punctuation was successfully removed.")
print("- The lexical content of the questions was preserved.")
print("- The resulting text is ready for tokenization.")

# %% [notebook cell 17]
# ============================================================
# Apply DD-003: Arabic Tokenization
# ============================================================

# Create a copy to preserve the previous stage
tokenized_df = punctuation_df.copy()

# Surface tokens are retained only for provenance
tokenized_df["Original_Tokens"] = (
    tokenized_df["Original_Clean_Question"]
    .apply(anlp.tokenize)
)

# Retrieval tokens remain the normalized tokens
tokenized_df["Tokens"] = (
    tokenized_df["Clean_Question"]
    .apply(anlp.tokenize)
)

# Normalization used here is character-level and should preserve token
# boundaries. Assert this so a future preprocessing change cannot silently
# corrupt original-term -> normalized-term provenance.
token_alignment_ok = (
    tokenized_df["Original_Tokens"].str.len()
    == tokenized_df["Tokens"].str.len()
).all()

if not token_alignment_ok:
    bad_rows = tokenized_df[
        tokenized_df["Original_Tokens"].str.len()
# JUPYTER:         != tokenized_df["Tokens"].str.len()
    ][
        [
            "Question_ID",
            "Question",
            "Original_Tokens",
            "Tokens"
        ]
    ]

    raise ValueError(
        "Original/normalized token alignment failed. "
        f"Affected questions: {len(bad_rows)}"
    )

print("=" * 70)
print("DD-003 COMPLETED: ARABIC TOKENIZATION")
print("=" * 70)

print(f"Total Questions Processed : {len(tokenized_df)}")
print(f"Token Alignment Valid     : {token_alignment_ok}")

# Display representative examples
display(
    tokenized_df[
        [
            "Question_ID",
            "Original_Tokens",
            "Tokens"
        ]
    ].head(10)
)

# %% [notebook cell 19]
# ============================================================
# Arabic Stopword List
# ============================================================

# The source list is preserved in readable Arabic orthography.
RAW_STOPWORDS = {
    "أن",
    "أو",
    "أين",
    "إلى",
    "إن",
    "التي",
    "الذي",
    "الذين",
    "اللاتي",
    "ب",
    "بماذا",
    "بين",
    "تلك",
    "ثم",
    "ذلك",
    "على",
    "عن",
    "في",
    "ك",
    "كان",
    "كانت",
    "كم",
    "كيف",
    "ل",
    "لا",
    "لم",
    "لماذا",
    "لن",
    "ما",
    "ماذا",
    "متى",
    "مع",
    "من",
    "هذا",
    "هذه",
    "هل",
    "هم",
    "هن",
    "هو",
    "هي",
    "و"
}

# Query tokens have already passed through normalize_arabic().
# Normalize the stopword set with the same function before comparison.
# This corrects cases such as إن -> ان, أين -> اين, إلى -> الي,
# على -> علي, أو -> او, and متى -> متي.
STOPWORDS = {
    anlp.normalize_arabic(word)
    for word in RAW_STOPWORDS
}

normalization_changes = sorted({
    (word, anlp.normalize_arabic(word))
    for word in RAW_STOPWORDS
    if word != anlp.normalize_arabic(word)
})

print("=" * 70)
print("ARABIC STOPWORDS LOADED")
print("=" * 70)
print(f"Source Stopwords     : {len(RAW_STOPWORDS)}")
print(f"Normalized Stopwords : {len(STOPWORDS)}")

print("\nStopwords changed by DD-001 normalization")
display(
    pd.DataFrame(
        normalization_changes,
        columns=["Source_Form", "Normalized_Form"]
    )
)

# %% [notebook cell 21]
# ============================================================
# Apply DD-004: Stopword Filtering
# ============================================================

# Create a copy to preserve the previous stage
filtered_df = tokenized_df.copy()

# Apply stopword filtering to the normalized token stream
filtered_df["Content_Tokens"] = (
    filtered_df["Tokens"]
    .apply(lambda tokens: anlp.remove_stopwords(tokens, STOPWORDS))
)

print("=" * 70)
print("DD-004 COMPLETED: STOPWORD FILTERING")
print("=" * 70)

print(f"Total Questions Processed : {len(filtered_df)}")

# Display representative examples
display(
    filtered_df[
        [
            "Question_ID",
            "Tokens",
            "Content_Tokens"
        ]
    ].head(10)
)

# %% [notebook cell 23]
# ============================================================
# DD-005: Automatic Query Term Selection
# ============================================================

def select_query_terms(tokens):
    """
    Select candidate query terms by removing duplicates
    while preserving the original token order.
    """

    selected = []
    seen = set()

    for token in tokens:
        token = token.strip()

        if not token:
            continue

        if token not in seen:
            selected.append(token)
            seen.add(token)

    return selected


def build_query_term_provenance(original_tokens, normalized_tokens, query_terms):
    """
    Preserve the first surface form corresponding to every selected
    normalized query term.

    Returns
    -------
    list of dict
        Ordered provenance records with token position, original term,
        and normalized term.
    """

    if len(original_tokens) != len(normalized_tokens):
        raise ValueError("Original and normalized token lists are misaligned.")

    selected_set = set(query_terms)
    seen = set()
    provenance = []

    for position, (original, normalized) in enumerate(
        zip(original_tokens, normalized_tokens)
    ):
        if normalized in selected_set and normalized not in seen:
            provenance.append({
                "Token_Position": int(position),
                "Original_Term": original,
                "Normalized_Term": normalized
            })
            seen.add(normalized)

    missing = [term for term in query_terms if term not in seen]

    if missing:
        raise ValueError(
            "Could not recover original forms for selected query terms: "
            + ", ".join(missing)
        )

    return provenance


# Create a copy
query_df = filtered_df.copy()

# Apply query term selection
query_df["Query_Terms"] = (
    query_df["Content_Tokens"]
    .apply(select_query_terms)
)

# Preserve original -> normalized query-term lineage
query_df["Query_Term_Provenance"] = query_df.apply(
    lambda row: build_query_term_provenance(
        row["Original_Tokens"],
        row["Tokens"],
        row["Query_Terms"]
    ),
    axis=1
)

print("=" * 70)
print("DD-005 COMPLETED: AUTOMATIC QUERY TERM SELECTION")
print("=" * 70)

print(f"Total Questions Processed : {len(query_df)}")

display(
    query_df[
        [
            "Question_ID",
            "Content_Tokens",
            "Query_Terms",
            "Query_Term_Provenance"
        ]
    ].head(10)
)

# %% [notebook cell 24]
# ============================================================
# Save the Preprocessed Query Dataset
# ============================================================

import json

output_file = OUTPUT_DIR / "QuranQA_Preprocessed_Queries_Corrected.csv"

export_df = query_df.copy()

# Preserve the existing list columns for compatibility, while exporting the
# new provenance record as standards-compliant JSON.
export_df["Query_Term_Provenance"] = (
    export_df["Query_Term_Provenance"]
    .apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False
        )
    )
)

export_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 70)
print("PREPROCESSED QUERY DATASET SAVED")
print("=" * 70)

print(f"Output File:\n{output_file}")
print(f"\nTotal Questions Saved: {len(export_df)}")

print(
    "\nPreservation rule: the historical "
    "'QuranQA_Preprocessed_Queries.csv' file is not overwritten."
)
