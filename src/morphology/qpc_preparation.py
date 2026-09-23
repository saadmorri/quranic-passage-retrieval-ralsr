"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
# ============================================================
# Import Required Libraries
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

from IPython.display import display

# ------------------------------------------------------------
# Display Options
# ------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

print("=" * 70)
print("LIBRARIES LOADED")
print("=" * 70)

print("✓ pathlib")
print("✓ json")
print("✓ numpy")
print("✓ pandas")
print("✓ IPython.display")

# %% [notebook cell 3]
# ============================================================
# Define Project Paths
# ============================================================

# Resolve the current Writing Workspace without a personal absolute path.
NOTEBOOK_DIR = Path.cwd().resolve()


def find_workspace_root(start_path):
    """Locate the thesis Writing Workspace from the execution directory."""
    start_path = Path(start_path).resolve()

    for candidate in [start_path, *start_path.parents]:
        authority_file = (
            candidate /
            "00_Project_Control" /
            "README_Workspace_Authority.txt"
        )
        qpc_file = (
            candidate /
            "02_Methodology_Evidence" /
            "Data and Resources" /
            "QuranQA" /
            "QQA23_TaskA_QPC_v1.1.tsv"
        )

        if authority_file.exists() and qpc_file.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the authoritative thesis Writing Workspace "
        "from the current execution directory."
    )


PROJECT_ROOT = find_workspace_root(NOTEBOOK_DIR)

# ------------------------------------------------------------
# Current Methodology-Evidence Directories
# ------------------------------------------------------------

DATA_RESOURCES_DIR = (
    PROJECT_ROOT /
    "02_Methodology_Evidence" /
    "Data and Resources"
)

QURANQA_DIR = DATA_RESOURCES_DIR / "QuranQA"

OUTPUT_DIR = (
    DATA_RESOURCES_DIR /
    "Corrected_Outputs" /
    "QPC_Passage_Side_Validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Input Files
# ------------------------------------------------------------

PASSAGE_COLLECTION_FILE = QURANQA_DIR / "QQA23_TaskA_QPC_v1.1.tsv"

print("=" * 70)
print("PROJECT PATHS")
print("=" * 70)

print(f"Project Root              : {PROJECT_ROOT}")
print(f"QuranQA Directory         : {QURANQA_DIR}")
print(f"Output Directory          : {OUTPUT_DIR}")

print("\nInput Files")

print(f"Passage Collection        : {PASSAGE_COLLECTION_FILE.name}")

print("\nValidation")

print(f"Passage Collection Exists : {PASSAGE_COLLECTION_FILE.exists()}")

# %% [notebook cell 5]
# ============================================================
# Load Quran Passage Collection
# ============================================================

# ------------------------------------------------------------
# Load Dataset (No Header in TSV)
# ------------------------------------------------------------

passages_df = pd.read_csv(
    PASSAGE_COLLECTION_FILE,
    sep="\t",
    header=None,
    names=["Passage_ID", "Passage_Text"]
)

# ------------------------------------------------------------
# Dataset Overview
# ------------------------------------------------------------

print("=" * 70)
print("QURAN PASSAGE COLLECTION LOADED")
print("=" * 70)

print(f"Total Passages : {len(passages_df):,}")
print(f"Total Columns  : {len(passages_df.columns)}")

print("\nColumns")

print(list(passages_df.columns))

print("\nSample Records")

display(passages_df.head())

# %% [notebook cell 7]
# ============================================================
# Explore Quran Passage Collection Structure
# ============================================================

# ------------------------------------------------------------
# Dataset Information
# ------------------------------------------------------------

print("=" * 70)
print("PASSAGE COLLECTION STRUCTURE")
print("=" * 70)

print("\nDataset Shape")
print(f"Rows    : {passages_df.shape[0]:,}")
print(f"Columns : {passages_df.shape[1]}")

print("\nData Types")
display(passages_df.dtypes.to_frame(name="Data Type"))

# ------------------------------------------------------------
# Missing Values
# ------------------------------------------------------------

print("\nMissing Values")

missing_df = (
    passages_df
    .isnull()
    .sum()
    .rename("Missing Values")
    .to_frame()
)

display(missing_df)

# ------------------------------------------------------------
# Duplicate Passage IDs
# ------------------------------------------------------------

duplicate_ids = passages_df["Passage_ID"].duplicated().sum()

print(f"\nDuplicate Passage IDs : {duplicate_ids}")

# ------------------------------------------------------------
# Duplicate Passage Texts
# ------------------------------------------------------------

duplicate_texts = passages_df["Passage_Text"].duplicated().sum()

print(f"Duplicate Passages    : {duplicate_texts}")

# ------------------------------------------------------------
# Passage ID Validation
# ------------------------------------------------------------

unique_ids = passages_df["Passage_ID"].nunique()

print(f"Unique Passage IDs    : {unique_ids:,}")

print("\nSample Passage IDs")

display(passages_df["Passage_ID"].head(15))

# %% [notebook cell 9]
# ============================================================
# Parse Passage Identifiers
# ============================================================

# ------------------------------------------------------------
# Parse Passage_ID
# ------------------------------------------------------------

def parse_passage_id(pid):

    surah, ayah_range = pid.split(":")

    if "-" in ayah_range:

        start_ayah, end_ayah = ayah_range.split("-")

    else:

        start_ayah = end_ayah = ayah_range

    start_ayah = int(start_ayah)
    end_ayah = int(end_ayah)

    return pd.Series({

        "Surah": int(surah),
        "Start_Ayah": start_ayah,
        "End_Ayah": end_ayah,
        "Ayah_Count": end_ayah - start_ayah + 1

    })

# ------------------------------------------------------------
# Apply Parsing
# ------------------------------------------------------------

passages_df = pd.concat(

    [

        passages_df,

        passages_df["Passage_ID"].apply(parse_passage_id)

    ],

    axis=1

)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 70)
print("PASSAGE IDENTIFIERS PARSED")
print("=" * 70)

print(f"Total Passages : {len(passages_df):,}")

print(f"Unique Surahs  : {passages_df['Surah'].nunique()}")

print(f"Average Ayahs per Passage : {passages_df['Ayah_Count'].mean():.2f}")

print()

display(passages_df.head(10))

# %% [notebook cell 11]
# ============================================================
# Descriptive Analysis of the Passage Collection
# ============================================================

# ------------------------------------------------------------
# Compute Passage Length Statistics
# ------------------------------------------------------------

passages_df["Character_Count"] = passages_df["Passage_Text"].str.len()

passages_df["Word_Count"] = passages_df["Passage_Text"].str.split().str.len()

# ------------------------------------------------------------
# Summary Statistics
# ------------------------------------------------------------

print("=" * 70)
print("PASSAGE COLLECTION STATISTICS")
print("=" * 70)

print(f"Average Characters per Passage : {passages_df['Character_Count'].mean():.2f}")
print(f"Minimum Characters             : {passages_df['Character_Count'].min()}")
print(f"Maximum Characters             : {passages_df['Character_Count'].max()}")

print()

print(f"Average Words per Passage      : {passages_df['Word_Count'].mean():.2f}")
print(f"Minimum Words                  : {passages_df['Word_Count'].min()}")
print(f"Maximum Words                  : {passages_df['Word_Count'].max()}")

# ------------------------------------------------------------
# Passages per Surah
# ------------------------------------------------------------

surah_summary = (
    passages_df
    .groupby("Surah")
    .size()
    .reset_index(name="Passage_Count")
)

print("\nSample Surah Statistics")

display(surah_summary.head(15))

print("\nSample Passage Statistics")

display(
    passages_df[
        [
            "Passage_ID",
            "Ayah_Count",
            "Word_Count",
            "Character_Count"
        ]
    ].head(20)
)

# %% [notebook cell 13]
# ============================================================
# Build Passage Lookup Dictionary
# ============================================================

passage_lookup = {}

# ------------------------------------------------------------
# Build Dictionary
# ------------------------------------------------------------

for _, row in passages_df.iterrows():

    passage_lookup[row["Passage_ID"]] = {

        "Passage_Text": row["Passage_Text"],

        "Surah": row["Surah"],

        "Start_Ayah": row["Start_Ayah"],

        "End_Ayah": row["End_Ayah"],

        "Ayah_Count": row["Ayah_Count"],

        "Word_Count": row["Word_Count"],

        "Character_Count": row["Character_Count"]

    }

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 70)
print("PASSAGE LOOKUP DICTIONARY CREATED")
print("=" * 70)

print(f"Dictionary Entries : {len(passage_lookup):,}")

sample_keys = list(passage_lookup.keys())[:5]

print("\nSample Passage IDs")

for key in sample_keys:
    print(f"• {key}")

print("\nSample Lookup Entry")

first_key = sample_keys[0]

display(
    pd.DataFrame([passage_lookup[first_key]])
)

# %% [notebook cell 15]
# ============================================================
# Export Processed Passage Collection
# ============================================================

OUTPUT_FILE = OUTPUT_DIR / "QuranQA_Processed_Passage_Collection.csv"

# ------------------------------------------------------------
# Export CSV
# ------------------------------------------------------------

passages_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# Export Summary
# ------------------------------------------------------------

print("=" * 70)
print("PROCESSED PASSAGE COLLECTION EXPORTED")
print("=" * 70)

print(f"Passages Exported : {len(passages_df):,}")
print(f"Output File       : {OUTPUT_FILE.name}")
print(f"Location          : {OUTPUT_DIR}")

print("\nExport completed successfully.")
