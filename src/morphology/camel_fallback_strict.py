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
    Locate the current thesis workspace without hard-coding a
    user-specific absolute path.
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

if str(QAC_ROOT_EXTRACTION_DIR) not in sys.path:
    sys.path.insert(0, str(QAC_ROOT_EXTRACTION_DIR))

print("Project Root:")
print(PROJECT_ROOT)

print("\nQAC / Root Extraction Directory:")
print(QAC_ROOT_EXTRACTION_DIR)

# %% [notebook cell 3]
# ============================================================
# Import Required Libraries
# ============================================================

# Standard library
import ast
import json
from pathlib import Path
from collections import Counter
from importlib.metadata import version, PackageNotFoundError

# Data manipulation
import pandas as pd
import numpy as np

# Display
from IPython.display import display

# CAMeL Tools
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

# Project module
import hybrid_root_extractor as hre

try:
    CAMEL_TOOLS_VERSION = version("camel-tools")
except PackageNotFoundError:
    CAMEL_TOOLS_VERSION = "UNKNOWN"

print(f"CAMeL Tools package version : {CAMEL_TOOLS_VERSION}")
print("CAMeL morphology database   : built-in database")

# %% [notebook cell 4]

# Load the built-in morphology database
db = MorphologyDB.builtin_db()

# Create the analyzer
analyzer = Analyzer(db)

print("CAMeL Morphology Analyzer loaded successfully.")

# %% [notebook cell 6]
# ============================================================
# Configure Display Settings
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 150)
pd.set_option("display.width", 1200)

print("=" * 70)
print("DISPLAY SETTINGS CONFIGURED")
print("=" * 70)

# %% [notebook cell 7]
# ============================================================
# Define Project Paths
# ============================================================

CORRECTED_OUTPUT_DIR = (
    QAC_ROOT_EXTRACTION_DIR /
    "Corrected_Outputs" /
    "Point_2_Query_Root_Provenance"
)

CORRECTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Corrected QAC output from Notebook 5
QAC_RESULTS_FILE = (
    CORRECTED_OUTPUT_DIR /
    "QuranQA_QAC_Root_Retrieval_Corrected.csv"
)

print("=" * 70)
print("PROJECT PATHS")
print("=" * 70)

print(f"Project Root      : {PROJECT_ROOT}")
print(f"Input File        : {QAC_RESULTS_FILE}")
print(f"Output Directory  : {CORRECTED_OUTPUT_DIR}")

print("\nValidation")
print(f"Input exists      : {QAC_RESULTS_FILE.exists()}")

# %% [notebook cell 9]
# ============================================================
# Load QAC Root Retrieval Results
# ============================================================

qac_results_df = pd.read_csv(QAC_RESULTS_FILE)

print("=" * 70)
print("QAC ROOT RETRIEVAL RESULTS LOADED")
print("=" * 70)

print(f"Total Questions : {len(qac_results_df):,}")
print(f"Total Columns   : {len(qac_results_df.columns)}")

print("\nColumns")
print(list(qac_results_df.columns))

# ------------------------------------------------------------
# Convert JSON strings back to Python objects
# ------------------------------------------------------------

qac_results_df["Query_Terms"] = (
    qac_results_df["Query_Terms"]
    .apply(json.loads)
)

qac_results_df["Query_Term_Provenance"] = (
    qac_results_df["Query_Term_Provenance"]
    .apply(json.loads)
)

qac_results_df["Lexical_Matches"] = (
    qac_results_df["Lexical_Matches"]
    .apply(json.loads)
)

# Ensure the corrected QAC output is being used.
required_match_fields = {
    "Original_Term",
    "Normalized_Term",
    "Lexical_Matched",
    "Root_Resolved",
    "QAC_Status",
    "QAC_Candidate_Roots"
}

sample_fields = set()

for matches in qac_results_df["Lexical_Matches"]:
    if matches:
        sample_fields = set(matches[0].keys())
        break

missing_match_fields = required_match_fields - sample_fields

if missing_match_fields:
    raise ValueError(
        "Notebook 6 requires the corrected Notebook-5 provenance fields. "
        "Missing: " + ", ".join(sorted(missing_match_fields))
    )

print("\nSample Record")
display(qac_results_df.head(1))

# %% [notebook cell 11]
# ============================================================
# Extract Query Terms Requiring CAMeL Fallback
# ============================================================

fallback_records = []

for _, row in qac_results_df.iterrows():

    for match in row["Lexical_Matches"]:

        if not match["Root_Resolved"]:

            fallback_records.append({

                "Dataset": row["Dataset"],
                "Question_ID": row["Question_ID"],
                "Question": row["Question"],

                "Query_Term": match["Query_Term"],
                "Original_Term": match["Original_Term"],
                "Normalized_Term": match["Normalized_Term"],

                "QAC_Lexical_Matched": match["Lexical_Matched"],
                "QAC_Lookup_Type": match["Lookup_Type"],
                "QAC_Status": match["QAC_Status"],
                "QAC_Candidate_Count": match["QAC_Candidate_Count"],
                "QAC_Candidate_Words": match["QAC_Candidate_Words"],
                "QAC_Candidate_Lemmas": match["QAC_Candidate_Lemmas"],
                "QAC_Candidate_Roots": match["QAC_Candidate_Roots"],
                "QAC_Candidate_POS": match["QAC_Candidate_POS"]

            })

fallback_df = pd.DataFrame(fallback_records)

print("=" * 70)
print("CAMeL FALLBACK TERMS EXTRACTED")
print("=" * 70)

print(f"Total Fallback Terms : {len(fallback_df):,}")

if len(fallback_df):

    print(
        f"Unique Terms         : "
        f"{fallback_df['Normalized_Term'].nunique():,}"
    )

    print("\nQAC Fallback Reasons")
    display(
        fallback_df["QAC_Status"]
        .value_counts()
        .rename_axis("QAC_Status")
        .reset_index(name="Count")
    )

    print("\nFirst Ten Fallback Terms")
    display(fallback_df.head(10))

# %% [notebook cell 13]
# ============================================================
# Rule-Based Morphological Normalization
# ============================================================

# Preserve the Notebook-4 normalized query term and record the additional
# conservative normalization used specifically for CAMeL analysis.
fallback_df["CAMeL_Normalized_Term"] = (
    fallback_df["Normalized_Term"]
    .apply(hre.normalize_for_camel)
)

modified_df = fallback_df[
    fallback_df["Normalized_Term"]
# JUPYTER:     != fallback_df["CAMeL_Normalized_Term"]
].copy()

print("=" * 70)
print("RULE-BASED CAMeL NORMALIZATION COMPLETED")
print("=" * 70)

print(f"Total Fallback Terms : {len(fallback_df):,}")
print(f"Terms Modified       : {len(modified_df):,}")

if len(fallback_df):
    print(
        f"Modification Rate    : "
        f"{100 * len(modified_df) / len(fallback_df):.2f}%"
    )

print("\nSample Normalization Results")

display(
    fallback_df[
        [
            "Original_Term",
            "Normalized_Term",
            "CAMeL_Normalized_Term"
        ]
    ].head(20)
)

# %% [notebook cell 15]
# ============================================================
# Evaluate Rule-Based Normalization
# ============================================================

modified_df = fallback_df[
    fallback_df["Query_Term"] != fallback_df["Normalized_Term"]
].copy()

print("=" * 70)
print("NORMALIZATION EVALUATION")
print("=" * 70)

print(f"Total Terms          : {len(fallback_df):,}")
print(f"Modified Terms       : {len(modified_df):,}")
print(f"Unchanged Terms      : {len(fallback_df)-len(modified_df):,}")

print(
    f"Modification Rate    : "
    f"{100*len(modified_df)/len(fallback_df):.2f}%"
)

print("\nSample Modified Terms")

display(
    modified_df[
        ["Query_Term", "Normalized_Term"]
    ].head(20)
)

# %% [notebook cell 17]
# ============================================================
# Explore CAMeL Morphological Analyses
# ============================================================

# Load the built-in morphology database
db = MorphologyDB.builtin_db()

# Create the analyzer
analyzer = Analyzer(db)

print("=" * 70)
print("CAMeL MORPHOLOGICAL ANALYSIS")
print("=" * 70)

print(f"CAMeL Tools Version : {CAMEL_TOOLS_VERSION}")
print("Database            : built-in morphology database")

# Representative sample
sample_terms = [
    "صبر",
    "سيدة",
    "معنى",
    "سيدنا",
    "عقوبة",
    "ملكة",
    "عقروا"
]

for term in sample_terms:

    print("\n" + "=" * 60)
    print(f"Word : {term}")
    print("=" * 60)

    analyses = analyzer.analyze(term)

    print(f"Number of Analyses : {len(analyses)}")

    if len(analyses) == 0:
        print("No analysis found.")
        continue

    print("\nFirst Analysis\n")

    for key, value in analyses[0].items():
        print(f"{key:20} : {value}")

# %% [notebook cell 19]
# ============================================================
# Test the Hybrid Root Extraction Function
# ============================================================

test_words = [
    "بالصبر",
    "السيدة",
    "معنى",
    "سيدنا",
    "ملكة",
    "عقوبة",
    "عقروا"
]

results = []

for word in test_words:
    results.append(
        hre.extract_root_with_camel(word)
    )

results_df = pd.DataFrame(results)

print("=" * 70)
print("HYBRID ROOT EXTRACTION TEST")
print("=" * 70)

display(results_df)

# %% [notebook cell 20]
# ============================================================
# Inspect Candidate Analyses
# ============================================================

inspection_words = [
    "سيدة",
    "معنى",
    "ملكة",
    "سيدنا"
]

for word in inspection_words:

    print("\n" + "=" * 80)
    print(f"WORD : {word}")
    print("=" * 80)

    analyses = analyzer.analyze(word)

    print(f"Number of Analyses : {len(analyses)}\n")

    for i, analysis in enumerate(analyses[:10], start=1):

        print(f"Analysis {i}")

        print(
            f"Root  : {analysis.get('root')}\n"
            f"Lemma : {analysis.get('lex')}\n"
            f"Stem  : {analysis.get('stem')}\n"
            f"POS   : {analysis.get('pos')}"
        )

        print("-" * 40)

# %% [notebook cell 22]
# ============================================================
# Test the Root Selection Strategy
# ============================================================

test_words = [
    "بالصبر",
    "السيدة",
    "معنى",
    "سيدنا",
    "ملكة",
    "عقوبة",
    "عقروا"
]

results = []

for word in test_words:

    normalized = hre.normalize_for_camel(word)
    analyses = analyzer.analyze(normalized)

    selection = hre.select_best_analysis_with_provenance(
        word,
        analyses
    )

    selected = selection["Selected_Analysis"]
    selected_root_validation = selection["Selected_Root_Validation"]

    results.append({

        "Query_Term": word,
        "CAMeL_Normalized_Term": normalized,
        "Analysis_Count": selection["Analysis_Count"],
        "Top_Score": selection["Top_Score"],
        "Top_Analysis_Count": selection["Top_Analysis_Count"],
        "Top_Roots": selection["Top_Roots"],
        "Top_Root_Validations": selection["Top_Root_Validations"],
        "Resolution_Status": selection["Status"],
        "Root_Resolved": selection["Root_Resolved"],
        "CAMeL_Root_Raw": (
            selected_root_validation["Raw"]
            if selected_root_validation is not None
            else None
        ),
        "CAMeL_Root_Normalized": (
            selected_root_validation["Normalized"]
            if selected_root_validation is not None
            else None
        ),
        "CAMeL_Root_Valid": (
            selected_root_validation["Valid"]
            if selected_root_validation is not None
            else False
        ),
        "CAMeL_Root_Invalid_Reason": (
            selected_root_validation["Invalid_Reason"]
            if selected_root_validation is not None
            else None
        ),
        "Selected_Root": (
            selected_root_validation["Normalized"]
            if selection["Root_Resolved"]
            and selected_root_validation is not None
            else None
        ),
        "Lemma": (
            selected.get("lex")
            if selected is not None
            else None
        ),
        "Stem": (
            selected.get("stem")
            if selected is not None
            else None
        ),
        "POS": (
            selected.get("pos")
            if selected is not None
            else None
        )

    })

results_df = pd.DataFrame(results)

print("=" * 70)
print("ROOT SELECTION STRATEGY TEST")
print("=" * 70)

display(results_df)

# %% [notebook cell 24]
# ============================================================
# Apply the Hybrid Root Extraction Framework
# ============================================================

camel_results = []

for _, row in fallback_df.iterrows():

    query_term = row["Normalized_Term"]
    camel_normalized_term = row["CAMeL_Normalized_Term"]

    analyses = analyzer.analyze(camel_normalized_term)

    selection = hre.select_best_analysis_with_provenance(
        query_term,
        analyses
    )

    selected = selection["Selected_Analysis"]

    selected_root_raw = None
    selected_root_normalized = None
    selected_root_valid = False
    selected_root_invalid_reason = None
    selected_lemma = None
    selected_stem = None
    selected_pos = None

    if selected is not None:
        root_validation = selection["Selected_Root_Validation"]
        selected_root_raw = root_validation["Raw"]
        selected_root_normalized = root_validation["Normalized"]
        selected_root_valid = root_validation["Valid"]
        selected_root_invalid_reason = (
            root_validation["Invalid_Reason"]
        )
        selected_lemma = selected.get("lex")
        selected_stem = selected.get("stem")
        selected_pos = selected.get("pos")

    root_resolved = bool(
        selection["Root_Resolved"]
        and selected_root_valid
        and selected_root_normalized
    )

    selected_analysis_record = None

    if selected is not None:
        selected_analysis_record = {
            "Root_Raw": selected_root_raw,
            "Root_Normalized": selected_root_normalized,
            "Root_Valid": selected_root_valid,
            "Root_Invalid_Reason": selected_root_invalid_reason,
            "Lemma": selected_lemma,
            "Stem": selected_stem,
            "POS": selected_pos
        }

    camel_results.append({

        "Dataset": row["Dataset"],
        "Question_ID": row["Question_ID"],
        "Question": row["Question"],

        "Query_Term": query_term,
        "Original_Term": row["Original_Term"],
        "Normalized_Term": row["Normalized_Term"],
        "CAMeL_Normalized_Term": camel_normalized_term,

        "QAC_Status": row["QAC_Status"],

        # Camel_Matched is retained for compatibility, but now means
        # that a usable root was actually resolved.
        "Camel_Matched": root_resolved,
        "CAMeL_Root_Resolved": root_resolved,
        "CAMeL_Analysis_Found": selection["Analysis_Found"],
        "CAMeL_Analysis_Count": selection["Analysis_Count"],
        "CAMeL_Top_Analysis_Count": selection["Top_Analysis_Count"],
        "CAMeL_Top_Roots": selection["Top_Roots"],
        "CAMeL_Top_Root_Validations": (
            selection["Top_Root_Validations"]
        ),
        "CAMeL_Root_Ambiguous": selection["Root_Ambiguous"],
        "CAMeL_Status": selection["Status"],

        "CAMeL_Root_Raw": selected_root_raw,
        "CAMeL_Root_Normalized": selected_root_normalized,
        "CAMeL_Root_Valid": selected_root_valid,
        "CAMeL_Root_Invalid_Reason": (
            selected_root_invalid_reason
        ),
        "Camel_Root": (
            selected_root_normalized
            if root_resolved
            else None
        ),
        "Lemma": selected_lemma,
        "Stem": selected_stem,
        "POS": selected_pos,

        "Analysis_Score": selection["Top_Score"],
        "CAMeL_Selected_Analysis": selected_analysis_record

    })

camel_results_df = pd.DataFrame(camel_results)

print("=" * 70)
print("HYBRID ROOT EXTRACTION COMPLETED")
print("=" * 70)

print(f"Terms Processed          : {len(camel_results_df):,}")
print(
    f"Roots Resolved by CAMeL  : "
    f"{camel_results_df['CAMeL_Root_Resolved'].sum():,}"
)
print(
    f"Roots Still Unresolved   : "
    f"{(~camel_results_df['CAMeL_Root_Resolved']).sum():,}"
)

print("\nCAMeL Resolution Status")
display(
    camel_results_df["CAMeL_Status"]
    .value_counts()
    .rename_axis("CAMeL_Status")
    .reset_index(name="Count")
)

print("\nCAMeL Invalid-Root Reasons")
display(
    camel_results_df["CAMeL_Root_Invalid_Reason"]
    .dropna()
    .value_counts()
    .rename_axis("Invalid_Reason")
    .reset_index(name="Count")
)

# Strict validity check
invalid_camel = camel_results_df[
    camel_results_df["CAMeL_Root_Resolved"]
    & (
        camel_results_df["Camel_Root"].isna()
        | ~camel_results_df["CAMeL_Root_Valid"]
    )
]

if len(invalid_camel):
    raise ValueError(
        "CAMeL validation failed: root-resolved records "
        "contain missing roots."
    )

print("\nCAMeL root-resolution validation passed.")

print("\nSample Results")
display(camel_results_df.head(10))

# %% [notebook cell 26]
# ============================================================
# Merge QAC and CAMeL Results
# ============================================================

merged_records = []

# Create a lookup for CAMeL fallback results
camel_lookup = {}

for _, row in camel_results_df.iterrows():

    key = (
        row["Question_ID"],
        row["Normalized_Term"]
    )

    camel_lookup[key] = row


def list_or_empty(value):
    if isinstance(value, list):
        return value
    return []


for _, row in qac_results_df.iterrows():

    question_id = row["Question_ID"]
    merged_matches = []

    for match in row["Lexical_Matches"]:

        normalized_term = match["Normalized_Term"]

        # ----------------------------------------------------
        # Authoritative QAC Root
        # ----------------------------------------------------

        if match["Root_Resolved"] and match["Root_AR"]:

            merged_matches.append({

                "Query_Term": match["Query_Term"],
                "Original_Term": match["Original_Term"],
                "Normalized_Term": normalized_term,

                "Matched": True,
                "Root_Resolved": True,
                "Resolution_Status": "RESOLVED_QAC",

                "Root_Source": "QAC",
                "Root_AR": match["Root_AR"],
                "Lemma_AR": match["Lemma_AR"],
                "POS": match["POS"],

                "QAC_Lexical_Matched": match["Lexical_Matched"],
                "QAC_Lookup_Type": match["Lookup_Type"],
                "QAC_Status": match["QAC_Status"],
                "QAC_Candidate_Count": match["QAC_Candidate_Count"],
                "QAC_Candidate_Words": list_or_empty(
                    match["QAC_Candidate_Words"]
                ),
                "QAC_Candidate_Lemmas": list_or_empty(
                    match["QAC_Candidate_Lemmas"]
                ),
                "QAC_Candidate_Roots": list_or_empty(
                    match["QAC_Candidate_Roots"]
                ),
                "QAC_Candidate_POS": list_or_empty(
                    match["QAC_Candidate_POS"]
                ),

                "CAMeL_Attempted": False,
                "CAMeL_Normalized_Term": None,
                "CAMeL_Analysis_Found": False,
                "CAMeL_Analysis_Count": 0,
                "CAMeL_Top_Analysis_Count": 0,
                "CAMeL_Top_Roots": [],
                "CAMeL_Top_Root_Validations": [],
                "CAMeL_Root_Ambiguous": False,
                "CAMeL_Status": "NOT_REQUIRED",
                "CAMeL_Root_Raw": None,
                "CAMeL_Root_Normalized": None,
                "CAMeL_Root_Valid": False,
                "CAMeL_Root_Invalid_Reason": None,
                "CAMeL_Selected_Root": None,
                "CAMeL_Selected_Lemma": None,
                "CAMeL_Selected_Stem": None,
                "CAMeL_Selected_POS": None,
                "CAMeL_Analysis_Score": None,
                "CAMeL_Selected_Analysis": None

            })

            continue

        # ----------------------------------------------------
        # CAMeL Fallback
        # ----------------------------------------------------

        key = (
            question_id,
            normalized_term
        )

        camel = camel_lookup.get(key)

        if (
            camel is not None
            and bool(camel["CAMeL_Root_Resolved"])
            and bool(camel["CAMeL_Root_Valid"])
            and pd.notna(camel["Camel_Root"])
        ):

            root_ar = camel["Camel_Root"]
            lemma_ar = camel["Lemma"]
            pos = camel["POS"]
            matched = True
            root_resolved = True
            root_source = "CAMeL"
            resolution_status = "RESOLVED_CAMEL"

        else:

            root_ar = None
            lemma_ar = None
            pos = None
            matched = False
            root_resolved = False
            root_source = "UNRESOLVED"
            resolution_status = "UNRESOLVED"

        merged_matches.append({

            "Query_Term": match["Query_Term"],
            "Original_Term": match["Original_Term"],
            "Normalized_Term": normalized_term,

            "Matched": matched,
            "Root_Resolved": root_resolved,
            "Resolution_Status": resolution_status,

            "Root_Source": root_source,
            "Root_AR": root_ar,
            "Lemma_AR": lemma_ar,
            "POS": pos,

            "QAC_Lexical_Matched": match["Lexical_Matched"],
            "QAC_Lookup_Type": match["Lookup_Type"],
            "QAC_Status": match["QAC_Status"],
            "QAC_Candidate_Count": match["QAC_Candidate_Count"],
            "QAC_Candidate_Words": list_or_empty(
                match["QAC_Candidate_Words"]
            ),
            "QAC_Candidate_Lemmas": list_or_empty(
                match["QAC_Candidate_Lemmas"]
            ),
            "QAC_Candidate_Roots": list_or_empty(
                match["QAC_Candidate_Roots"]
            ),
            "QAC_Candidate_POS": list_or_empty(
                match["QAC_Candidate_POS"]
            ),

            "CAMeL_Attempted": camel is not None,
            "CAMeL_Normalized_Term": (
                camel["CAMeL_Normalized_Term"]
                if camel is not None
                else None
            ),
            "CAMeL_Analysis_Found": (
                bool(camel["CAMeL_Analysis_Found"])
                if camel is not None
                else False
            ),
            "CAMeL_Analysis_Count": (
                int(camel["CAMeL_Analysis_Count"])
                if camel is not None
                else 0
            ),
            "CAMeL_Top_Analysis_Count": (
                int(camel["CAMeL_Top_Analysis_Count"])
                if camel is not None
                else 0
            ),
            "CAMeL_Top_Roots": (
                camel["CAMeL_Top_Roots"]
                if camel is not None
                and isinstance(camel["CAMeL_Top_Roots"], list)
                else []
            ),
            "CAMeL_Top_Root_Validations": (
                camel["CAMeL_Top_Root_Validations"]
                if camel is not None
                and isinstance(
                    camel["CAMeL_Top_Root_Validations"],
                    list
                )
                else []
            ),
            "CAMeL_Root_Ambiguous": (
                bool(camel["CAMeL_Root_Ambiguous"])
                if camel is not None
                else False
            ),
            "CAMeL_Status": (
                camel["CAMeL_Status"]
                if camel is not None
                else "NO_FALLBACK_RECORD"
            ),
            "CAMeL_Root_Raw": (
                camel["CAMeL_Root_Raw"]
                if camel is not None
                and pd.notna(camel["CAMeL_Root_Raw"])
                else None
            ),
            "CAMeL_Root_Normalized": (
                camel["CAMeL_Root_Normalized"]
                if camel is not None
                and pd.notna(camel["CAMeL_Root_Normalized"])
                else None
            ),
            "CAMeL_Root_Valid": (
                bool(camel["CAMeL_Root_Valid"])
                if camel is not None
                else False
            ),
            "CAMeL_Root_Invalid_Reason": (
                camel["CAMeL_Root_Invalid_Reason"]
                if camel is not None
                and pd.notna(
                    camel["CAMeL_Root_Invalid_Reason"]
                )
                else None
            ),
            "CAMeL_Selected_Root": (
                camel["Camel_Root"]
                if camel is not None
                and pd.notna(camel["Camel_Root"])
                else None
            ),
            "CAMeL_Selected_Lemma": (
                camel["Lemma"]
                if camel is not None
                and pd.notna(camel["Lemma"])
                else None
            ),
            "CAMeL_Selected_Stem": (
                camel["Stem"]
                if camel is not None
                and pd.notna(camel["Stem"])
                else None
            ),
            "CAMeL_Selected_POS": (
                camel["POS"]
                if camel is not None
                and pd.notna(camel["POS"])
                else None
            ),
            "CAMeL_Analysis_Score": (
                int(camel["Analysis_Score"])
                if camel is not None
                and pd.notna(camel["Analysis_Score"])
                else None
            ),
            "CAMeL_Selected_Analysis": (
                camel["CAMeL_Selected_Analysis"]
                if camel is not None
                and isinstance(
                    camel["CAMeL_Selected_Analysis"],
                    dict
                )
                else None
            )

        })

    merged_records.append({

        "Dataset": row["Dataset"],
        "Question_ID": question_id,
        "Question": row["Question"],
        "Query_Terms": row["Query_Terms"],
        "Query_Term_Provenance": row["Query_Term_Provenance"],
        "Lexical_Matches": merged_matches

    })

hybrid_results_df = pd.DataFrame(merged_records)

print("=" * 70)
print("HYBRID ROOT RETRIEVAL COMPLETED")
print("=" * 70)

print(f"Questions Processed : {len(hybrid_results_df):,}")

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

flat_matches = [
    match
    for matches in hybrid_results_df["Lexical_Matches"]
    for match in matches
]

total_terms = len(flat_matches)
resolved_terms = sum(
    bool(match["Root_Resolved"])
    for match in flat_matches
)
qac_terms = sum(
    match["Root_Source"] == "QAC"
    for match in flat_matches
)
camel_terms = sum(
    match["Root_Source"] == "CAMeL"
    for match in flat_matches
)
unresolved_terms = sum(
    match["Root_Source"] == "UNRESOLVED"
    for match in flat_matches
)

print(f"Total Query Terms : {total_terms:,}")
print(f"Resolved Roots    : {resolved_terms:,}")
print(f"QAC Roots         : {qac_terms:,}")
print(f"CAMeL Roots       : {camel_terms:,}")
print(f"Unresolved        : {unresolved_terms:,}")

if total_terms:
    print(
        f"\nRoot Coverage     : "
        f"{100 * resolved_terms / total_terms:.2f}%"
    )

# Strict lineage / status validations
invalid_resolved = [
    match
    for match in flat_matches
    if match["Root_Resolved"] and not match["Root_AR"]
]

invalid_sources = [
    match
    for match in flat_matches
    if (
        match["Root_Resolved"]
        and match["Root_Source"] not in {"QAC", "CAMeL"}
    )
]

invalid_unresolved = [
    match
    for match in flat_matches
    if (
        not match["Root_Resolved"]
        and match["Root_Source"] != "UNRESOLVED"
    )
]

if invalid_resolved or invalid_sources or invalid_unresolved:
    raise ValueError(
        "Hybrid root provenance validation failed. "
        f"resolved-without-root={len(invalid_resolved)}, "
        f"invalid-resolved-source={len(invalid_sources)}, "
        f"invalid-unresolved-source={len(invalid_unresolved)}"
    )

print("\nHybrid root provenance validation passed.")

print("\nSample Record")
display(hybrid_results_df.head(1))

# %% [notebook cell 28]
# ============================================================
# Analyze Remaining Unresolved Terms
# ============================================================

unresolved_records = []

for _, row in hybrid_results_df.iterrows():

    for match in row["Lexical_Matches"]:

        if not match["Root_Resolved"]:

            unresolved_records.append({

                "Dataset": row["Dataset"],
                "Question_ID": row["Question_ID"],
                "Question": row["Question"],

                "Original_Term": match["Original_Term"],
                "Normalized_Term": match["Normalized_Term"],

                "QAC_Status": match["QAC_Status"],
                "QAC_Candidate_Roots": match["QAC_Candidate_Roots"],

                "CAMeL_Status": match["CAMeL_Status"],
                "CAMeL_Top_Roots": match["CAMeL_Top_Roots"],
                "CAMeL_Top_Root_Validations": (
                    match["CAMeL_Top_Root_Validations"]
                ),
                "CAMeL_Root_Raw": match["CAMeL_Root_Raw"],
                "CAMeL_Root_Normalized": (
                    match["CAMeL_Root_Normalized"]
                ),
                "CAMeL_Root_Valid": match["CAMeL_Root_Valid"],
                "CAMeL_Root_Invalid_Reason": (
                    match["CAMeL_Root_Invalid_Reason"]
                ),
                "CAMeL_Selected_Lemma": (
                    match["CAMeL_Selected_Lemma"]
                ),
                "CAMeL_Selected_POS": (
                    match["CAMeL_Selected_POS"]
                )

            })

remaining_unresolved = pd.DataFrame(unresolved_records)

print("=" * 70)
print("REMAINING UNRESOLVED TERMS")
print("=" * 70)

print(f"Number of Unresolved Terms : {len(remaining_unresolved):,}")

if len(remaining_unresolved) == 0:

    print("\nAll query terms have an acceptable root.")

else:

    print(
        f"Unique Unresolved Terms    : "
        f"{remaining_unresolved['Normalized_Term'].nunique():,}"
    )

    print("\nFrequency of Unresolved Terms")

    frequency = (
        remaining_unresolved["Normalized_Term"]
        .value_counts()
        .reset_index()
    )

    frequency.columns = ["Normalized_Term", "Frequency"]

    display(frequency)

    print("\nDetailed Records")
    display(
        remaining_unresolved.sort_values(
            ["Normalized_Term", "Question_ID"]
        )
    )

# %% [notebook cell 30]
# ============================================================
# Export Hybrid Root Retrieval Results
# ============================================================

export_df = hybrid_results_df.copy()

# Convert list/dict columns to standards-compliant JSON strings.
export_df["Query_Terms"] = (
    export_df["Query_Terms"]
    .apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False
        )
    )
)

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

export_df["Lexical_Matches"] = (
    export_df["Lexical_Matches"]
    .apply(
        lambda value: json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False
        )
    )
)

OUTPUT_FILE = (
    CORRECTED_OUTPUT_DIR /
    "QuranQA_Hybrid_Root_Retrieval_Corrected.csv"
)

export_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# Export Summary
# ------------------------------------------------------------

flat_matches = [
    match
    for matches in hybrid_results_df["Lexical_Matches"]
    for match in matches
]

total_questions = len(hybrid_results_df)
total_terms = len(flat_matches)
resolved_terms = sum(
    bool(match["Root_Resolved"])
    for match in flat_matches
)
qac_terms = sum(
    match["Root_Source"] == "QAC"
    for match in flat_matches
)
camel_terms = sum(
    match["Root_Source"] == "CAMeL"
    for match in flat_matches
)
remaining_terms = sum(
    match["Root_Source"] == "UNRESOLVED"
    for match in flat_matches
)

coverage = (
    resolved_terms / total_terms * 100
    if total_terms
    else 0.0
)

print("=" * 70)
print("HYBRID ROOT RETRIEVAL RESULTS EXPORTED")
print("=" * 70)

print(f"Output File          : {OUTPUT_FILE.name}")
print(f"Location             : {OUTPUT_FILE.parent}")

print("\nDataset Summary")

print(f"Questions            : {total_questions:,}")
print(f"Total Query Terms    : {total_terms:,}")
print(f"Resolved Roots       : {resolved_terms:,}")
print(f"QAC Roots            : {qac_terms:,}")
print(f"CAMeL Roots          : {camel_terms:,}")
print(f"Remaining Unresolved : {remaining_terms:,}")
print(f"Root Coverage        : {coverage:.2f}%")

print(
    "\nPreservation rule: the historical "
    "'QuranQA_Hybrid_Root_Retrieval.csv' file is not overwritten."
)

print("\nExport completed successfully.")
