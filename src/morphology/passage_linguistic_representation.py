"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
from pathlib import Path
from importlib.metadata import version
import sys
import json
import re
import sqlite3

import numpy as np
import pandas as pd

from IPython.display import display


def find_workspace_root(start_path):
    """Locate the authoritative Writing Workspace without a personal absolute path."""
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


NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = find_workspace_root(NOTEBOOK_DIR)

ROOT_EXTRACTION_DIR = (
    PROJECT_ROOT /
    "02_Methodology_Evidence" /
    "QAC and Root Extraction"
)

MAQAYIS_DIR = (
    PROJECT_ROOT /
    "02_Methodology_Evidence" /
    "Maqāyīs Lookup"
)

for module_dir in [ROOT_EXTRACTION_DIR, MAQAYIS_DIR]:
    module_path = str(module_dir)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import arabic_preprocessing as ap
import hybrid_root_extractor as hre
import semantic_lookup as sl

print("=" * 70)
print("PASSAGE-SIDE ENVIRONMENT")
print("=" * 70)
print(f"Python             : {sys.version.split()[0]}")
print(f"CAMeL Tools        : {version('camel-tools')}")
print(f"pandas             : {pd.__version__}")
print(f"NumPy              : {np.__version__}")
print(f"Root helper        : {Path(hre.__file__).resolve()}")
print(f"Preprocess helper  : {Path(ap.__file__).resolve()}")
print(f"Semantic helper    : {Path(sl.__file__).resolve()}")

# %% [notebook cell 3]
# ============================================================
# Define Current Workspace Paths
# ============================================================

DATA_RESOURCES_DIR = (
    PROJECT_ROOT /
    "02_Methodology_Evidence" /
    "Data and Resources"
)

QURANQA_DIR = DATA_RESOURCES_DIR / "QuranQA"

PASSAGE_PREPARATION_DIR = (
    DATA_RESOURCES_DIR /
    "Corrected_Outputs" /
    "QPC_Passage_Side_Validation"
)

LEXICON_DIR = MAQAYIS_DIR / "Lexicon Database and Metadata"
MAQAYIS_DB_FILE = LEXICON_DIR / "db.sqlite"

PASSAGE_ENRICHMENT_DIR = (
    ROOT_EXTRACTION_DIR /
    "Corrected_Outputs" /
    "QPC_Passage_Side_Validation"
)

PASSAGE_ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)

PASSAGE_COLLECTION_FILE = (
    PASSAGE_PREPARATION_DIR /
    "QuranQA_Processed_Passage_Collection.csv"
)

OUTPUT_FILE = (
    PASSAGE_ENRICHMENT_DIR /
    "QuranQA_Linguistically_Enriched_Passages.csv"
)

TOKEN_PROVENANCE_FILE = (
    PASSAGE_ENRICHMENT_DIR /
    "QuranQA_Passage_Token_Provenance.csv"
)

PASSAGE_REPRESENTATION_FILE = (
    PASSAGE_ENRICHMENT_DIR /
    "QuranQA_Enriched_Passage_Representation.csv"
)

print("=" * 70)
print("PROJECT PATHS")
print("=" * 70)
print(f"Project Root                 : {PROJECT_ROOT}")
print(f"Processed Passage Collection : {PASSAGE_COLLECTION_FILE}")
print(f"Maqayis Database             : {MAQAYIS_DB_FILE}")
print(f"Corrected Output Directory   : {PASSAGE_ENRICHMENT_DIR}")

required_paths = [PASSAGE_COLLECTION_FILE, MAQAYIS_DB_FILE]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    raise FileNotFoundError(f"Missing required resources: {missing_paths}")

# %% [notebook cell 5]
# ============================================================
# Load Processed Quran Passage Collection
# ============================================================

passages_df = pd.read_csv(
    PASSAGE_COLLECTION_FILE,
    encoding="utf-8-sig"
)

print("=" * 70)
print("PROCESSED PASSAGE COLLECTION LOADED")
print("=" * 70)

print(f"Total Passages : {len(passages_df):,}")
print(f"Total Columns  : {passages_df.shape[1]}")

print("\nColumns")

print(list(passages_df.columns))

print("\nSample Passages")

display(
    passages_df[
        [
            "Passage_ID",
            "Passage_Text"
        ]
    ].head(5)
)

# %% [notebook cell 7]
# ============================================================
# Preprocess the Quran Passage Collection with Token Provenance
# ============================================================


def original_surface_tokens(text):
    """Tokenize after punctuation removal but before orthographic normalization."""
    return ap.tokenize(ap.remove_punctuation(text))


passages_df = passages_df.copy()

passages_df["Original_Tokens"] = (
    passages_df["Passage_Text"]
    .apply(original_surface_tokens)
)

passages_df["Normalized_Text"] = (
    passages_df["Passage_Text"]
    .apply(ap.normalize_arabic)
    .apply(ap.remove_punctuation)
)

passages_df["Tokens"] = (
    passages_df["Normalized_Text"]
    .apply(ap.tokenize)
)

token_records = []

for _, row in passages_df.iterrows():
    original_tokens = row["Original_Tokens"]
    normalized_tokens = row["Tokens"]

    if len(original_tokens) != len(normalized_tokens):
        raise ValueError(
            f"Token provenance length mismatch for passage {row['Passage_ID']}: "
            f"{len(original_tokens)} original vs {len(normalized_tokens)} normalized"
        )

    for position, (original_token, normalized_token) in enumerate(
        zip(original_tokens, normalized_tokens),
        start=1
    ):
        token_records.append({
            "Passage_ID": row["Passage_ID"],
            "Token_Position": position,
            "Original_Token": original_token,
            "Normalized_Token": normalized_token,
        })

token_provenance_df = pd.DataFrame(token_records)

total_tokens = len(token_provenance_df)

print("=" * 70)
print("PASSAGE COLLECTION PREPROCESSED")
print("=" * 70)
print(f"Total Passages          : {len(passages_df):,}")
print(f"Total Token Occurrences : {total_tokens:,}")
print(f"Token Mapping Mismatches: 0")

display(
    token_provenance_df.head(10)
)

# %% [notebook cell 9]
# ============================================================
# Construct the Deterministic Quran Lexical Vocabulary
# ============================================================

from collections import Counter

all_tokens = token_provenance_df["Normalized_Token"].tolist()
token_counter = Counter(all_tokens)

original_form_lookup = {}
for normalized_token, group in token_provenance_df.groupby(
    "Normalized_Token",
    sort=False
):
    original_form_lookup[normalized_token] = list(
        dict.fromkeys(group["Original_Token"].tolist())
    )

vocabulary_df = pd.DataFrame(
    token_counter.items(),
    columns=["Word", "Frequency"]
)

vocabulary_df = (
    vocabulary_df
    .sort_values(
        by=["Frequency", "Word"],
        ascending=[False, True],
        kind="mergesort"
    )
    .reset_index(drop=True)
)

vocabulary_df["Normalized_Token"] = vocabulary_df["Word"]
vocabulary_df["Original_Token_Forms"] = vocabulary_df["Word"].apply(
    lambda token: json.dumps(
        original_form_lookup[token],
        ensure_ascii=False,
        allow_nan=False
    )
)

print("=" * 70)
print("QURAN LEXICAL VOCABULARY CREATED")
print("=" * 70)
print(f"Total Token Occurrences : {len(all_tokens):,}")
print(f"Unique Lexical Terms    : {len(vocabulary_df):,}")

display(vocabulary_df.head(20))

# %% [notebook cell 11]
# ============================================================
# Prepare Vocabulary Provenance Columns
# ============================================================

lexical_vocabulary_df = vocabulary_df.copy()

lexical_vocabulary_df.insert(
    0,
    "Vocabulary_ID",
    range(1, len(lexical_vocabulary_df) + 1)
)

defaults = {
    "CAMeL_Input_Token": None,
    "Analysis_Source": "CAMeL",
    "Analysis_Found": False,
    "Analysis_Count": 0,
    "Analysis_Score": None,
    "Top_Analysis_Count": 0,
    "Top_Roots": "[]",
    "Top_Root_Validations": "[]",
    "Selected_Analysis_Evidence": "null",
    "Root_Ambiguous": False,
    "Resolution_Status": None,
    "Root_Unresolved_Reason": None,
    "CAMeL_Root_Raw": None,
    "CAMeL_Root_Normalized": None,
    "CAMeL_Root_Valid": False,
    "CAMeL_Root_Invalid_Reason": None,
    "Root_AR": None,
    "Hybrid_Root": None,
    "Lemma_AR": None,
    "Stem_AR": None,
    "POS": None,
    "Root_Source": None,
    "Root_Resolved": False,
    "Root_Found": False,
    "Semantic_Meaning": None,
}

for column, default in defaults.items():
    lexical_vocabulary_df[column] = default

print("=" * 70)
print("LEXICAL VOCABULARY PREPARED")
print("=" * 70)
print(f"Vocabulary Entries : {len(lexical_vocabulary_df):,}")
print(f"Columns            : {len(lexical_vocabulary_df.columns):,}")

display(lexical_vocabulary_df.head(10))

# %% [notebook cell 13]
# ============================================================
# Analyze the Quran Lexical Vocabulary with CAMeL Tools
# ============================================================


def strict_json(value):
    """Serialize provenance with standard JSON null values, never NaN."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        default=str
    )


selected_evidence_fields = [
    "root",
    "lex",
    "stem",
    "pos",
    "bw",
    "gloss",
    "source",
    "pattern",
]

for idx, row in lexical_vocabulary_df.iterrows():
    token = row["Word"]
    camel_input = hre.normalize_for_camel(token)
    analyses = hre.CAMEL_ANALYZER.analyze(camel_input)
    selection = hre.select_best_analysis_with_provenance(token, analyses)

    analysis = selection["Selected_Analysis"]
    root_validation = selection["Selected_Root_Validation"]

    if analysis is None:
        selected_evidence = None
    else:
        selected_evidence = {
            field: analysis.get(field)
            for field in selected_evidence_fields
            if analysis.get(field) is not None
        }

    if root_validation is None:
        raw_root = None
        normalized_root = None
        root_valid = False
        invalid_reason = None
    else:
        raw_root = root_validation["Raw"]
        normalized_root = root_validation["Normalized"]
        root_valid = bool(root_validation["Valid"])
        invalid_reason = root_validation["Invalid_Reason"]

    root_resolved = bool(
        selection["Root_Resolved"]
        and root_valid
        and normalized_root
    )

    if root_resolved:
        unresolved_reason = None
    elif selection["Status"] == "NO_ANALYSIS":
        unresolved_reason = "no_analysis"
    elif selection["Status"] == "AMBIGUOUS_TOP_ROOT":
        unresolved_reason = "ambiguous_top_root"
    else:
        unresolved_reason = invalid_reason or "unresolved_root"

    lexical_vocabulary_df.at[idx, "CAMeL_Input_Token"] = camel_input
    lexical_vocabulary_df.at[idx, "Analysis_Found"] = bool(selection["Analysis_Found"])
    lexical_vocabulary_df.at[idx, "Analysis_Count"] = int(selection["Analysis_Count"])
    lexical_vocabulary_df.at[idx, "Analysis_Score"] = selection["Top_Score"]
    lexical_vocabulary_df.at[idx, "Top_Analysis_Count"] = int(selection["Top_Analysis_Count"])
    lexical_vocabulary_df.at[idx, "Top_Roots"] = strict_json(selection["Top_Roots"])
    lexical_vocabulary_df.at[idx, "Top_Root_Validations"] = strict_json(selection["Top_Root_Validations"])
    lexical_vocabulary_df.at[idx, "Selected_Analysis_Evidence"] = strict_json(selected_evidence)
    lexical_vocabulary_df.at[idx, "Root_Ambiguous"] = bool(selection["Root_Ambiguous"])
    lexical_vocabulary_df.at[idx, "Resolution_Status"] = selection["Status"]
    lexical_vocabulary_df.at[idx, "Root_Unresolved_Reason"] = unresolved_reason
    lexical_vocabulary_df.at[idx, "CAMeL_Root_Raw"] = raw_root
    lexical_vocabulary_df.at[idx, "CAMeL_Root_Normalized"] = normalized_root
    lexical_vocabulary_df.at[idx, "CAMeL_Root_Valid"] = root_valid
    lexical_vocabulary_df.at[idx, "CAMeL_Root_Invalid_Reason"] = invalid_reason
    lexical_vocabulary_df.at[idx, "Root_AR"] = normalized_root if root_resolved else None
    lexical_vocabulary_df.at[idx, "Hybrid_Root"] = normalized_root if root_resolved else None
    lexical_vocabulary_df.at[idx, "Lemma_AR"] = analysis.get("lex") if analysis else None
    lexical_vocabulary_df.at[idx, "Stem_AR"] = analysis.get("stem") if analysis else None
    lexical_vocabulary_df.at[idx, "POS"] = analysis.get("pos") if analysis else None
    lexical_vocabulary_df.at[idx, "Root_Source"] = "CAMeL" if root_resolved else None
    lexical_vocabulary_df.at[idx, "Root_Resolved"] = root_resolved
    lexical_vocabulary_df.at[idx, "Root_Found"] = root_resolved

vocabulary_id_lookup = dict(zip(
    lexical_vocabulary_df["Word"],
    lexical_vocabulary_df["Vocabulary_ID"]
))

token_provenance_df["Vocabulary_ID"] = (
    token_provenance_df["Normalized_Token"]
    .map(vocabulary_id_lookup)
)

if token_provenance_df["Vocabulary_ID"].isna().any():
    raise ValueError("At least one passage token has no vocabulary record")

token_provenance_df["Vocabulary_ID"] = (
    token_provenance_df["Vocabulary_ID"].astype(int)
)

print("=" * 70)
print("CAMEL VOCABULARY ANALYSIS COMPLETED")
print("=" * 70)
print(f"Vocabulary Entries  : {len(lexical_vocabulary_df):,}")
print(f"Analyses Examined   : {lexical_vocabulary_df['Analysis_Count'].sum():,}")
print(f"Valid Roots         : {lexical_vocabulary_df['Root_Resolved'].sum():,}")
print(f"Unresolved Terms    : {(~lexical_vocabulary_df['Root_Resolved']).sum():,}")

# %% [notebook cell 15]
# ============================================================
# Validate Strict CAMeL Root Decisions and Report Diagnostics
# ============================================================

resolved_mask = lexical_vocabulary_df["Root_Resolved"].astype(bool)
root_nonblank = (
    lexical_vocabulary_df["Root_AR"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)
source_present = (
    lexical_vocabulary_df["Root_Source"]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)

if (resolved_mask & ~root_nonblank).any():
    raise AssertionError("Resolved vocabulary record has a blank root")
if (resolved_mask & ~source_present).any():
    raise AssertionError("Resolved vocabulary record has no root source")
if (resolved_mask != lexical_vocabulary_df["Root_Found"].astype(bool)).any():
    raise AssertionError("Root_Found compatibility field disagrees with Root_Resolved")
if (resolved_mask & ~lexical_vocabulary_df["CAMeL_Root_Valid"].astype(bool)).any():
    raise AssertionError("Resolved vocabulary record failed strict CAMeL validation")

invalid_reason_counts = (
    lexical_vocabulary_df["CAMeL_Root_Invalid_Reason"]
    .dropna()
    .value_counts()
    .to_dict()
)

status_counts = (
    lexical_vocabulary_df["Resolution_Status"]
    .value_counts(dropna=False)
    .to_dict()
)

raw_roots = lexical_vocabulary_df["CAMeL_Root_Raw"].fillna("").astype(str)
normalized_roots = lexical_vocabulary_df["Root_AR"].fillna("").astype(str)

diacritic_pattern = re.compile(
    r"[ؐ-ًؚ-ٰٟۖ-ۭ]"
)

diacritic_retained_mask = (
    resolved_mask &
    raw_roots.str.contains(diacritic_pattern, regex=True)
)

one_character_mask = resolved_mask & normalized_roots.str.len().eq(1)
two_character_mask = resolved_mask & normalized_roots.str.len().eq(2)

print("=" * 70)
print("STRICT PASSAGE-SIDE ROOT VALIDATION")
print("=" * 70)
print(f"Valid-root vocabulary terms       : {resolved_mask.sum():,}")
print(f"Unresolved vocabulary terms       : {(~resolved_mask).sum():,}")
print(f"Sentinel-root rejections          : {invalid_reason_counts.get('sentinel_root', 0):,}")
print(f"Incomplete-# rejections           : {invalid_reason_counts.get('incomplete_placeholder_root', 0):,}")
print(f"Non-Arabic-root rejections        : {invalid_reason_counts.get('non_arabic_root', 0):,}")
print(f"Missing-root rejections           : {invalid_reason_counts.get('missing_root', 0):,}")
print(f"Diacritic-bearing roots retained  : {diacritic_retained_mask.sum():,}")
print(f"Valid one-character roots         : {one_character_mask.sum():,}")
print(f"Valid two-character roots         : {two_character_mask.sum():,}")

print("\nResolution Status")
display(pd.DataFrame(
    sorted(status_counts.items()),
    columns=["Resolution_Status", "Count"]
))

print("\nOne-character valid-root examples")
display(lexical_vocabulary_df.loc[
    one_character_mask,
    ["Word", "CAMeL_Root_Raw", "Root_AR", "Lemma_AR", "POS"]
].head(10))

print("\nTwo-character valid-root examples")
display(lexical_vocabulary_df.loc[
    two_character_mask,
    ["Word", "CAMeL_Root_Raw", "Root_AR", "Lemma_AR", "POS"]
].head(10))

# %% [notebook cell 17]
# ============================================================
# Load Maqāyīs al-Lughah Semantic Lookup
# ============================================================

connection = sqlite3.connect(MAQAYIS_DB_FILE)

maqayis_df = pd.read_sql_query(
    """
    SELECT word, meanings
    FROM maqayeesul_luga
    """,
    connection
)

connection.close()

maqayis_lookup, duplicate_entries = sl.build_maqayis_lookup(
    maqayis_df
)

print("=" * 70)
print("MAQAYIS SEMANTIC LOOKUP CREATED")
print("=" * 70)

print(f"Dictionary Entries : {len(maqayis_lookup):,}")
print(f"Duplicate Entries  : {duplicate_entries:,}")

print("\nSample Dictionary Entries")

sample_df = (
    pd.DataFrame(
        list(maqayis_lookup.items()),
        columns=["Root_AR", "Semantic_Meaning"]
    )
)

display(sample_df.head(10))

# %% [notebook cell 19]
# ============================================================
# Apply Semantic Enrichment
# ============================================================

semantic_matches = 0

for idx, row in lexical_vocabulary_df.iterrows():

    root = row["Hybrid_Root"]

    if pd.isna(root):
        continue

    meaning = maqayis_lookup.get(root)

    if meaning is not None:

        lexical_vocabulary_df.at[idx, "Semantic_Meaning"] = meaning
        semantic_matches += 1

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

coverage = semantic_matches / len(lexical_vocabulary_df) * 100

print("=" * 70)
print("SEMANTIC ENRICHMENT COMPLETED")
print("=" * 70)

print(f"Vocabulary Entries : {len(lexical_vocabulary_df):,}")
print(f"Semantic Matches   : {semantic_matches:,}")
print(f"Coverage           : {coverage:.2f}%")

print("\nSample Enriched Vocabulary")

display(
    lexical_vocabulary_df[
        [
            "Word",
            "Hybrid_Root",
            "Semantic_Meaning"
        ]
    ]
    .dropna(subset=["Semantic_Meaning"])
    .head(20)
)

# %% [notebook cell 20]
# ============================================================
# Export Corrected Vocabulary and Token Provenance
# ============================================================

export_df = lexical_vocabulary_df.copy()

export_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

token_provenance_df.to_csv(
    TOKEN_PROVENANCE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 70)
print("CORRECTED VOCABULARY AND TOKEN PROVENANCE EXPORTED")
print("=" * 70)
print(f"Vocabulary Entries : {len(export_df):,}")
print(f"Token Records      : {len(token_provenance_df):,}")
print(f"Vocabulary File    : {OUTPUT_FILE}")
print(f"Token Trace File   : {TOKEN_PROVENANCE_FILE}")

# %% [notebook cell 23]
# ============================================================
# Construct Passages Exclusively from the Validated Vocabulary
# ============================================================

vocabulary_lookup = (
    lexical_vocabulary_df
    .set_index("Vocabulary_ID")
    .to_dict("index")
)

token_groups = {
    passage_id: group.sort_values("Token_Position", kind="mergesort")
    for passage_id, group in token_provenance_df.groupby("Passage_ID", sort=False)
}

passage_records = []

for _, row in passages_df.iterrows():
    passage_id = row["Passage_ID"]
    group = token_groups.get(passage_id)

    if group is None:
        raise ValueError(f"No token provenance found for passage {passage_id}")

    original_tokens = group["Original_Token"].tolist()
    normalized_tokens = group["Normalized_Token"].tolist()
    vocabulary_ids = group["Vocabulary_ID"].astype(int).tolist()

    roots = []
    semantic_meanings = []
    resolved_token_count = 0

    for vocabulary_id in vocabulary_ids:
        vocabulary_record = vocabulary_lookup[vocabulary_id]

        if not bool(vocabulary_record["Root_Resolved"]):
            continue

        root = vocabulary_record["Root_AR"]
        if root is None or str(root).strip() == "":
            raise AssertionError(
                f"Resolved vocabulary record {vocabulary_id} has no root"
            )

        resolved_token_count += 1

        if root not in roots:
            roots.append(root)

        meaning = vocabulary_record["Semantic_Meaning"]
        if meaning is not None and not pd.isna(meaning):
            if meaning not in semantic_meanings:
                semantic_meanings.append(meaning)

    passage_records.append({
        "Passage_ID": passage_id,
        "Surah": row["Surah"],
        "Start_Ayah": row["Start_Ayah"],
        "End_Ayah": row["End_Ayah"],
        "Passage_Text": row["Passage_Text"],
        "Original_Tokens": strict_json(original_tokens),
        "Tokens": strict_json(normalized_tokens),
        "Token_Vocabulary_IDs": strict_json(vocabulary_ids),
        "Roots": strict_json(roots),
        "Semantic_Meanings": strict_json(semantic_meanings),
        "Token_Count": len(normalized_tokens),
        "Resolved_Token_Count": resolved_token_count,
        "Unresolved_Token_Count": len(normalized_tokens) - resolved_token_count,
        "Root_Count": len(roots),
        "Semantic_Count": len(semantic_meanings),
    })

passage_representation_df = pd.DataFrame(passage_records)

print("=" * 70)
print("AUTHORITATIVE PASSAGE REPRESENTATION CREATED")
print("=" * 70)
print(f"Total Passages : {len(passage_representation_df):,}")
print(f"Passages with Valid Roots : {(passage_representation_df['Root_Count'] > 0).sum():,}")
print(f"Passages with No Valid Roots: {(passage_representation_df['Root_Count'] == 0).sum():,}")

display(passage_representation_df.head())

# %% [notebook cell 24]
# ============================================================
# Export Corrected Passage Representation
# ============================================================

passage_representation_df.to_csv(
    PASSAGE_REPRESENTATION_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 70)
print("CORRECTED PASSAGE REPRESENTATION EXPORTED")
print("=" * 70)
print(f"Total Passages : {len(passage_representation_df):,}")
print(f"Output File    : {PASSAGE_REPRESENTATION_FILE}")

# %% [notebook cell 25]
# ============================================================
# Final Vocabulary-to-Passage Consistency Validation
# ============================================================

inconsistent_passages = []
passage_id_order = passages_df["Passage_ID"].astype(str).tolist()
representation_id_order = (
    passage_representation_df["Passage_ID"].astype(str).tolist()
)

if passage_id_order != representation_id_order:
    raise AssertionError("Passage ID order changed during Notebook 9A")

for _, passage in passage_representation_df.iterrows():
    vocabulary_ids = json.loads(passage["Token_Vocabulary_IDs"])
    actual_roots = json.loads(passage["Roots"])

    expected_roots = []
    for vocabulary_id in vocabulary_ids:
        vocabulary_record = vocabulary_lookup[int(vocabulary_id)]
        if not bool(vocabulary_record["Root_Resolved"]):
            continue
        root = vocabulary_record["Root_AR"]
        if root not in expected_roots:
            expected_roots.append(root)

    if actual_roots != expected_roots:
        inconsistent_passages.append(passage["Passage_ID"])

if inconsistent_passages:
    raise AssertionError(
        f"Vocabulary-to-passage inconsistencies: {inconsistent_passages[:10]}"
    )

print("=" * 70)
print("FINAL PASSAGE-SIDE VALIDATION")
print("=" * 70)
print(f"Passages                               : {len(passage_representation_df):,}")
print(f"Unique Passage IDs                    : {passage_representation_df['Passage_ID'].nunique():,}")
print(f"Vocabulary-to-passage inconsistencies : {len(inconsistent_passages):,}")
print(f"Duplicate Passage IDs                 : {passage_representation_df['Passage_ID'].duplicated().sum():,}")
print("No second CAMeL analysis path was used during passage construction.")
