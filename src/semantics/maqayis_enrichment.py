"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
from pathlib import Path
import sys


def discover_project_root(start: Path) -> Path:
    """Locate the workspace from its authority marker and expected layout."""
    for candidate in [start.resolve(), *start.resolve().parents]:
        marker = candidate / "00_Project_Control" / "README_Workspace_Authority.txt"
        point2 = (
            candidate
            / "02_Methodology_Evidence"
            / "QAC and Root Extraction"
            / "Corrected_Outputs"
            / "Point_2_Query_Root_Provenance"
            / "QuranQA_Hybrid_Root_Retrieval_Corrected.csv"
        )
        if marker.exists() and point2.exists():
            return candidate
    raise FileNotFoundError("Could not locate the thesis workspace from the current notebook directory")


NOTEBOOK_DIR = Path.cwd().resolve()
PROJECT_ROOT = discover_project_root(NOTEBOOK_DIR)
SEMANTIC_CODE_DIR = PROJECT_ROOT / "02_Methodology_Evidence" / "Maqāyīs Lookup"

if str(SEMANTIC_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_CODE_DIR))

import semantic_lookup as sl

print("=" * 70)
print("SEMANTIC LOOKUP MODULE")
print("=" * 70)
print(f"Project Root    : {PROJECT_ROOT}")
print(f"Semantic Helper : {Path(sl.__file__).resolve()}")

# %% [notebook cell 3]
# ============================================================
# Import Required Libraries
# ============================================================

from pathlib import Path
import sqlite3
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
print("✓ sqlite3")
print("✓ json")
print("✓ numpy")
print("✓ pandas")
print("✓ IPython.display")

# %% [notebook cell 4]
# ============================================================
# Define Current Project Paths
# ============================================================

METHODOLOGY_DIR = PROJECT_ROOT / "02_Methodology_Evidence"
ROOT_EXTRACTION_DIR = METHODOLOGY_DIR / "QAC and Root Extraction"
LEXICON_DIR = METHODOLOGY_DIR / "Maqāyīs Lookup" / "Lexicon Database and Metadata"
OUTPUT_DIR = (
    METHODOLOGY_DIR
    / "Maqāyīs Lookup"
    / "Corrected_Outputs"
    / "Notebook_7_Maqayis_Semantic_Enrichment"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HYBRID_ROOT_FILE = (
    ROOT_EXTRACTION_DIR
    / "Corrected_Outputs"
    / "Point_2_Query_Root_Provenance"
    / "QuranQA_Hybrid_Root_Retrieval_Corrected.csv"
)
MAQAYIS_DB_FILE = LEXICON_DIR / "db.sqlite"
OUTPUT_FILE = OUTPUT_DIR / "QuranQA_Semantic_Enrichment_Corrected.csv"

if not HYBRID_ROOT_FILE.exists():
    raise FileNotFoundError(HYBRID_ROOT_FILE)
if not MAQAYIS_DB_FILE.exists():
    raise FileNotFoundError(MAQAYIS_DB_FILE)

print("=" * 70)
print("PROJECT PATHS")
print("=" * 70)
print(f"Authoritative Point-2 Input : {HYBRID_ROOT_FILE}")
print(f"Maqāyīs Database           : {MAQAYIS_DB_FILE}")
print(f"Corrected Output           : {OUTPUT_FILE}")

# %% [notebook cell 5]
# ============================================================
# Load Hybrid Root Retrieval Results
# ============================================================

hybrid_df = pd.read_csv(HYBRID_ROOT_FILE)

print("=" * 70)
print("HYBRID ROOT RETRIEVAL RESULTS LOADED")
print("=" * 70)

print(f"Total Questions : {len(hybrid_df):,}")
print(f"Total Columns   : {len(hybrid_df.columns)}")

print("\nColumns")

print(list(hybrid_df.columns))

print("\nSample Record")

display(hybrid_df.head(1))

# %% [notebook cell 7]
# ============================================================
# Parse and Validate Authoritative Point-2 Structures
# ============================================================

NESTED_COLUMNS = ["Query_Terms", "Query_Term_Provenance", "Lexical_Matches"]

for column in NESTED_COLUMNS:
    hybrid_df[column] = hybrid_df[column].apply(
        lambda value: json.loads(value) if isinstance(value, str) else value
    )
    if not hybrid_df[column].map(lambda value: isinstance(value, list)).all():
        raise TypeError(f"{column} must contain one list per question")

term_records = []
for _, row in hybrid_df.iterrows():
    matches = row["Lexical_Matches"]
    provenance = row["Query_Term_Provenance"]
    if len(matches) != len(row["Query_Terms"]) or len(matches) != len(provenance):
        raise ValueError(f"Nested query-term lengths disagree for Question_ID={row['Question_ID']}")
    for position, (match, term_provenance) in enumerate(zip(matches, provenance), start=1):
        term_records.append({
            "Dataset": row["Dataset"],
            "Question_ID": row["Question_ID"],
            "Question": row["Question"],
            "Term_Order": position,
            "Token_Position": term_provenance.get("Token_Position"),
            "Original_Term": match.get("Original_Term", term_provenance.get("Original_Term")),
            "Normalized_Term": match.get("Normalized_Term", term_provenance.get("Normalized_Term")),
            "Root_Resolved": bool(match.get("Root_Resolved", False)),
            "Resolution_Status": match.get("Resolution_Status"),
            "Root_AR": match.get("Root_AR"),
            "Root_Source": match.get("Root_Source"),
        })

term_lineage_df = pd.DataFrame(term_records)
resolved_mask = term_lineage_df["Root_Resolved"]

print("=" * 70)
print("AUTHORITATIVE POINT-2 REPRESENTATION PARSED")
print("=" * 70)
print(f"Questions             : {len(hybrid_df):,}")
print(f"Total Query Terms     : {len(term_lineage_df):,}")
print(f"Strict Resolved Roots : {int(resolved_mask.sum()):,}")
print(f"Unresolved Terms      : {int((~resolved_mask).sum()):,}")
print("\nResolved Roots by Source")
display(term_lineage_df.loc[resolved_mask, "Root_Source"].value_counts().rename_axis("Root_Source").reset_index(name="Terms"))

# %% [notebook cell 9]
# ============================================================
# Extract the Authoritative Root Inventory
# ============================================================

root_records = []

for _, row in hybrid_df.iterrows():
    for position, (match, term_provenance) in enumerate(
        zip(row["Lexical_Matches"], row["Query_Term_Provenance"]), start=1
    ):
        resolved = bool(match.get("Root_Resolved", False))
        root = match.get("Root_AR")
        source = match.get("Root_Source")

        if not resolved:
            if root is not None and str(root).strip():
                raise ValueError("An unresolved Point-2 term contains a nonblank authoritative root")
            continue

        if root is None or not str(root).strip():
            raise ValueError("A resolved Point-2 term has a blank authoritative root")
        if source not in {"QAC", "CAMeL"}:
            raise ValueError(f"A resolved Point-2 term has invalid Root_Source={source!r}")

        root_records.append({
            "Dataset": row["Dataset"],
            "Question_ID": row["Question_ID"],
            "Question": row["Question"],
            "Term_Order": position,
            "Token_Position": term_provenance.get("Token_Position"),
            "Original_Term": match.get("Original_Term", term_provenance.get("Original_Term")),
            "Normalized_Term": match.get("Normalized_Term", term_provenance.get("Normalized_Term")),
            "Root_AR": str(root),
            "Root_Source": source,
            "Resolution_Status": match.get("Resolution_Status"),
            "Lemma_AR": match.get("Lemma_AR"),
            "POS": match.get("POS"),
        })

roots_df = pd.DataFrame(root_records)

unique_roots_df = (
    roots_df.sort_values(
        ["Root_AR", "Dataset", "Question_ID", "Term_Order"],
        kind="mergesort",
    )
    .drop_duplicates(subset=["Root_AR"], keep="first")
    .reset_index(drop=True)
)

print("=" * 70)
print("AUTHORITATIVE ROOT INVENTORY EXTRACTED")
print("=" * 70)
print(f"Resolved Root Occurrences : {len(roots_df):,}")
print(f"Unique Authoritative Roots: {len(unique_roots_df):,}")
print("\nRoot Occurrences by Source")
display(roots_df["Root_Source"].value_counts().rename_axis("Root_Source").reset_index(name="Terms"))
print("\nSample Authoritative Roots")
display(unique_roots_df[["Root_AR", "Root_Source", "Original_Term", "Normalized_Term", "Lemma_AR", "POS"]].head(20))

# %% [notebook cell 11]
# ============================================================
# Load Maqāyīs al-Lughah Lexicon Deterministically
# ============================================================

connection = sqlite3.connect(MAQAYIS_DB_FILE)
maqayis_df = pd.read_sql_query(
    '''
    SELECT word, meanings
    FROM maqayeesul_luga
    ORDER BY id
    ''',
    connection,
)
connection.close()

print("=" * 70)
print("MAQĀYĪS AL-LUGHAH LOADED")
print("=" * 70)
print(f"Total Lexicon Entries : {len(maqayis_df):,}")
print(f"Unique Trimmed Roots  : {maqayis_df['word'].astype(str).str.strip().nunique():,}")
print(f"Blank Roots           : {int(maqayis_df['word'].isna().sum() + maqayis_df['word'].astype(str).str.strip().eq('').sum()):,}")
print(f"Blank Meanings        : {int(maqayis_df['meanings'].isna().sum() + maqayis_df['meanings'].astype(str).str.strip().eq('').sum()):,}")
print("\nSample Entries")
display(maqayis_df.head(10))

# %% [notebook cell 13]
# ============================================================
# Build Maqāyīs Semantic Lookup Dictionary
# ============================================================
# ------------------------------------------------------------
# Build Lookup Dictionary
# ------------------------------------------------------------
maqayis_lookup, duplicate_entries = (
    sl.build_maqayis_lookup(maqayis_df)
)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("=" * 70)
print("SEMANTIC LOOKUP DICTIONARY CREATED")
print("=" * 70)

print(f"Dictionary Entries : {len(maqayis_lookup):,}")
print(f"Duplicate Entries  : {duplicate_entries:,}")

# ------------------------------------------------------------
# Sample Lookup Entries
# ------------------------------------------------------------

sample_roots = list(maqayis_lookup.keys())[:10]

sample_df = pd.DataFrame({

    "Root": sample_roots,

    "Semantic_Meaning": [

        maqayis_lookup[root]

        for root in sample_roots

    ]

})

print("\nSample Dictionary Entries")

display(sample_df)

# %% [notebook cell 15]
# ============================================================
# Build Semantic Root Table Using Exact Authoritative Roots
# ============================================================

semantic_roots_df = sl.lookup_semantic_meanings(unique_roots_df, maqayis_lookup)
semantic_roots_df["Maqayis_Lookup_Attempted"] = True
semantic_roots_df["Maqayis_Lookup_Root"] = semantic_roots_df["Root_AR"]

found_unique = int(semantic_roots_df["Semantic_Found"].sum())
missing_unique = len(semantic_roots_df) - found_unique
unique_coverage = 100 * found_unique / len(semantic_roots_df) if len(semantic_roots_df) else 0.0

root_occurrence_semantics_df = roots_df.merge(
    semantic_roots_df[["Root_AR", "Semantic_Found", "Semantic_Meaning"]],
    on="Root_AR",
    how="left",
    validate="many_to_one",
)

print("=" * 70)
print("SEMANTIC ROOT TABLE CREATED")
print("=" * 70)
print(f"Unique Roots            : {len(semantic_roots_df):,}")
print(f"Unique Semantic Matches : {found_unique:,}")
print(f"Unique Roots Without Match: {missing_unique:,}")
print(f"Unique-Root Coverage     : {unique_coverage:.2f}%")
print("\nSample Semantic Entries")
display(semantic_roots_df.head(20))

# %% [notebook cell 17]
# ============================================================
# Analyze Unmatched Semantic Roots
# ============================================================

missing_semantics_df = semantic_roots_df.loc[~semantic_roots_df["Semantic_Found"]].copy()
missing_occurrences_df = root_occurrence_semantics_df.loc[
    ~root_occurrence_semantics_df["Semantic_Found"]
].copy()

source_summary = (
    missing_occurrences_df["Root_Source"]
    .value_counts()
    .rename_axis("Root_Source")
    .reset_index(name="Unmatched_Terms")
)
pos_summary = (
    missing_occurrences_df["POS"]
    .value_counts(dropna=False)
    .rename_axis("POS")
    .reset_index(name="Frequency")
)

print("=" * 70)
print("UNMATCHED SEMANTIC ROOTS")
print("=" * 70)
print(f"Unique Roots Without Match : {len(missing_semantics_df):,}")
print(f"Term Occurrences Without Match: {len(missing_occurrences_df):,}")
print("\nUnmatched Term Occurrences by Root Source")
display(source_summary)
print("\nUnmatched Term Occurrences by Part of Speech")
display(pos_summary)
print("\nDetailed Unique Roots Without Match")
display(missing_semantics_df[["Root_AR", "Root_Source", "Lemma_AR", "POS"]].sort_values("Root_AR"))

# %% [notebook cell 19]
# ============================================================
# Diagnostic Search for Unmatched Roots (Non-Enriching)
# ============================================================

connection = sqlite3.connect(MAQAYIS_DB_FILE)
roots_to_check = missing_semantics_df["Root_AR"].sort_values(kind="mergesort").tolist()
diagnostic_results = []

for root in roots_to_check:
    matches = pd.read_sql_query(
        '''
        SELECT word
        FROM maqayeesul_luga
        WHERE word LIKE ?
        ORDER BY word
        LIMIT 10
        ''',
        connection,
        params=[f"%{root}%"],
    )
    diagnostic_results.append({
        "Root_AR": root,
        "Found_Diagnostic_Candidate": len(matches) > 0,
        "Candidate_Entries": ", ".join(matches["word"].astype(str).tolist()),
    })

connection.close()
diagnostic_df = pd.DataFrame(diagnostic_results)

print("=" * 70)
print("NON-ENRICHING DIAGNOSTIC SEARCH RESULTS")
print("=" * 70)
print(f"Roots Investigated : {len(diagnostic_df):,}")
print(f"Potential Candidates: {int(diagnostic_df['Found_Diagnostic_Candidate'].sum()):,}")
print(f"No Candidates      : {int((~diagnostic_df['Found_Diagnostic_Candidate']).sum()):,}")
display(diagnostic_df.head(30))

# %% [notebook cell 21]
# ============================================================
# Attach Semantic Evidence While Preserving Point-2 Lineage
# ============================================================

meaning_by_root = dict(zip(semantic_roots_df["Root_AR"], semantic_roots_df["Semantic_Meaning"]))
enriched_records = []

for _, row in hybrid_df.iterrows():
    enriched_matches = []
    for match in row["Lexical_Matches"]:
        enriched_match = match.copy()
        resolved = bool(match.get("Root_Resolved", False))
        root = match.get("Root_AR")

        if resolved:
            meaning = meaning_by_root.get(root)
            semantic_found = bool(pd.notna(meaning))
            if not semantic_found:
                meaning = None
            enriched_match["Maqayis_Lookup_Attempted"] = True
            enriched_match["Maqayis_Lookup_Root"] = root
            enriched_match["Semantic_Found"] = semantic_found
            enriched_match["Semantic_Meaning"] = meaning
            enriched_match["Semantic_Source"] = "Maqāyīs al-Lughah" if semantic_found else None
        else:
            enriched_match["Maqayis_Lookup_Attempted"] = False
            enriched_match["Maqayis_Lookup_Root"] = None
            enriched_match["Semantic_Found"] = False
            enriched_match["Semantic_Meaning"] = None
            enriched_match["Semantic_Source"] = None

        enriched_matches.append(enriched_match)

    enriched_row = row.to_dict()
    enriched_row["Lexical_Matches"] = enriched_matches
    enriched_records.append(enriched_row)

semantic_enrichment_df = pd.DataFrame(enriched_records, columns=hybrid_df.columns)

input_matches = [match for matches in hybrid_df["Lexical_Matches"] for match in matches]
output_matches = [match for matches in semantic_enrichment_df["Lexical_Matches"] for match in matches]

root_mismatches = sum(a.get("Root_AR") != b.get("Root_AR") for a, b in zip(input_matches, output_matches))
source_mismatches = sum(a.get("Root_Source") != b.get("Root_Source") for a, b in zip(input_matches, output_matches))
invented_unresolved_roots = sum(
    (not bool(a.get("Root_Resolved", False)))
    and (b.get("Root_AR") is not None and str(b.get("Root_AR")).strip() != "")
    for a, b in zip(input_matches, output_matches)
)

if root_mismatches or source_mismatches or invented_unresolved_roots:
    raise AssertionError("Authoritative Point-2 root lineage changed during semantic enrichment")

term_semantic_records = []
for match in output_matches:
    term_semantic_records.append({
        "Root_Resolved": bool(match.get("Root_Resolved", False)),
        "Root_Source": match.get("Root_Source"),
        "Lookup_Attempted": bool(match.get("Maqayis_Lookup_Attempted", False)),
        "Semantic_Found": bool(match.get("Semantic_Found", False)),
    })
term_semantic_df = pd.DataFrame(term_semantic_records)

print("=" * 70)
print("SEMANTIC ENRICHMENT COMPLETED")
print("=" * 70)
print(f"Questions Processed        : {len(semantic_enrichment_df):,}")
print(f"Query Terms                : {len(term_semantic_df):,}")
print(f"Maqāyīs Lookup Attempts    : {int(term_semantic_df['Lookup_Attempted'].sum()):,}")
print(f"Successful Semantic Matches: {int(term_semantic_df['Semantic_Found'].sum()):,}")
print(f"Root Mismatches            : {root_mismatches:,}")
print(f"Root-Source Mismatches     : {source_mismatches:,}")
print(f"Invented Unresolved Roots  : {invented_unresolved_roots:,}")

coverage_by_source = (
    term_semantic_df.loc[term_semantic_df["Root_Resolved"]]
    .groupby("Root_Source", dropna=False)["Semantic_Found"]
    .agg(Terms="size", Semantic_Matches="sum")
    .reset_index()
)
coverage_by_source["Coverage_Percent"] = 100 * coverage_by_source["Semantic_Matches"] / coverage_by_source["Terms"]
print("\nSemantic Coverage by Authoritative Root Source")
display(coverage_by_source)

# %% [notebook cell 23]
# ============================================================
# Export Corrected Semantic Enrichment Dataset
# ============================================================

export_df = semantic_enrichment_df.copy()

for column in NESTED_COLUMNS:
    export_df[column] = export_df[column].apply(
        lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
    )

export_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

reloaded_df = pd.read_csv(OUTPUT_FILE)
if len(reloaded_df) != len(export_df):
    raise AssertionError("CSV reload changed the number of question records")
if reloaded_df["Question_ID"].tolist() != export_df["Question_ID"].tolist():
    raise AssertionError("CSV reload changed Question_ID ordering")
for column in NESTED_COLUMNS:
    reloaded_df[column].map(json.loads)

print("=" * 70)
print("CORRECTED SEMANTIC ENRICHMENT DATASET EXPORTED")
print("=" * 70)
print(f"Questions Exported : {len(export_df):,}")
print(f"Output File        : {OUTPUT_FILE.name}")
print(f"Location           : {OUTPUT_DIR}")
print("CSV reload and strict JSON parsing completed successfully.")
