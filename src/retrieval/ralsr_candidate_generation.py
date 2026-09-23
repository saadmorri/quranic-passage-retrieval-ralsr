"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
# ============================================================
# Import Required Libraries
# ============================================================

from pathlib import Path
import hashlib
import json

import pandas as pd

from collections import defaultdict
from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

print("=" * 70)
print("LIBRARIES LOADED")
print("=" * 70)
print("✓ pathlib")
print("✓ hashlib")
print("✓ json")
print("✓ pandas")
print("✓ collections.defaultdict")
print("✓ IPython.display")

# %% [notebook cell 4]
# ============================================================
# Define Project Paths
# ============================================================

current_path = Path.cwd().resolve()
PROJECT_ROOT = None

for parent in [current_path] + list(current_path.parents):
    if (parent / "00_Project_Control").exists() and (parent / "02_Methodology_Evidence").exists():
        PROJECT_ROOT = parent
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the thesis workspace containing 00_Project_Control and 02_Methodology_Evidence."
    )

METHODOLOGY_DIR = PROJECT_ROOT / "02_Methodology_Evidence"

SEMANTIC_QUERY_FILE = (
    METHODOLOGY_DIR / "Maqāyīs Lookup" / "Corrected_Outputs" /
    "Notebook_7_Maqayis_Semantic_Enrichment" /
    "QuranQA_Semantic_Enrichment_Corrected.csv"
)

PASSAGE_CORRECTED_DIR = (
    METHODOLOGY_DIR / "QAC and Root Extraction" / "Corrected_Outputs" /
    "QPC_Passage_Side_Validation"
)

PASSAGE_REPRESENTATION_FILE = (
    PASSAGE_CORRECTED_DIR / "QuranQA_Enriched_Passage_Representation.csv"
)

PASSAGE_VOCABULARY_FILE = (
    PASSAGE_CORRECTED_DIR / "QuranQA_Linguistically_Enriched_Passages.csv"
)

OFFICIAL_QPC_FILE = (
    METHODOLOGY_DIR / "Data and Resources" / "QuranQA" /
    "QQA23_TaskA_QPC_v1.1.tsv"
)

CORRECTED_OUTPUT_DIR = (
    METHODOLOGY_DIR / "RALSR" / "Corrected_Outputs" / "RALSR_Lineage_Repair"
)
CORRECTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_RESULTS_FILE = (
    CORRECTED_OUTPUT_DIR / "QuranQA_Candidate_Retrieval_Corrected.csv"
)

required_files = [
    SEMANTIC_QUERY_FILE,
    PASSAGE_REPRESENTATION_FILE,
    PASSAGE_VOCABULARY_FILE,
    OFFICIAL_QPC_FILE,
]
missing_files = [path for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError("Missing authoritative input(s):\n" + "\n".join(map(str, missing_files)))

print("=" * 70)
print("AUTHORITATIVE RALSR PATHS")
print("=" * 70)
print(f"Project Root              : {PROJECT_ROOT}")
print(f"Notebook 7 Query Input    : {SEMANTIC_QUERY_FILE}")
print(f"Notebook 9A Passage Input : {PASSAGE_REPRESENTATION_FILE}")
print(f"Notebook 9A Vocabulary    : {PASSAGE_VOCABULARY_FILE}")
print(f"Official QPC              : {OFFICIAL_QPC_FILE}")
print(f"Corrected Candidate Output: {CANDIDATE_RESULTS_FILE}")

# %% [notebook cell 7]
# ============================================================
# Load Authoritative Retrieval Resources
# ============================================================

semantic_queries_df = pd.read_csv(SEMANTIC_QUERY_FILE, encoding="utf-8-sig")
passage_representation_df = pd.read_csv(PASSAGE_REPRESENTATION_FILE, encoding="utf-8-sig")
passage_vocab_df = pd.read_csv(PASSAGE_VOCABULARY_FILE, encoding="utf-8-sig")
official_qpc_df = pd.read_csv(
    OFFICIAL_QPC_FILE,
    sep="\t",
    header=None,
    names=["Passage_ID", "Passage_Text"],
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

expected_query_columns = {
    "Dataset", "Question_ID", "Question", "Query_Terms",
    "Query_Term_Provenance", "Lexical_Matches"
}
expected_passage_columns = {
    "Passage_ID", "Passage_Text", "Roots", "Semantic_Meanings",
    "Root_Count", "Semantic_Count"
}

if not expected_query_columns.issubset(semantic_queries_df.columns):
    raise ValueError(f"Notebook 7 input is missing columns: {sorted(expected_query_columns - set(semantic_queries_df.columns))}")
if not expected_passage_columns.issubset(passage_representation_df.columns):
    raise ValueError(f"Notebook 9A input is missing columns: {sorted(expected_passage_columns - set(passage_representation_df.columns))}")

official_qpc_ids = set(official_qpc_df["Passage_ID"].astype(str))
passage_ids = passage_representation_df["Passage_ID"].astype(str)

assert len(official_qpc_df) == 1266
assert official_qpc_df["Passage_ID"].nunique() == 1266
assert len(passage_representation_df) == 1266
assert passage_ids.nunique() == 1266
assert set(passage_ids) == official_qpc_ids

print("=" * 70)
print("AUTHORITATIVE RETRIEVAL RESOURCES LOADED")
print("=" * 70)
print(f"Semantic Query Records : {len(semantic_queries_df):,}")
print(f"Official QPC Passages   : {len(official_qpc_df):,}")
print(f"Passage Records         : {len(passage_representation_df):,}")
print(f"Vocabulary Terms        : {len(passage_vocab_df):,}")
print("✓ Corrected Notebook 9A passage IDs exactly match the official QPC")

# %% [notebook cell 9]
# ============================================================
# Build Root Inverted Index from Corrected Notebook 9A
# ============================================================

def parse_json_list(value, field_name):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must contain a JSON list")
    return parsed


root_index = defaultdict(set)

for _, row in passage_representation_df.iterrows():
    passage_id = str(row["Passage_ID"])
    roots = parse_json_list(row["Roots"], "Roots")

    for root in roots:
        root = "" if root is None else str(root).strip()
        if root:
            root_index[root].add(passage_id)

indexed_passage_ids = set().union(*root_index.values()) if root_index else set()
assert indexed_passage_ids.issubset(official_qpc_ids)

print("=" * 70)
print("AUTHORITATIVE ROOT INVERTED INDEX CREATED")
print("=" * 70)
print(f"Unique Roots Indexed : {len(root_index):,}")
print(f"Root → Passage Links : {sum(len(v) for v in root_index.values()):,}")
print(f"Passages Indexed     : {len(indexed_passage_ids):,}")
print("✓ No passage root was reconstructed")

# %% [notebook cell 11]
# ============================================================
# Validate Corrected Notebook 9A Passage-Root Lineage
# ============================================================

resolved_vocab = passage_vocab_df.loc[
    passage_vocab_df["Root_Resolved"].fillna(False).astype(bool)
].copy()

valid_vocabulary_roots = {
    str(root).strip()
    for root in resolved_vocab["Root_AR"].dropna()
    if str(root).strip()
}

representation_roots = set(root_index.keys())
passage_root_lineage_mismatches = sorted(representation_roots - valid_vocabulary_roots)

assert not passage_root_lineage_mismatches, (
    "Passage representation contains roots absent from the validated Notebook 9A vocabulary: "
    f"{passage_root_lineage_mismatches[:10]}"
)

print("=" * 70)
print("PASSAGE ROOT LINEAGE VALIDATED")
print("=" * 70)
print(f"Validated Vocabulary Roots : {len(valid_vocabulary_roots):,}")
print(f"Representation Roots       : {len(representation_roots):,}")
print(f"Lineage Mismatches          : {len(passage_root_lineage_mismatches):,}")
print("✓ Passage vocabulary is validation evidence only, never query-root authority")

# %% [notebook cell 14]
# ============================================================
# Authoritative Query-Lineage Utilities
# ============================================================

def ordered_unique(values):
    return list(dict.fromkeys(values))


def compact_root_record(item):
    return {
        "Query_Term": item.get("Query_Term"),
        "Original_Term": item.get("Original_Term"),
        "Normalized_Term": item.get("Normalized_Term"),
        "Root_AR": item.get("Root_AR"),
        "Root_Source": item.get("Root_Source"),
        "Resolution_Status": item.get("Resolution_Status"),
        "Semantic_Found": bool(item.get("Semantic_Found", False)),
        "Semantic_Source": item.get("Semantic_Source"),
    }


print("=" * 70)
print("AUTHORITATIVE QUERY-LINEAGE UTILITIES READY")
print("=" * 70)
print("✓ No passage-vocabulary query lookup exists in the corrected path")

# %% [notebook cell 15]
# ============================================================
# Query-Root Authority Guard
# ============================================================

def validate_authoritative_match(item):
    resolved = bool(item.get("Root_Resolved", False))
    root = item.get("Root_AR")
    source = item.get("Root_Source")

    root_text = "" if root is None else str(root).strip()
    source_text = "" if source is None else str(source).strip()

    if resolved and (not root_text or not source_text):
        raise ValueError(
            "Notebook 7 contains a resolved query record without a root or Root_Source"
        )
    if not resolved and root_text:
        raise ValueError(
            "Notebook 7 contains an unresolved query record with a selected root"
        )

    return resolved, root_text, source_text


print("=" * 70)
print("QUERY-ROOT AUTHORITY GUARD READY")
print("=" * 70)

# %% [notebook cell 16]
# ============================================================
# Retrieve Candidate Passages
# ============================================================

candidate_results = []

for _, query in semantic_queries_df.iterrows():
    question_id = query["Question_ID"]
    question = query["Question"]
    dataset = query["Dataset"]

    query_terms = parse_json_list(query["Query_Terms"], "Query_Terms")
    query_term_provenance = parse_json_list(
        query["Query_Term_Provenance"], "Query_Term_Provenance"
    )
    lexical_matches = parse_json_list(query["Lexical_Matches"], "Lexical_Matches")

    query_roots = []
    query_semantics = []
    query_root_records = []

    for item in lexical_matches:
        resolved, root, source = validate_authoritative_match(item)
        if not resolved:
            continue

        query_roots.append(root)
        query_root_records.append(compact_root_record(item))

        semantic = item.get("Semantic_Meaning")
        if bool(item.get("Semantic_Found", False)) and semantic is not None:
            semantic = str(semantic).strip()
            if semantic:
                query_semantics.append(semantic)

    query_roots = ordered_unique(query_roots)
    query_semantics = ordered_unique(query_semantics)

    candidate_passages = set()
    for root in query_roots:
        candidate_passages.update(root_index.get(root, set()))

    invalid_candidate_ids = candidate_passages - official_qpc_ids
    if invalid_candidate_ids:
        raise ValueError(f"Candidate IDs outside official QPC: {sorted(invalid_candidate_ids)[:10]}")

    candidate_results.append({
        "Dataset": dataset,
        "Question_ID": question_id,
        "Question": question,
        "Query_Terms": query_terms,
        "Query_Term_Provenance": query_term_provenance,
        "Query_Lexical_Matches": lexical_matches,
        "Query_Root_Records": query_root_records,
        "Query_Roots": query_roots,
        "Query_Semantics": query_semantics,
        "Candidate_Passages": sorted(candidate_passages),
        "Candidate_Count": len(candidate_passages),
    })

candidate_df = pd.DataFrame(candidate_results)

assert len(candidate_df) == len(semantic_queries_df)
assert candidate_df["Question_ID"].nunique() == semantic_queries_df["Question_ID"].nunique()

print("=" * 70)
print("AUTHORITATIVE CANDIDATE RETRIEVAL COMPLETED")
print("=" * 70)
print(f"Questions Processed : {len(candidate_df):,}")
print(f"Average Candidates  : {candidate_df['Candidate_Count'].mean():.2f}")
print(f"Zero-Candidate QIDs : {(candidate_df['Candidate_Count'] == 0).sum():,}")
display(candidate_df[["Question_ID", "Query_Roots", "Candidate_Count"]].head(10))

# %% [notebook cell 17]
# ============================================================
# Validate and Export Corrected Candidate Retrieval Results
# ============================================================

json_columns = [
    "Query_Terms",
    "Query_Term_Provenance",
    "Query_Lexical_Matches",
    "Query_Root_Records",
    "Query_Roots",
    "Query_Semantics",
    "Candidate_Passages",
]

export_candidate_df = candidate_df.copy()
for column in json_columns:
    export_candidate_df[column] = export_candidate_df[column].map(
        lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
    )

export_candidate_df.to_csv(
    CANDIDATE_RESULTS_FILE,
    index=False,
    encoding="utf-8-sig",
)

reloaded_candidate_df = pd.read_csv(CANDIDATE_RESULTS_FILE, encoding="utf-8-sig")
assert len(reloaded_candidate_df) == len(candidate_df)
for column in json_columns:
    reloaded_candidate_df[column].map(json.loads)

candidate_sha256 = hashlib.sha256(CANDIDATE_RESULTS_FILE.read_bytes()).hexdigest().upper()

print("=" * 70)
print("CORRECTED CANDIDATE RETRIEVAL EXPORTED")
print("=" * 70)
print(f"Questions : {len(candidate_df):,}")
print(f"Output    : {CANDIDATE_RESULTS_FILE}")
print(f"SHA-256   : {candidate_sha256}")
print("✓ Strict JSON fields reload successfully")
