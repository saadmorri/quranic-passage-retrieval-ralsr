"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 3]
# ============================================================
# Cell 1: Imports, Portable Paths, and Preparation Check
# ============================================================

from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from IPython.display import display

# ------------------------------------------------------------
# Locate the live thesis workspace
# ------------------------------------------------------------

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

QUERY_SEMANTIC_FILE = (
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

PROCESSED_QPC_FILE = (
    METHODOLOGY_DIR / "Data and Resources" / "Corrected_Outputs" /
    "QPC_Passage_Side_Validation" /
    "QuranQA_Processed_Passage_Collection.csv"
)

OFFICIAL_QPC_FILE = (
    METHODOLOGY_DIR / "Data and Resources" / "QuranQA" /
    "QQA23_TaskA_QPC_v1.1.tsv"
)

CSR_OUTPUT_DIR = (
    METHODOLOGY_DIR / "CSR" / "Corrected_Outputs" /
    "CSR_Authoritative_Lineage"
)
CSR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_CSR_FILE = CSR_OUTPUT_DIR / "QuranQA_Query_CSR_Corrected.csv"
PASSAGE_CSR_FILE = CSR_OUTPUT_DIR / "QuranQA_Passage_CSR_Corrected.csv"
QUERY_TERM_LINEAGE_FILE = CSR_OUTPUT_DIR / "QuranQA_Query_CSR_Term_Lineage_Corrected.csv"
CSR_STATISTICS_FILE = CSR_OUTPUT_DIR / "CSR_Construction_Statistics_Corrected.csv"

required_files = {
    "Corrected Notebook 7 Query Representation": QUERY_SEMANTIC_FILE,
    "Corrected Notebook 9A Passage Representation": PASSAGE_REPRESENTATION_FILE,
    "Corrected Notebook 9A Vocabulary": PASSAGE_VOCABULARY_FILE,
    "Corrected Notebook 8 Processed QPC": PROCESSED_QPC_FILE,
    "Official QPC": OFFICIAL_QPC_FILE,
}

missing = [path for path in required_files.values() if not path.exists()]
if missing:
    raise FileNotFoundError("Missing authoritative input(s):\n" + "\n".join(map(str, missing)))

print("=" * 70)
print("NOTEBOOK 9D AUTHORITATIVE PREPARATION CHECK")
print("=" * 70)
print(f"Project Root : {PROJECT_ROOT}")
print(f"Output Dir   : {CSR_OUTPUT_DIR}")
for label, path in required_files.items():
    print(f"{label:<48}: FOUND")
    print(f"  {path}")

# %% [notebook cell 5]
# ============================================================
# Cell 2: Load and Validate Authoritative Input Resources
# ============================================================

query_semantics_df = pd.read_csv(QUERY_SEMANTIC_FILE, encoding="utf-8-sig")
passage_enriched_df = pd.read_csv(PASSAGE_REPRESENTATION_FILE, encoding="utf-8-sig")
passage_vocab_df = pd.read_csv(PASSAGE_VOCABULARY_FILE, encoding="utf-8-sig")
processed_qpc_df = pd.read_csv(PROCESSED_QPC_FILE, encoding="utf-8-sig")
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
    "Passage_ID", "Surah", "Start_Ayah", "End_Ayah", "Passage_Text",
    "Tokens", "Token_Vocabulary_IDs", "Roots", "Semantic_Meanings",
    "Root_Count", "Semantic_Count"
}

missing_query_columns = expected_query_columns - set(query_semantics_df.columns)
missing_passage_columns = expected_passage_columns - set(passage_enriched_df.columns)
if missing_query_columns:
    raise ValueError(f"Corrected Notebook 7 input is missing: {sorted(missing_query_columns)}")
if missing_passage_columns:
    raise ValueError(f"Corrected Notebook 9A input is missing: {sorted(missing_passage_columns)}")

official_qpc_df["Passage_ID"] = official_qpc_df["Passage_ID"].astype(str)
processed_qpc_df["Passage_ID"] = processed_qpc_df["Passage_ID"].astype(str)
passage_enriched_df["Passage_ID"] = passage_enriched_df["Passage_ID"].astype(str)

assert len(query_semantics_df) == 251
assert query_semantics_df["Question_ID"].nunique() == 251
assert len(official_qpc_df) == 1266 and official_qpc_df["Passage_ID"].nunique() == 1266
assert len(processed_qpc_df) == 1266 and processed_qpc_df["Passage_ID"].nunique() == 1266
assert len(passage_enriched_df) == 1266 and passage_enriched_df["Passage_ID"].nunique() == 1266

official_identity = official_qpc_df[["Passage_ID", "Passage_Text"]].reset_index(drop=True)
processed_identity = processed_qpc_df[["Passage_ID", "Passage_Text"]].reset_index(drop=True)
enriched_identity = passage_enriched_df[["Passage_ID", "Passage_Text"]].reset_index(drop=True)

assert official_identity.equals(processed_identity)
assert processed_identity.equals(enriched_identity)

print("=" * 70)
print("AUTHORITATIVE INPUT RESOURCES VALIDATED")
print("=" * 70)
print(f"Query Records        : {len(query_semantics_df):,}")
print(f"Passage Records      : {len(passage_enriched_df):,}")
print(f"Passage Vocabulary   : {len(passage_vocab_df):,}")
print("✓ Official, processed, and enriched QPC IDs/text/order are identical")

# %% [notebook cell 7]
# ============================================================
# Cell 3: Parse and Inspect Structured Columns
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


query_semantics_df["Parsed_Query_Terms"] = query_semantics_df["Query_Terms"].map(
    lambda value: parse_json_list(value, "Query_Terms")
)
query_semantics_df["Parsed_Query_Term_Provenance"] = query_semantics_df["Query_Term_Provenance"].map(
    lambda value: parse_json_list(value, "Query_Term_Provenance")
)
query_semantics_df["Parsed_Lexical_Matches"] = query_semantics_df["Lexical_Matches"].map(
    lambda value: parse_json_list(value, "Lexical_Matches")
)

passage_enriched_df["Parsed_Tokens"] = passage_enriched_df["Tokens"].map(
    lambda value: parse_json_list(value, "Tokens")
)
passage_enriched_df["Parsed_Token_Vocabulary_IDs"] = passage_enriched_df["Token_Vocabulary_IDs"].map(
    lambda value: parse_json_list(value, "Token_Vocabulary_IDs")
)
passage_enriched_df["Parsed_Roots"] = passage_enriched_df["Roots"].map(
    lambda value: parse_json_list(value, "Roots")
)
passage_enriched_df["Parsed_Semantic_Meanings"] = passage_enriched_df["Semantic_Meanings"].map(
    lambda value: parse_json_list(value, "Semantic_Meanings")
)

print("=" * 70)
print("STRUCTURED INPUTS PARSED")
print("=" * 70)
print(f"Query term records : {query_semantics_df['Parsed_Lexical_Matches'].map(len).sum():,}")
print(f"Passage rows       : {len(passage_enriched_df):,}")
print("✓ All required JSON structures parsed without fallback")

# %% [notebook cell 9]
# ============================================================
# Cell 4: Define Compact Semantic Representation Helpers
# ============================================================

# The established semantic-keyword normalization and caps are preserved.
ARABIC_DIACRITICS_PATTERN = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u0640]")
NON_ARABIC_LETTER_PATTERN = re.compile(r"[^\u0621-\u064A\s]")
ARABIC_ONLY_PATTERN = re.compile(r"^[\u0621-\u064A]+$")

RAW_ARABIC_STOPWORDS = {
    "من", "ما", "ماذا", "متى", "أين", "اين", "كيف", "كم", "هل",
    "هو", "هي", "هم", "هن", "انت", "انتم", "انا", "نحن",
    "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "الذي", "التي", "الذين", "اللاتي", "اللذين",
    "في", "على", "علي", "عن", "إلى", "الى", "منه", "منها", "به", "بها",
    "و", "ف", "ثم", "أو", "او", "بل", "لا", "ولا", "لم", "لن",
    "إن", "ان", "أن", "كان", "كانت", "يكون", "تكون", "قد", "لقد",
    "الله", "تعالى", "تعالي", "قال", "يقول", "يقولون", "يقال", "قوله",
    "رب", "ربه", "ربهم", "اصل", "اصلا", "اصلان", "اصول", "باب", "صحيح", "صحيحان",
    "يدل", "تدل", "احد", "احدهما", "الاخر", "والاخر", "الاول", "فالاول", "الثاني", "والثاني",
    "ومن", "ومنه", "فاما", "واما", "ويقال", "وذلك", "لانه", "بمعنى", "معنى", "نحو",
    "واحد", "واحده", "واحدان", "القاف", "الكاف", "اللام", "الميم", "النون", "الهاء",
    "الواو", "الياء", "الالف", "الباء", "التاء", "الثاء", "الجيم", "الحاء", "الخاء",
    "الدال", "الذال", "الراء", "الزاي", "السين", "الشين", "الصاد", "الضاد", "الطاء",
    "الظاء", "العين", "الغين", "الفاء", "كل", "غير", "اي", "أي", "إذا", "اذا", "إذ", "اذ",
    "وقد", "فقد", "كما", "حتى", "حين", "حيث", "بعد", "قبل", "ربما", "وربما", "الا", "إلا"
}


def normalize_arabic_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = ARABIC_DIACRITICS_PATTERN.sub("", str(text))
    text = (text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                .replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه"))
    text = NON_ARABIC_LETTER_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


ARABIC_STOPWORDS = {normalize_arabic_text(word) for word in RAW_ARABIC_STOPWORDS}


def strip_common_prefixes(token):
    variants = [token]
    if len(token) > 3 and token[0] in {"و", "ف"}:
        variants.append(token[1:])
    if len(token) > 4 and token[:2] in {"وال", "فال", "بال", "كال", "لل"}:
        variants.append("ال" + token[2:])
        variants.append(token[2:])
    if len(token) > 4 and token[0] in {"ب", "ك", "ل"}:
        variants.append(token[1:])
    return list(dict.fromkeys(variants))


def is_stopword_or_formulaic(token):
    token = normalize_arabic_text(token)
    return token in ARABIC_STOPWORDS or any(
        variant in ARABIC_STOPWORDS for variant in strip_common_prefixes(token)
    )


def is_valid_arabic_token(token, min_length=3):
    if not isinstance(token, str):
        return False
    token = normalize_arabic_text(token)
    return (
        len(token) >= min_length
        and bool(ARABIC_ONLY_PATTERN.fullmatch(token))
        and not is_stopword_or_formulaic(token)
    )


def tokenize_arabic_text(text, min_length=3):
    return [
        token for token in normalize_arabic_text(text).split()
        if is_valid_arabic_token(token, min_length=min_length)
    ]


def unique_normalized_keywords(items):
    seen = set()
    output = []
    for item in items:
        normalized = normalize_arabic_text(item)
        if normalized and normalized not in seen:
            output.append(normalized)
            seen.add(normalized)
    return output


def ordered_unique_exact(items):
    return list(dict.fromkeys(items))


def extract_authoritative_query_data(matches):
    root_records = []
    semantic_records = []
    roots = []
    meanings = []

    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("Every Notebook 7 lexical record must be a dictionary")

        resolved = bool(match.get("Root_Resolved", False))
        root = "" if match.get("Root_AR") is None else str(match.get("Root_AR")).strip()
        source = "" if match.get("Root_Source") is None else str(match.get("Root_Source")).strip()
        semantic_found = bool(match.get("Semantic_Found", False))
        meaning = match.get("Semantic_Meaning")

        if resolved:
            if not root or not source or source == "UNRESOLVED":
                raise ValueError("Resolved Notebook 7 record is missing authoritative root/source")
            roots.append(root)
            root_records.append({
                "Query_Term": match.get("Query_Term"),
                "Original_Term": match.get("Original_Term"),
                "Normalized_Term": match.get("Normalized_Term"),
                "Root_AR": root,
                "Root_Source": source,
                "Resolution_Status": match.get("Resolution_Status"),
                "Semantic_Found": semantic_found,
                "Semantic_Source": match.get("Semantic_Source"),
            })
        elif root:
            raise ValueError("Unresolved Notebook 7 record contains a selected root")

        if semantic_found:
            if not resolved or meaning is None or not str(meaning).strip():
                raise ValueError("Semantic evidence exists without a resolved root/meaning")
            meanings.append(str(meaning).strip())
            semantic_records.append({
                "Query_Term": match.get("Query_Term"),
                "Root_AR": root,
                "Root_Source": source,
                "Semantic_Source": match.get("Semantic_Source"),
                "Semantic_Meaning": str(meaning).strip(),
            })

    return root_records, semantic_records, ordered_unique_exact(roots), meanings


def extract_compact_keywords_from_meanings(meanings, max_keywords=20, max_keywords_per_meaning=5):
    selected_tokens = []
    for meaning in meanings:
        selected_tokens.extend(tokenize_arabic_text(meaning)[:max_keywords_per_meaning])
    return unique_normalized_keywords(selected_tokens)[:max_keywords]


def validate_authoritative_passage_roots(roots):
    validated = []
    for root in roots:
        root_text = "" if root is None else str(root).strip()
        if not root_text:
            raise ValueError("Corrected Notebook 9A passage roots may not be blank")
        validated.append(root_text)
    if len(validated) != len(ordered_unique_exact(validated)):
        raise ValueError("Corrected Notebook 9A passage root list contains duplicates")
    return validated


def build_csr_text(original_text, roots, semantic_keywords):
    parts = [str(original_text).strip(), " ".join(roots), " ".join(semantic_keywords)]
    return re.sub(r"\s+", " ", " ".join(part for part in parts if part)).strip()


print("=" * 70)
print("AUTHORITATIVE CSR HELPERS DEFINED")
print("=" * 70)
print("✓ Existing semantic keyword normalization and caps preserved")
print("✓ Query and passage roots are consumed exactly, without length/stopword filtering")

# %% [notebook cell 11]
# ============================================================
# Cell 5: Test CSR Construction on Authoritative Samples
# ============================================================

sample_question_id = 101
sample_passage_id = "7:85-93"

sample_query_row = query_semantics_df.loc[
    query_semantics_df["Question_ID"].astype(int) == sample_question_id
].iloc[0]
sample_passage_row = passage_enriched_df.loc[
    passage_enriched_df["Passage_ID"] == sample_passage_id
].iloc[0]

sample_root_records, sample_semantic_records, sample_query_roots, sample_query_meanings = (
    extract_authoritative_query_data(sample_query_row["Parsed_Lexical_Matches"])
)
sample_query_keywords = extract_compact_keywords_from_meanings(
    sample_query_meanings, max_keywords=20, max_keywords_per_meaning=5
)
sample_query_csr_text = build_csr_text(
    sample_query_row["Question"], sample_query_roots, sample_query_keywords
)

sample_passage_roots = validate_authoritative_passage_roots(sample_passage_row["Parsed_Roots"])
sample_passage_keywords = extract_compact_keywords_from_meanings(
    sample_passage_row["Parsed_Semantic_Meanings"],
    max_keywords=60,
    max_keywords_per_meaning=3,
)
sample_passage_csr_text = build_csr_text(
    sample_passage_row["Passage_Text"], sample_passage_roots, sample_passage_keywords
)

print("=" * 70)
print("AUTHORITATIVE SAMPLE CSR CONSTRUCTION")
print("=" * 70)
print(f"Question ID : {sample_question_id}")
print(f"Roots       : {sample_query_roots}")
print(f"Sources     : {[record['Root_Source'] for record in sample_root_records]}")
print(f"Keywords    : {sample_query_keywords}")
print(f"CSR Text    : {sample_query_csr_text}")
print(f"Passage ID  : {sample_passage_id}")
print(f"Passage roots preserved: {sample_passage_roots == sample_passage_row['Parsed_Roots']}")
print(sample_passage_csr_text[:500])

# %% [notebook cell 13]
# ============================================================
# Cell 6: Build Authoritative Compact Query Representations
# ============================================================

query_csr_records = []
query_term_lineage_records = []

for _, row in query_semantics_df.iterrows():
    question_id = row["Question_ID"]
    question = row["Question"]
    matches = row["Parsed_Lexical_Matches"]

    root_records, semantic_records, roots, semantic_meanings = (
        extract_authoritative_query_data(matches)
    )
    semantic_keywords = extract_compact_keywords_from_meanings(
        semantic_meanings,
        max_keywords=20,
        max_keywords_per_meaning=5,
    )
    csr_text = build_csr_text(question, roots, semantic_keywords)

    query_csr_records.append({
        "Dataset": row["Dataset"],
        "Question_ID": question_id,
        "Question": question,
        "Query_Terms": row["Parsed_Query_Terms"],
        "Query_Term_Provenance": row["Parsed_Query_Term_Provenance"],
        "Query_Lexical_Matches": matches,
        "CSR_Root_Records": root_records,
        "CSR_Semantic_Evidence": semantic_records,
        "CSR_Roots": roots,
        "CSR_Semantic_Keywords": semantic_keywords,
        "CSR_Text": csr_text,
        "Query_Term_Count": len(matches),
        "CSR_Root_Count": len(roots),
        "CSR_Semantic_Evidence_Count": len(semantic_records),
        "CSR_Keyword_Count": len(semantic_keywords),
        "CSR_Text_Length": len(csr_text.split()),
    })

    for match in matches:
        query_term_lineage_records.append({
            "Dataset": row["Dataset"],
            "Question_ID": question_id,
            "Question": question,
            "Query_Term": match.get("Query_Term"),
            "Original_Term": match.get("Original_Term"),
            "Normalized_Term": match.get("Normalized_Term"),
            "Root_Resolved": bool(match.get("Root_Resolved", False)),
            "Resolution_Status": match.get("Resolution_Status"),
            "Root_AR": match.get("Root_AR"),
            "Root_Source": match.get("Root_Source"),
            "Semantic_Found": bool(match.get("Semantic_Found", False)),
            "Semantic_Source": match.get("Semantic_Source"),
            "Semantic_Meaning": match.get("Semantic_Meaning"),
        })

query_csr_df = pd.DataFrame(query_csr_records)
query_term_lineage_df = pd.DataFrame(query_term_lineage_records)

print("=" * 70)
print("AUTHORITATIVE COMPACT QUERY REPRESENTATIONS CONSTRUCTED")
print("=" * 70)
print(f"Queries processed             : {len(query_csr_df):,}")
print(f"Query-term records            : {len(query_term_lineage_df):,}")
print(f"Queries with roots            : {(query_csr_df['CSR_Root_Count'] > 0).sum():,}")
print(f"Queries with semantic evidence: {(query_csr_df['CSR_Semantic_Evidence_Count'] > 0).sum():,}")
print(f"Queries with keywords         : {(query_csr_df['CSR_Keyword_Count'] > 0).sum():,}")
display(query_csr_df.head(5))

# %% [notebook cell 15]
# ============================================================
# Cell 7: Build Authoritative Compact Passage Representations
# ============================================================

valid_vocab_roots = {
    str(root).strip()
    for root in passage_vocab_df.loc[
        passage_vocab_df["Root_Resolved"].fillna(False).astype(bool), "Root_AR"
    ].dropna()
    if str(root).strip()
}

passage_csr_records = []
passage_lineage_mismatch_count = 0

for _, row in passage_enriched_df.iterrows():
    passage_id = row["Passage_ID"]
    passage_text = row["Passage_Text"]
    roots = validate_authoritative_passage_roots(row["Parsed_Roots"])

    if any(root not in valid_vocab_roots for root in roots):
        passage_lineage_mismatch_count += 1

    semantic_keywords = extract_compact_keywords_from_meanings(
        row["Parsed_Semantic_Meanings"],
        max_keywords=60,
        max_keywords_per_meaning=3,
    )
    csr_text = build_csr_text(passage_text, roots, semantic_keywords)

    passage_csr_records.append({
        "Passage_ID": passage_id,
        "Surah": row["Surah"],
        "Start_Ayah": row["Start_Ayah"],
        "End_Ayah": row["End_Ayah"],
        "Passage_Text": passage_text,
        "Source_Tokens": row["Parsed_Tokens"],
        "Source_Token_Vocabulary_IDs": row["Parsed_Token_Vocabulary_IDs"],
        "CSR_Root_Source": "CAMeL",
        "CSR_Roots": roots,
        "CSR_Semantic_Keywords": semantic_keywords,
        "CSR_Text": csr_text,
        "CSR_Root_Count": len(roots),
        "CSR_Keyword_Count": len(semantic_keywords),
        "CSR_Text_Length": len(csr_text.split()),
    })

passage_csr_df = pd.DataFrame(passage_csr_records)
assert passage_lineage_mismatch_count == 0

print("=" * 70)
print("AUTHORITATIVE COMPACT PASSAGE REPRESENTATIONS CONSTRUCTED")
print("=" * 70)
print(f"Passages processed       : {len(passage_csr_df):,}")
print(f"Unique passage IDs       : {passage_csr_df['Passage_ID'].nunique():,}")
print(f"Passage lineage mismatches: {passage_lineage_mismatch_count:,}")
print(f"Passages with roots      : {(passage_csr_df['CSR_Root_Count'] > 0).sum():,}")
print(f"Passages with keywords   : {(passage_csr_df['CSR_Keyword_Count'] > 0).sum():,}")
display(passage_csr_df.head(5))

# %% [notebook cell 17]
# ============================================================
# Cell 8: Export Corrected CSR Outputs with Strict JSON
# ============================================================

query_json_columns = [
    "Query_Terms", "Query_Term_Provenance", "Query_Lexical_Matches",
    "CSR_Root_Records", "CSR_Semantic_Evidence", "CSR_Roots",
    "CSR_Semantic_Keywords",
]
passage_json_columns = [
    "Source_Tokens", "Source_Token_Vocabulary_IDs", "CSR_Roots",
    "CSR_Semantic_Keywords",
]


def serialize_json_columns(frame, columns):
    exported = frame.copy()
    for column in columns:
        exported[column] = exported[column].map(
            lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    return exported


query_export_df = serialize_json_columns(query_csr_df, query_json_columns)
passage_export_df = serialize_json_columns(passage_csr_df, passage_json_columns)

query_export_df.to_csv(QUERY_CSR_FILE, index=False, encoding="utf-8-sig")
passage_export_df.to_csv(PASSAGE_CSR_FILE, index=False, encoding="utf-8-sig")
query_term_lineage_df.to_csv(QUERY_TERM_LINEAGE_FILE, index=False, encoding="utf-8-sig")

csr_statistics_df = pd.DataFrame([
    {
        "Resource": "Queries",
        "Records": len(query_csr_df),
        "Records_With_Roots": int((query_csr_df["CSR_Root_Count"] > 0).sum()),
        "Records_With_Keywords": int((query_csr_df["CSR_Keyword_Count"] > 0).sum()),
        "Average_Root_Count": query_csr_df["CSR_Root_Count"].mean(),
        "Average_Keyword_Count": query_csr_df["CSR_Keyword_Count"].mean(),
        "Average_CSR_Text_Length": query_csr_df["CSR_Text_Length"].mean(),
        "Minimum_CSR_Text_Length": query_csr_df["CSR_Text_Length"].min(),
        "Maximum_CSR_Text_Length": query_csr_df["CSR_Text_Length"].max(),
    },
    {
        "Resource": "Passages",
        "Records": len(passage_csr_df),
        "Records_With_Roots": int((passage_csr_df["CSR_Root_Count"] > 0).sum()),
        "Records_With_Keywords": int((passage_csr_df["CSR_Keyword_Count"] > 0).sum()),
        "Average_Root_Count": passage_csr_df["CSR_Root_Count"].mean(),
        "Average_Keyword_Count": passage_csr_df["CSR_Keyword_Count"].mean(),
        "Average_CSR_Text_Length": passage_csr_df["CSR_Text_Length"].mean(),
        "Minimum_CSR_Text_Length": passage_csr_df["CSR_Text_Length"].min(),
        "Maximum_CSR_Text_Length": passage_csr_df["CSR_Text_Length"].max(),
    },
])
csr_statistics_df.to_csv(CSR_STATISTICS_FILE, index=False, encoding="utf-8-sig")

print("=" * 70)
print("CORRECTED CSR OUTPUTS EXPORTED")
print("=" * 70)
for path in [QUERY_CSR_FILE, PASSAGE_CSR_FILE, QUERY_TERM_LINEAGE_FILE, CSR_STATISTICS_FILE]:
    print(path)
display(csr_statistics_df.round(2))

# %% [notebook cell 18]
# ============================================================
# Final Cell: Reload and Hash Corrected Notebook 9D Outputs
# ============================================================

expected_csr_files = [
    QUERY_CSR_FILE,
    PASSAGE_CSR_FILE,
    QUERY_TERM_LINEAGE_FILE,
    CSR_STATISTICS_FILE,
]

reloaded_query = pd.read_csv(QUERY_CSR_FILE, encoding="utf-8-sig")
reloaded_passage = pd.read_csv(PASSAGE_CSR_FILE, encoding="utf-8-sig")
reloaded_lineage = pd.read_csv(QUERY_TERM_LINEAGE_FILE, encoding="utf-8-sig")
reloaded_statistics = pd.read_csv(CSR_STATISTICS_FILE, encoding="utf-8-sig")

for column in query_json_columns:
    reloaded_query[column].map(json.loads)
for column in passage_json_columns:
    reloaded_passage[column].map(json.loads)

assert len(reloaded_query) == 251
assert len(reloaded_passage) == 1266
assert len(reloaded_lineage) == 1342
assert len(reloaded_statistics) == 2

print("=" * 70)
print("NOTEBOOK 9D CORRECTED EXPORT VERIFICATION")
print("=" * 70)
for file_path in expected_csr_files:
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest().upper()
    print(f"{file_path.name:<55} {digest}")
print("✓ All CSV files reload successfully")
print("✓ All nested fields are standards-compliant JSON")
