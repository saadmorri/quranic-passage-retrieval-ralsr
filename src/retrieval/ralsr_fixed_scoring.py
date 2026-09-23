"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 2]
# ============================================================
# Notebook 9C – Root-Aware Lexical Semantic Retrieval (RALSR)
# Ranking Model
#
# Cell 1: Load Libraries
# ============================================================

from pathlib import Path

import ast
import json

import numpy as np
import pandas as pd

from collections import defaultdict

from IPython.display import display

print("=" * 70)
print("LIBRARIES LOADED SUCCESSFULLY")
print("=" * 70)

print("✓ pathlib")
print("✓ ast")
print("✓ json")
print("✓ numpy")
print("✓ pandas")
print("✓ collections")
print("✓ IPython.display")

# %% [notebook cell 3]
# ============================================================
# Locate Thesis Root and Define Split-Safe RALSR Paths
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
RALSR_SOURCE_DIR = (
    METHODOLOGY_DIR / "RALSR" / "Corrected_Outputs" / "RALSR_Lineage_Repair"
)
RALSR_OUTPUT_DIR = (
    METHODOLOGY_DIR / "RALSR" / "Corrected_Outputs" /
    "RALSR_Split_Safe_Normalization_No_Answer"
)
RALSR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_FILE = RALSR_SOURCE_DIR / "QuranQA_Candidate_Retrieval_Corrected.csv"
RAW_FEATURE_FILE = RALSR_SOURCE_DIR / "RALSR_LEM_Features_Corrected.csv"
OLD_ALL_RESULTS_FILE = RALSR_SOURCE_DIR / "RALSR_All_Ranked_Corrected.csv"
OLD_TOP100_RESULTS_FILE = RALSR_SOURCE_DIR / "RALSR_Top100_Corrected.csv"

PASSAGE_FILE = (
    METHODOLOGY_DIR / "QAC and Root Extraction" / "Corrected_Outputs" /
    "QPC_Passage_Side_Validation" / "QuranQA_Enriched_Passage_Representation.csv"
)

QUESTION_RESOURCE_DIR = METHODOLOGY_DIR / "Data and Resources" / "QuranQA"
TRAIN_QUESTION_FILE = QUESTION_RESOURCE_DIR / "QQA23_TaskA_ayatec_v1.2_train.tsv"
DEV_QUESTION_FILE = QUESTION_RESOURCE_DIR / "QQA23_TaskA_ayatec_v1.2_dev.tsv"
TEST_QUESTION_FILE = QUESTION_RESOURCE_DIR / "QQA23_TaskA_ayatec_v1.2_test.tsv"
OFFICIAL_QPC_FILE = QUESTION_RESOURCE_DIR / "QQA23_TaskA_QPC_v1.1.tsv"

NORMALIZATION_PARAMETERS_FILE = RALSR_OUTPUT_DIR / "RALSR_Train_Only_Normalization_Parameters.csv"
SPLIT_MAPPING_FILE = RALSR_OUTPUT_DIR / "QuranQA_Question_Split_Mapping.csv"
OUT_OF_RANGE_FILE = RALSR_OUTPUT_DIR / "RALSR_Dev_Test_Out_Of_Training_Range.csv"
LEM_FEATURES_FILE = RALSR_OUTPUT_DIR / "RALSR_LEM_Features_Split_Safe.csv"
ALL_RESULTS_FILE = RALSR_OUTPUT_DIR / "RALSR_All_Ranked_Split_Safe_With_No_Answer.csv"
TOP100_RESULTS_FILE = RALSR_OUTPUT_DIR / "RALSR_Top100_Split_Safe_With_No_Answer.csv"
OLD_NEW_COMPARISON_FILE = RALSR_OUTPUT_DIR / "RALSR_Old_Global_vs_Train_Only_Comparison.csv"
STRUCTURAL_VALIDATION_FILE = RALSR_OUTPUT_DIR / "RALSR_Split_Safe_Structural_Validation.json"

required_files = [
    CANDIDATE_FILE, RAW_FEATURE_FILE, OLD_ALL_RESULTS_FILE,
    PASSAGE_FILE, TRAIN_QUESTION_FILE, DEV_QUESTION_FILE,
    TEST_QUESTION_FILE, OFFICIAL_QPC_FILE,
]
for required in required_files:
    if not required.exists():
        raise FileNotFoundError(f"Missing authoritative input: {required}")

print("=" * 70)
print("SPLIT-SAFE RALSR PATHS INITIALIZED")
print("=" * 70)
print(f"Project Root        : {PROJECT_ROOT}")
print(f"Notebook 9B Input   : {CANDIDATE_FILE}")
print(f"Raw Feature Input   : {RAW_FEATURE_FILE}")
print(f"Corrected Output Dir: {RALSR_OUTPUT_DIR}")

# %% [notebook cell 5]
# ============================================================
# Load Authoritative Ranking Resources
# ============================================================

candidate_df = pd.read_csv(CANDIDATE_FILE, encoding="utf-8-sig")
passage_representation_df = pd.read_csv(PASSAGE_FILE, encoding="utf-8-sig")

expected_candidate_columns = {
    "Dataset", "Question_ID", "Question", "Query_Terms",
    "Query_Lexical_Matches", "Query_Root_Records", "Query_Roots",
    "Query_Semantics", "Candidate_Passages", "Candidate_Count"
}
missing_candidate_columns = expected_candidate_columns - set(candidate_df.columns)
if missing_candidate_columns:
    raise ValueError(f"Corrected Notebook 9B input is missing: {sorted(missing_candidate_columns)}")

print("=" * 70)
print("AUTHORITATIVE RANKING RESOURCES LOADED")
print("=" * 70)
print(f"Candidate Questions : {len(candidate_df):,}")
print(f"Passage Records     : {len(passage_representation_df):,}")
print(f"Candidate Input     : {CANDIDATE_FILE.name}")
print(f"Passage Input       : {PASSAGE_FILE.name}")

# %% [notebook cell 7]
# ============================================================
# Build Authoritative Passage Lookup
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


passage_lookup = {}

for _, row in passage_representation_df.iterrows():
    passage_id = str(row["Passage_ID"])
    roots = parse_json_list(row["Roots"], "Roots")
    semantics = parse_json_list(row["Semantic_Meanings"], "Semantic_Meanings")

    passage_lookup[passage_id] = {
        "Passage_Text": row["Passage_Text"],
        "Roots": {str(root).strip() for root in roots if root is not None and str(root).strip()},
        "Semantic_Meanings": {
            str(item).strip() for item in semantics if item is not None and str(item).strip()
        },
        "Root_Count": row["Root_Count"],
        "Semantic_Count": row["Semantic_Count"],
        "Surah": row["Surah"],
        "Start_Ayah": row["Start_Ayah"],
        "End_Ayah": row["End_Ayah"],
    }

assert len(passage_lookup) == 1266

print("=" * 70)
print("AUTHORITATIVE PASSAGE LOOKUP CREATED")
print("=" * 70)
print(f"Indexed Passages : {len(passage_lookup):,}")
print("✓ No passage token was reanalysed")

# %% [notebook cell 9]
# ============================================================
# Construct the Linguistic Evidence Model (LEM)
# ============================================================

lem_records = []

for _, query in candidate_df.iterrows():
    question_id = query["Question_ID"]
    question = query["Question"]
    dataset = query["Dataset"]

    query_terms = parse_json_list(query["Query_Terms"], "Query_Terms")
    query_root_records = parse_json_list(query["Query_Root_Records"], "Query_Root_Records")
    query_roots = parse_json_list(query["Query_Roots"], "Query_Roots")
    query_semantics = parse_json_list(query["Query_Semantics"], "Query_Semantics")
    candidates = parse_json_list(query["Candidate_Passages"], "Candidate_Passages")

    query_root_set = {str(root).strip() for root in query_roots if root is not None and str(root).strip()}
    query_semantic_set = {
        str(item).strip() for item in query_semantics if item is not None and str(item).strip()
    }
    query_root_sources = [
        {
            "Query_Term": item.get("Query_Term"),
            "Root_AR": item.get("Root_AR"),
            "Root_Source": item.get("Root_Source"),
            "Resolution_Status": item.get("Resolution_Status"),
            "Semantic_Found": item.get("Semantic_Found"),
            "Semantic_Source": item.get("Semantic_Source"),
        }
        for item in query_root_records
    ]

    for passage_id in candidates:
        passage_id = str(passage_id)
        if passage_id not in passage_lookup:
            raise ValueError(f"Candidate passage absent from corrected Notebook 9A input: {passage_id}")

        passage = passage_lookup[passage_id]
        passage_root_set = passage["Roots"]
        passage_semantic_set = passage["Semantic_Meanings"]

        shared_roots = sorted(query_root_set & passage_root_set)
        query_only_roots = sorted(query_root_set - passage_root_set)
        passage_only_roots = sorted(passage_root_set - query_root_set)
        shared_semantics = sorted(query_semantic_set & passage_semantic_set)

        query_root_count = len(query_root_set)
        passage_root_count = len(passage_root_set)
        semantic_count = len(passage_semantic_set)
        root_coverage = round(len(shared_roots) / query_root_count, 4) if query_root_count else 0.0

        lem_records.append({
            "Dataset": dataset,
            "Question_ID": question_id,
            "Question": question,
            "Passage_ID": passage_id,
            "Query_Terms": query_terms,
            "Query_Roots": sorted(query_root_set),
            "Query_Root_Sources": query_root_sources,
            "Shared_Roots": shared_roots,
            "Query_Only_Roots": query_only_roots,
            "Passage_Only_Roots": passage_only_roots,
            "Shared_Semantics": shared_semantics,
            "Query_Root_Count": query_root_count,
            "Passage_Root_Count": passage_root_count,
            "Semantic_Count": semantic_count,
            "Root_Coverage": root_coverage,
        })

lem_df = pd.DataFrame(lem_records)

print("=" * 70)
print("LINGUISTIC EVIDENCE MODEL (LEM) CREATED")
print("=" * 70)
print(f"Question–Passage Pairs : {len(lem_df):,}")
print(f"Average Root Coverage  : {lem_df['Root_Coverage'].mean():.3f}")
display(lem_df.head(10))

# %% [notebook cell 11]
# ============================================================
# Build the Existing Six-Feature RALSR Matrix
# ============================================================

feature_records = []

for _, row in lem_df.iterrows():
    shared_roots = row["Shared_Roots"]
    query_only = row["Query_Only_Roots"]
    shared_semantics = row["Shared_Semantics"]

    root_overlap = len(shared_roots)
    semantic_overlap = len(shared_semantics)
    root_coverage = row["Root_Coverage"]

    union_size = len(query_only) + row["Passage_Root_Count"]
    jaccard_similarity = round(root_overlap / union_size, 4) if union_size > 0 else 0.0

    feature_records.append({
        "Dataset": row["Dataset"],
        "Question_ID": row["Question_ID"],
        "Question": row["Question"],
        "Passage_ID": row["Passage_ID"],
        "Query_Terms": row["Query_Terms"],
        "Query_Roots": row["Query_Roots"],
        "Query_Root_Sources": row["Query_Root_Sources"],
        "Shared_Roots": shared_roots,
        "Query_Only_Roots": query_only,
        "Passage_Only_Roots": row["Passage_Only_Roots"],
        "Shared_Semantics": shared_semantics,
        "Root_Overlap": root_overlap,
        "Semantic_Overlap": semantic_overlap,
        "Root_Coverage": root_coverage,
        "Jaccard_Root_Similarity": jaccard_similarity,
        "Passage_Root_Count": row["Passage_Root_Count"],
        "Semantic_Count": row["Semantic_Count"],
    })

feature_df = pd.DataFrame(feature_records)

print("=" * 70)
print("RALSR SIX-FEATURE MATRIX CREATED")
print("=" * 70)
print(f"Feature Vectors : {len(feature_df):,}")
display(feature_df[[
    "Root_Overlap", "Semantic_Overlap", "Root_Coverage",
    "Jaccard_Root_Similarity", "Passage_Root_Count", "Semantic_Count"
]].mean().to_frame("Average"))

# %% [notebook cell 13]
# ============================================================
# Compute Split-Safe RALSR Scores from Preserved Raw Features
# ============================================================

FEATURE_WEIGHTS = {
    "Root_Coverage": 0.30,
    "Root_Overlap": 0.25,
    "Semantic_Overlap": 0.20,
    "Jaccard_Root_Similarity": 0.15,
    "Passage_Root_Count": 0.05,
    "Semantic_Count": 0.05,
}


def load_question_split(path, split_name):
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["Question_ID", "Question"],
        dtype={"Question_ID": str, "Question": str},
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    frame["Question_ID"] = frame["Question_ID"].astype(str).str.strip().astype(int)
    frame["Split"] = split_name
    return frame


split_mapping_df = pd.concat(
    [
        load_question_split(TRAIN_QUESTION_FILE, "train"),
        load_question_split(DEV_QUESTION_FILE, "dev"),
        load_question_split(TEST_QUESTION_FILE, "test"),
    ],
    ignore_index=True,
)

expected_split_counts = {"train": 174, "dev": 25, "test": 52}
actual_split_counts = split_mapping_df.groupby("Split")["Question_ID"].nunique().to_dict()
assert actual_split_counts == expected_split_counts
assert len(split_mapping_df) == 251
assert split_mapping_df["Question_ID"].nunique() == 251
assert split_mapping_df.loc[split_mapping_df["Question_ID"] == 504, "Split"].tolist() == ["test"]

split_by_qid = split_mapping_df.set_index("Question_ID")["Split"].to_dict()

# Reuse the already validated Notebook-9B candidate artifact and the preserved
# raw six-feature table. No candidate generation or upstream notebook is run.
candidate_authority_df = pd.read_csv(CANDIDATE_FILE, encoding="utf-8-sig")
candidate_authority_df["Question_ID"] = candidate_authority_df["Question_ID"].astype(int)
candidate_authority_df["Split"] = candidate_authority_df["Question_ID"].map(split_by_qid)

raw_feature_df = pd.read_csv(RAW_FEATURE_FILE, encoding="utf-8-sig")
raw_feature_df["Question_ID"] = raw_feature_df["Question_ID"].astype(int)
raw_feature_df["Passage_ID"] = raw_feature_df["Passage_ID"].astype(str)
raw_feature_df["Split"] = raw_feature_df["Question_ID"].map(split_by_qid)

assert len(candidate_authority_df) == 251
assert candidate_authority_df["Question_ID"].nunique() == 251
assert len(raw_feature_df) == 58618
assert raw_feature_df["Question_ID"].nunique() == 238
assert not candidate_authority_df["Split"].isna().any()
assert not raw_feature_df["Split"].isna().any()

ranking_df = raw_feature_df.copy()
feature_columns = list(FEATURE_WEIGHTS.keys())
for feature in feature_columns:
    ranking_df[feature] = pd.to_numeric(ranking_df[feature], errors="raise")

train_feature_df = ranking_df.loc[ranking_df["Split"] == "train"]
if train_feature_df.empty:
    raise ValueError("Training candidate subset is empty")

normalization_records = []
out_of_range_records = []

for feature in feature_columns:
    train_minimum = float(train_feature_df[feature].min())
    train_maximum = float(train_feature_df[feature].max())
    denominator = train_maximum - train_minimum
    zero_range = denominator == 0.0

    normalization_records.append({
        "Feature": feature,
        "Training_Minimum": train_minimum,
        "Training_Maximum": train_maximum,
        "Denominator": denominator,
        "Zero_Range": zero_range,
        "Clipping_Applied": False,
    })

    if zero_range:
        ranking_df[feature + "_Norm"] = 0.0
    else:
        ranking_df[feature + "_Norm"] = (
            (ranking_df[feature] - train_minimum) / denominator
        )

    for split_name in ["dev", "test"]:
        split_values = ranking_df.loc[ranking_df["Split"] == split_name, feature]
        below = split_values < train_minimum
        above = split_values > train_maximum
        out_of_range_records.append({
            "Feature": feature,
            "Split": split_name,
            "Below_Training_Minimum_Count": int(below.sum()),
            "Above_Training_Maximum_Count": int(above.sum()),
            "Below_Examples": json.dumps(
                sorted(split_values.loc[below].unique().tolist())[:10],
                ensure_ascii=False,
                allow_nan=False,
            ),
            "Above_Examples": json.dumps(
                sorted(split_values.loc[above].unique().tolist())[:10],
                ensure_ascii=False,
                allow_nan=False,
            ),
        })

normalization_parameters_df = pd.DataFrame(normalization_records)
out_of_range_df = pd.DataFrame(out_of_range_records)

ranking_df["RALSR_Score"] = 0.0
for feature, weight in FEATURE_WEIGHTS.items():
    ranking_df["RALSR_Score"] += ranking_df[feature + "_Norm"] * weight
ranking_df["RALSR_Score"] = ranking_df["RALSR_Score"].round(6)

candidate_pair_counts = ranking_df.groupby("Split").size().to_dict()

print("=" * 70)
print("TRAIN-ONLY NORMALIZATION AND RALSR SCORING COMPLETED")
print("=" * 70)
print(f"Question split counts : {actual_split_counts}")
print(f"Candidate pair counts : {candidate_pair_counts}")
print(f"Rescored pairs         : {len(ranking_df):,}")
print("Weights                : fixed and unchanged")
print("Qrels loaded or used   : False")
print("Clipping applied       : False")
display(normalization_parameters_df)
display(out_of_range_df)

# %% [notebook cell 15]
# ============================================================
# Deterministic Ranking and Explicit Zero-Candidate No-Answer Rows
# ============================================================

ranking_df = ranking_df.sort_values(
    by=["Question_ID", "RALSR_Score", "Passage_ID"],
    ascending=[True, False, True],
    kind="mergesort",
).reset_index(drop=True)
ranking_df["Rank"] = ranking_df.groupby("Question_ID").cumcount() + 1
ranking_df["No_Answer_Prediction"] = False
ranking_df["No_Answer_Reason"] = ""

zero_candidate_df = candidate_authority_df.loc[
    pd.to_numeric(candidate_authority_df["Candidate_Count"], errors="raise") == 0
].copy()
zero_candidate_qids = sorted(zero_candidate_df["Question_ID"].astype(int).tolist())

no_answer_rows = []
nested_columns = [
    "Query_Terms", "Query_Roots", "Query_Root_Sources",
    "Shared_Roots", "Query_Only_Roots", "Passage_Only_Roots",
    "Shared_Semantics",
]

for _, row in zero_candidate_df.iterrows():
    record = {column: np.nan for column in ranking_df.columns}
    record.update({
        "Dataset": row.get("Dataset"),
        "Split": row["Split"],
        "Question_ID": int(row["Question_ID"]),
        "Question": row["Question"],
        "Passage_ID": "-1",
        "RALSR_Score": 0.0,
        "Rank": 1,
        "No_Answer_Prediction": True,
        "No_Answer_Reason": "zero_valid_candidates",
    })
    for column in nested_columns:
        if column in record:
            record[column] = "[]"
    no_answer_rows.append(record)

no_answer_df = pd.DataFrame(no_answer_rows, columns=ranking_df.columns)

complete_ranking_df = pd.concat(
    [ranking_df, no_answer_df], ignore_index=True
).sort_values(
    by=["Question_ID", "Rank", "Passage_ID"],
    ascending=[True, True, True],
    kind="mergesort",
).reset_index(drop=True)

normal_top100_df = (
    ranking_df.groupby("Question_ID", group_keys=False).head(100).reset_index(drop=True)
)
top100_with_no_answer_df = pd.concat(
    [normal_top100_df, no_answer_df], ignore_index=True
).sort_values(
    by=["Question_ID", "Rank", "Passage_ID"],
    ascending=[True, True, True],
    kind="mergesort",
).reset_index(drop=True)

print("=" * 70)
print("SPLIT-SAFE RALSR RANKING AND NO-ANSWER DECISIONS COMPLETED")
print("=" * 70)
print(f"Ranked questions        : {ranking_df['Question_ID'].nunique():,}")
print(f"Zero-candidate questions: {len(zero_candidate_qids):,}")
print(f"Explicit -1 rows        : {len(no_answer_df):,}")
print(f"All questions represented: {complete_ranking_df['Question_ID'].nunique():,}")
print(f"Zero-candidate qids     : {zero_candidate_qids}")

# %% [notebook cell 16]
# ============================================================
# Diagnostic: Questions Without Ranked Candidates
# ============================================================

all_questions = set(candidate_df["Question_ID"])

ranked_questions = set(ranking_df["Question_ID"])

missing_questions = sorted(all_questions - ranked_questions)

print("=" * 70)
print("QUESTIONS WITHOUT RANKED PASSAGES")
print("=" * 70)

print(missing_questions)

display(

    candidate_df[
        candidate_df["Question_ID"].isin(missing_questions)
    ]

)

# %% [notebook cell 18]
# ============================================================
# Cell 9: Inspect Top Ranked Passages
# ============================================================

TOP_K = 10

# ------------------------------------------------------------
# Select a question to inspect
# ------------------------------------------------------------

QUESTION_ID = 101

inspection_df = (

    ranking_df

    .loc[ranking_df["Question_ID"] == QUESTION_ID]

    .sort_values("Rank")

    .head(TOP_K)

)

print("=" * 70)
print(f"TOP {TOP_K} RANKED PASSAGES")
print("=" * 70)

print(f"Question ID : {QUESTION_ID}")
print(f"Question    : {inspection_df.iloc[0]['Question']}")

display(

    inspection_df[
        [
            "Rank",
            "Passage_ID",
            "RALSR_Score",
            "Root_Overlap",
            "Semantic_Overlap",
            "Root_Coverage",
            "Jaccard_Root_Similarity"
        ]
    ]

)

# %% [notebook cell 20]
# ============================================================
# Export, Compare, Validate, and Hash Split-Safe RALSR Artifacts
# ============================================================

import hashlib


def canonicalize_nested_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        parsed = []
    elif isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return json.dumps(parsed, ensure_ascii=False, allow_nan=False)


def serialize_nested_columns(frame):
    exported = frame.copy()
    for column in nested_columns:
        if column in exported.columns:
            exported[column] = exported[column].map(canonicalize_nested_value)
    return exported


split_mapping_df.to_csv(SPLIT_MAPPING_FILE, index=False, encoding="utf-8-sig")
normalization_parameters_df.to_csv(
    NORMALIZATION_PARAMETERS_FILE, index=False, encoding="utf-8-sig"
)
out_of_range_df.to_csv(OUT_OF_RANGE_FILE, index=False, encoding="utf-8-sig")
serialize_nested_columns(ranking_df).to_csv(
    LEM_FEATURES_FILE, index=False, encoding="utf-8-sig"
)
serialize_nested_columns(complete_ranking_df).to_csv(
    ALL_RESULTS_FILE, index=False, encoding="utf-8-sig"
)
serialize_nested_columns(top100_with_no_answer_df).to_csv(
    TOP100_RESULTS_FILE, index=False, encoding="utf-8-sig"
)

# Compare prior global-normalization ranking with the new train-only result.
old_ranking_df = pd.read_csv(OLD_ALL_RESULTS_FILE, encoding="utf-8-sig")
old_ranking_df["Question_ID"] = old_ranking_df["Question_ID"].astype(int)
old_ranking_df["Passage_ID"] = old_ranking_df["Passage_ID"].astype(str)

old_pairs = set(zip(old_ranking_df["Question_ID"], old_ranking_df["Passage_ID"]))
new_pairs = set(zip(ranking_df["Question_ID"], ranking_df["Passage_ID"]))
candidate_sets_identical = old_pairs == new_pairs

score_comparison_df = old_ranking_df[
    ["Question_ID", "Passage_ID", "RALSR_Score", "Rank"]
].rename(columns={"RALSR_Score": "Old_Global_Score", "Rank": "Old_Rank"}).merge(
    ranking_df[["Question_ID", "Passage_ID", "RALSR_Score", "Rank"]].rename(
        columns={"RALSR_Score": "New_Train_Only_Score", "Rank": "New_Rank"}
    ),
    on=["Question_ID", "Passage_ID"],
    how="outer",
    validate="one_to_one",
)
score_comparison_df["Score_Changed"] = (
    score_comparison_df["Old_Global_Score"] != score_comparison_df["New_Train_Only_Score"]
)
score_comparison_df["Rank_Changed"] = (
    score_comparison_df["Old_Rank"] != score_comparison_df["New_Rank"]
)
score_comparison_df.to_csv(
    OLD_NEW_COMPARISON_FILE, index=False, encoding="utf-8-sig"
)


def ranked_sequence(frame, qid, depth):
    return (
        frame.loc[frame["Question_ID"] == qid]
        .sort_values("Rank")["Passage_ID"].astype(str).head(depth).tolist()
    )


ranked_qids = sorted(ranking_df["Question_ID"].unique().tolist())
top1_changed = 0
top10_membership_changed = 0
top10_order_changed = 0
for qid in ranked_qids:
    old_top1 = ranked_sequence(old_ranking_df, qid, 1)
    new_top1 = ranked_sequence(ranking_df, qid, 1)
    old_top10 = ranked_sequence(old_ranking_df, qid, 10)
    new_top10 = ranked_sequence(ranking_df, qid, 10)
    top1_changed += old_top1 != new_top1
    top10_membership_changed += set(old_top10) != set(new_top10)
    top10_order_changed += old_top10 != new_top10

# Structural validation.
official_qpc_df = pd.read_csv(
    OFFICIAL_QPC_FILE,
    sep="\t",
    header=None,
    names=["Passage_ID", "Passage_Text"],
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)
official_qpc_ids = set(official_qpc_df["Passage_ID"].astype(str))
authoritative_qids = set(split_mapping_df["Question_ID"].astype(int))
output_qids = set(top100_with_no_answer_df["Question_ID"].astype(int))

normal_output = top100_with_no_answer_df.loc[
    top100_with_no_answer_df["Passage_ID"].astype(str) != "-1"
].copy()
no_answer_output = top100_with_no_answer_df.loc[
    top100_with_no_answer_df["Passage_ID"].astype(str) == "-1"
].copy()

invalid_normal_qpc_ids = sorted(
    set(normal_output["Passage_ID"].astype(str)) - official_qpc_ids
)
mixed_output_qids = sorted(
    set(normal_output["Question_ID"].astype(int)) &
    set(no_answer_output["Question_ID"].astype(int))
)
duplicate_pairs = int(
    top100_with_no_answer_df.duplicated(["Question_ID", "Passage_ID"]).sum()
)
score_values = pd.to_numeric(top100_with_no_answer_df["RALSR_Score"], errors="coerce")
missing_or_nonfinite_scores = int((~np.isfinite(score_values)).sum())

rank_continuity_violations = 0
score_order_violations = 0
tie_order_violations = 0
for qid, group in normal_output.groupby("Question_ID", sort=False):
    ordered = group.sort_values("Rank")
    expected_ranks = list(range(1, len(ordered) + 1))
    if ordered["Rank"].astype(int).tolist() != expected_ranks:
        rank_continuity_violations += 1
    scores = ordered["RALSR_Score"].astype(float).tolist()
    if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
        score_order_violations += 1
    for _, tied in ordered.groupby("RALSR_Score", sort=False):
        passage_ids = tied["Passage_ID"].astype(str).tolist()
        if passage_ids != sorted(passage_ids):
            tie_order_violations += 1

zero_set_from_candidates = set(zero_candidate_qids)
zero_set_from_output = set(no_answer_output["Question_ID"].astype(int))

validation = {
    "authoritative_question_count": len(authoritative_qids),
    "output_question_count": len(output_qids),
    "missing_qids": sorted(authoritative_qids - output_qids),
    "unexpected_qids": sorted(output_qids - authoritative_qids),
    "ranked_question_count": int(normal_output["Question_ID"].nunique()),
    "no_answer_question_count": int(no_answer_output["Question_ID"].nunique()),
    "zero_candidate_qids": zero_candidate_qids,
    "zero_candidate_split_distribution": {
        key: int(value)
        for key, value in zero_candidate_df.groupby("Split").size().to_dict().items()
    },
    "explicit_minus_one_rows": len(no_answer_output),
    "minus_one_rank_violations": int((no_answer_output["Rank"].astype(int) != 1).sum()),
    "minus_one_reason_violations": int(
        (no_answer_output["No_Answer_Reason"] != "zero_valid_candidates").sum()
    ),
    "minus_one_set_matches_zero_candidates": zero_set_from_candidates == zero_set_from_output,
    "mixed_minus_one_and_normal_qids": mixed_output_qids,
    "invalid_normal_qpc_ids": invalid_normal_qpc_ids,
    "duplicate_pairs": duplicate_pairs,
    "missing_or_nonfinite_scores": missing_or_nonfinite_scores,
    "rank_continuity_violations": rank_continuity_violations,
    "score_order_violations": score_order_violations,
    "tie_order_violations": tie_order_violations,
    "full_ranking_rows": len(complete_ranking_df),
    "top100_normal_passage_rows": len(normal_output),
    "top100_total_rows": len(top100_with_no_answer_df),
    "candidate_sets_identical_to_previous": candidate_sets_identical,
    "candidate_scores_changed": int(score_comparison_df["Score_Changed"].sum()),
    "candidate_ranks_changed": int(score_comparison_df["Rank_Changed"].sum()),
    "questions_with_changed_top1": int(top1_changed),
    "questions_with_changed_top10_membership": int(top10_membership_changed),
    "questions_with_changed_top10_order": int(top10_order_changed),
    "qrels_loaded_or_used": False,
    "clipping_applied": False,
    "qid_504_split": split_by_qid[504],
    "qid_504_used_for_training_normalization": False,
}

if validation["missing_qids"] or validation["unexpected_qids"]:
    raise AssertionError("Question coverage validation failed")
if invalid_normal_qpc_ids or mixed_output_qids or duplicate_pairs:
    raise AssertionError("Passage/no-answer structural validation failed")
if missing_or_nonfinite_scores or rank_continuity_violations:
    raise AssertionError("Score/rank structural validation failed")
if score_order_violations or tie_order_violations:
    raise AssertionError("Deterministic ranking validation failed")
if not validation["minus_one_set_matches_zero_candidates"]:
    raise AssertionError("No-answer outputs do not match zero-candidate qids")

STRUCTURAL_VALIDATION_FILE.write_text(
    json.dumps(validation, ensure_ascii=False, allow_nan=False, indent=2),
    encoding="utf-8",
)

output_files = [
    NORMALIZATION_PARAMETERS_FILE,
    SPLIT_MAPPING_FILE,
    OUT_OF_RANGE_FILE,
    LEM_FEATURES_FILE,
    ALL_RESULTS_FILE,
    TOP100_RESULTS_FILE,
    OLD_NEW_COMPARISON_FILE,
    STRUCTURAL_VALIDATION_FILE,
]

print("=" * 70)
print("SPLIT-SAFE RALSR OUTPUTS EXPORTED AND VALIDATED")
print("=" * 70)
for output_file in output_files:
    digest = hashlib.sha256(output_file.read_bytes()).hexdigest().upper()
    print(f"{output_file.name}: SHA-256 {digest}")
print(json.dumps(validation, ensure_ascii=False, indent=2))
