"""Public code snapshot mechanically extracted from the authoritative notebook.

Notebook cell order is preserved. Configure external paths as described in
docs/REPRODUCIBILITY.md. No scientific algorithm or parameter was changed.
"""


# %% [notebook cell 5]
# ============================================================
# Cell 1: Imports, Portable Paths, and Preparation Check
# ============================================================

from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    """Locate the thesis workspace without embedding a personal machine path."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "02_Methodology_Evidence").is_dir():
            return candidate
    raise FileNotFoundError("Could not locate the thesis workspace root.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


PROJECT_ROOT = find_project_root(Path.cwd())
METHODOLOGY_DIR = PROJECT_ROOT / "02_Methodology_Evidence"
CSR_DIR = METHODOLOGY_DIR / "CSR"

CSR_INPUT_DIR = CSR_DIR / "Corrected_Outputs" / "CSR_Authoritative_Lineage"
CORRECTED_OUTPUT_DIR = CSR_DIR / "Corrected_Outputs" / "Notebook_10D_CSR_Retrieval"
DENSE_CACHE_DIR = CORRECTED_OUTPUT_DIR / "Dense_CSR_Cache"

CORRECTED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DENSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

QUERY_CSR_FILE = CSR_INPUT_DIR / "QuranQA_Query_CSR_Corrected.csv"
PASSAGE_CSR_FILE = CSR_INPUT_DIR / "QuranQA_Passage_CSR_Corrected.csv"
OFFICIAL_QPC_FILE = (
    METHODOLOGY_DIR / "Data and Resources" / "QuranQA" /
    "QQA23_TaskA_QPC_v1.1.tsv"
)

EXPECTED_QUERY_CSR_SHA256 = "6972313CF9A0A8897DA4DD0DF64DB8132A7B68ED766F2C199072ECCBF3D10987"
EXPECTED_PASSAGE_CSR_SHA256 = "7D839476B922628E9123A538DBBDCFF6227D7A762B7866E4A151A8E11964A745"

# Portable default plus an execution-time override for an externally stored local model.
MODEL_PATH = Path(
    os.environ.get(
        "QURANQA_E5_MODEL",
        str(PROJECT_ROOT / "04_Code" / "Models" / "multilingual-e5-base"),
    )
).expanduser().resolve()

required_files = [QUERY_CSR_FILE, PASSAGE_CSR_FILE, OFFICIAL_QPC_FILE]
missing = [path for path in required_files if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Missing required file(s): {missing}")
if not MODEL_PATH.is_dir():
    raise FileNotFoundError(
        "Local multilingual-e5-base directory not found. Set QURANQA_E5_MODEL "
        "to the verified local model directory."
    )

query_input_sha256 = sha256_file(QUERY_CSR_FILE)
passage_input_sha256 = sha256_file(PASSAGE_CSR_FILE)
assert query_input_sha256 == EXPECTED_QUERY_CSR_SHA256
assert passage_input_sha256 == EXPECTED_PASSAGE_CSR_SHA256

print("NOTEBOOK 10D CONTROLLED PREPARATION")
print(f"Project root: {PROJECT_ROOT}")
print(f"Query CSR: {QUERY_CSR_FILE}")
print(f"Passage CSR: {PASSAGE_CSR_FILE}")
print(f"Output: {CORRECTED_OUTPUT_DIR}")
print(f"Dense model: {MODEL_PATH}")
print("Qrels loaded: False")

# %% [notebook cell 7]
# ============================================================
# Cell 2: Load and Validate Authoritative CSR Inputs
# ============================================================

query_csr_df = pd.read_csv(QUERY_CSR_FILE, encoding="utf-8-sig")
passage_csr_df = pd.read_csv(PASSAGE_CSR_FILE, encoding="utf-8-sig")
official_qpc_df = pd.read_csv(
    OFFICIAL_QPC_FILE,
    sep="\t",
    header=None,
    names=["Passage_ID", "Official_Passage_Text"],
    dtype=str,
    keep_default_na=False,
)

required_query_columns = {"Question_ID", "Question", "CSR_Text"}
required_passage_columns = {"Passage_ID", "Passage_Text", "CSR_Text"}
assert not (required_query_columns - set(query_csr_df.columns))
assert not (required_passage_columns - set(passage_csr_df.columns))

query_csr_df["Question_ID"] = pd.to_numeric(
    query_csr_df["Question_ID"], errors="raise"
).astype(int)
passage_csr_df["Passage_ID"] = passage_csr_df["Passage_ID"].astype(str)
official_qpc_df["Passage_ID"] = official_qpc_df["Passage_ID"].astype(str)

query_csr_df = query_csr_df.sort_values("Question_ID", kind="mergesort").reset_index(drop=True)
passage_csr_df = passage_csr_df.reset_index(drop=True)

assert len(query_csr_df) == 251
assert query_csr_df["Question_ID"].nunique() == 251
assert query_csr_df["Question_ID"].notna().all()
assert query_csr_df["CSR_Text"].fillna("").str.len().gt(0).all()

assert len(passage_csr_df) == 1266
assert passage_csr_df["Passage_ID"].nunique() == 1266
assert passage_csr_df["CSR_Text"].fillna("").str.len().gt(0).all()
assert set(passage_csr_df["Passage_ID"]) == set(official_qpc_df["Passage_ID"])

authoritative_question_ids = query_csr_df["Question_ID"].tolist()
authoritative_question_id_set = set(authoritative_question_ids)
official_passage_id_set = set(official_qpc_df["Passage_ID"])

assert 504 in authoritative_question_id_set

print("AUTHORITATIVE CSR INPUTS VALIDATED")
print(f"Questions: {len(authoritative_question_ids):,}")
print(f"Passages: {len(passage_csr_df):,}")
print("qid 504 present: True")
print("Retrieval eligibility source: corrected Query CSR only")

# %% [notebook cell 9]
# ============================================================
# Cell 3: Prepare Exact CSR Text for Retrieval
# ============================================================

query_retrieval_df = query_csr_df.copy()
passage_retrieval_df = passage_csr_df.copy()

query_retrieval_df["CSR_Tokens"] = (
    query_retrieval_df["CSR_Text"].fillna("").astype(str).str.split()
)
passage_retrieval_df["CSR_Tokens"] = (
    passage_retrieval_df["CSR_Text"].fillna("").astype(str).str.split()
)

assert query_retrieval_df["CSR_Text"].equals(query_csr_df["CSR_Text"])
assert passage_retrieval_df["CSR_Text"].equals(passage_csr_df["CSR_Text"])

passage_ids = passage_retrieval_df["Passage_ID"].astype(str).to_numpy()
TOP_K = 100

print("RETRIEVAL INPUTS PREPARED")
print(f"Queries: {len(query_retrieval_df):,}")
print(f"Passages: {len(passage_retrieval_df):,}")
print(f"Top-K: {TOP_K}")
print("Additional linguistic preprocessing: None")

# %% [notebook cell 11]
# ============================================================
# Cell 4: Run or Reuse Validated BM25 + CSR Top-100 Retrieval
# ============================================================

from rank_bm25 import BM25Okapi

BM25_CSR_TOP100_FILE = CORRECTED_OUTPUT_DIR / "BM25_CSR_Top100_Corrected.csv"
EXPECTED_BM25_CSR_SHA256 = "DF1AFAD0838436DFA8C74CFC57DE2440470AF503FDB5B22FD54F6CF506E95811"

if (
    BM25_CSR_TOP100_FILE.is_file()
    and sha256_file(BM25_CSR_TOP100_FILE) == EXPECTED_BM25_CSR_SHA256
):
    bm25_csr_top100_df = pd.read_csv(BM25_CSR_TOP100_FILE, encoding="utf-8-sig")
    print("Loaded the already validated corrected BM25+CSR run; retrieval was not repeated.")
    bm25_parameters = {"k1": 1.5, "b": 0.75, "epsilon": 0.25}
else:
    bm25_csr_model = BM25Okapi(passage_retrieval_df["CSR_Tokens"].tolist())
    bm25_parameters = {
        "k1": bm25_csr_model.k1,
        "b": bm25_csr_model.b,
        "epsilon": bm25_csr_model.epsilon,
    }
    bm25_csr_records = []

    for query_row in query_retrieval_df.itertuples(index=False):
        scores = np.asarray(bm25_csr_model.get_scores(query_row.CSR_Tokens), dtype=np.float64)
        # Primary key: descending score. Secondary key: ascending official passage ID.
        top_indices = np.lexsort((passage_ids, -scores))[:TOP_K]

        for rank_position, passage_index in enumerate(top_indices, start=1):
            bm25_csr_records.append(
                {
                    "System": "BM25+CSR",
                    "Question_ID": int(query_row.Question_ID),
                    "Question": query_row.Question,
                    "CSR_Query_Text": query_row.CSR_Text,
                    "Passage_ID": passage_ids[passage_index],
                    "BM25_CSR_Score": float(scores[passage_index]),
                    "Rank": rank_position,
                }
            )

    bm25_csr_top100_df = pd.DataFrame(bm25_csr_records)
    bm25_csr_top100_df.to_csv(
        BM25_CSR_TOP100_FILE, index=False, encoding="utf-8-sig"
    )

print("BM25 + CSR RUN READY")
print(f"Questions: {bm25_csr_top100_df['Question_ID'].nunique():,}")
print(f"Rows: {len(bm25_csr_top100_df):,}")
print(f"BM25 parameters: {bm25_parameters}")
print(BM25_CSR_TOP100_FILE)

# %% [notebook cell 13]
# ============================================================
# Cell 5: Load Deterministic Dense E5 Model
# ============================================================

import torch
import sentence_transformers
import transformers
from sentence_transformers import SentenceTransformer

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True, warn_only=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dense_csr_model = SentenceTransformer(str(MODEL_PATH), device=DEVICE)
dense_csr_model.eval()

model_config_hash = sha256_file(MODEL_PATH / "config.json")
modules_hash = sha256_file(MODEL_PATH / "modules.json")
pooling_hash = sha256_file(MODEL_PATH / "1_Pooling" / "config.json")

print("DENSE E5 MODEL LOADED")
print(f"Model: multilingual-e5-base")
print(f"Device: {DEVICE}")
print(f"SentenceTransformers: {sentence_transformers.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Torch: {torch.__version__}")
print(f"Pooling: configured mean pooling; normalized output")

# %% [notebook cell 15]
# ============================================================
# Cell 6: Generate, Resume, or Load Input-Linked Passage Embeddings
# ============================================================

CSR_PASSAGE_EMBEDDINGS_FILE = DENSE_CACHE_DIR / "CSR_Passage_Embeddings_Corrected.npy"
CSR_PASSAGE_METADATA_FILE = DENSE_CACHE_DIR / "CSR_Passage_Metadata_Corrected.csv"
CSR_QUERY_EMBEDDINGS_FILE = DENSE_CACHE_DIR / "CSR_Query_Embeddings_Corrected.npy"
CSR_QUERY_METADATA_FILE = DENSE_CACHE_DIR / "CSR_Query_Metadata_Corrected.csv"
CSR_CACHE_MANIFEST_FILE = DENSE_CACHE_DIR / "CSR_Embedding_Cache_Manifest.json"
PASSAGE_BATCH_DIR = DENSE_CACHE_DIR / "Passage_Batches"
QUERY_BATCH_DIR = DENSE_CACHE_DIR / "Query_Batches"
PASSAGE_BATCH_DIR.mkdir(parents=True, exist_ok=True)
QUERY_BATCH_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
expected_cache_identity = {
    "query_csr_sha256": query_input_sha256,
    "passage_csr_sha256": passage_input_sha256,
    "model_config_sha256": model_config_hash,
    "model_modules_sha256": modules_hash,
    "model_pooling_sha256": pooling_hash,
    "model_name": "multilingual-e5-base",
    "passage_prefix": "passage: ",
    "query_prefix": "query: ",
    "batch_size": BATCH_SIZE,
    "normalize_embeddings": True,
}

cache_manifest = None
if CSR_CACHE_MANIFEST_FILE.is_file():
    cache_manifest = json.loads(CSR_CACHE_MANIFEST_FILE.read_text(encoding="utf-8"))
    if cache_manifest != expected_cache_identity:
        raise RuntimeError(
            "Existing corrected dense cache does not match the authoritative inputs/model. "
            "It was not overwritten."
        )
else:
    CSR_CACHE_MANIFEST_FILE.write_text(
        json.dumps(expected_cache_identity, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cache_manifest = expected_cache_identity


def encode_resumable(texts, batch_dir: Path, prefix: str) -> np.ndarray:
    """Encode in the established batch size and preserve completed batches."""
    arrays = []
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        batch_file = batch_dir / f"{prefix}_{start:04d}_{end:04d}.npy"
        if batch_file.is_file():
            batch_array = np.load(batch_file)
            if batch_array.shape[0] != end - start:
                raise RuntimeError(f"Invalid cached batch shape: {batch_file}")
        else:
            batch_array = dense_csr_model.encode(
                texts[start:end],
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            np.save(batch_file, batch_array)
        arrays.append(batch_array)
        print(f"{prefix}: {end:,}/{len(texts):,}")
    return np.concatenate(arrays, axis=0)


passage_cache_valid = (
    cache_manifest == expected_cache_identity
    and CSR_PASSAGE_EMBEDDINGS_FILE.is_file()
    and CSR_PASSAGE_METADATA_FILE.is_file()
)

if passage_cache_valid:
    csr_passage_embeddings = np.load(CSR_PASSAGE_EMBEDDINGS_FILE)
    csr_passage_metadata_df = pd.read_csv(CSR_PASSAGE_METADATA_FILE, encoding="utf-8-sig")
    print("Loaded verified corrected passage embedding cache.")
else:
    csr_passage_texts = (
        "passage: " + passage_retrieval_df["CSR_Text"].fillna("").astype(str)
    ).tolist()
    csr_passage_embeddings = encode_resumable(
        csr_passage_texts, PASSAGE_BATCH_DIR, "passage"
    )
    csr_passage_metadata_df = passage_retrieval_df[
        ["Passage_ID", "Passage_Text", "CSR_Text"]
    ].copy()
    np.save(CSR_PASSAGE_EMBEDDINGS_FILE, csr_passage_embeddings)
    csr_passage_metadata_df.to_csv(
        CSR_PASSAGE_METADATA_FILE, index=False, encoding="utf-8-sig"
    )

assert csr_passage_embeddings.shape[0] == 1266
assert np.allclose(np.linalg.norm(csr_passage_embeddings, axis=1), 1.0, atol=1e-5)
assert csr_passage_metadata_df["Passage_ID"].astype(str).tolist() == passage_ids.tolist()
print(f"Passage embedding shape: {csr_passage_embeddings.shape}")

# %% [notebook cell 17]
# ============================================================
# Cell 7: Generate, Resume, or Load Input-Linked Query Embeddings
# ============================================================

query_cache_valid = (
    cache_manifest == expected_cache_identity
    and CSR_QUERY_EMBEDDINGS_FILE.is_file()
    and CSR_QUERY_METADATA_FILE.is_file()
)

if query_cache_valid:
    csr_query_embeddings = np.load(CSR_QUERY_EMBEDDINGS_FILE)
    csr_query_metadata_df = pd.read_csv(CSR_QUERY_METADATA_FILE, encoding="utf-8-sig")
    print("Loaded verified corrected query embedding cache.")
else:
    csr_query_texts = (
        "query: " + query_retrieval_df["CSR_Text"].fillna("").astype(str)
    ).tolist()
    csr_query_embeddings = encode_resumable(
        csr_query_texts, QUERY_BATCH_DIR, "query"
    )
    csr_query_metadata_df = query_retrieval_df[
        ["Question_ID", "Question", "CSR_Text"]
    ].copy()
    np.save(CSR_QUERY_EMBEDDINGS_FILE, csr_query_embeddings)
    csr_query_metadata_df.to_csv(
        CSR_QUERY_METADATA_FILE, index=False, encoding="utf-8-sig"
    )

assert csr_query_embeddings.shape[0] == 251
assert np.allclose(np.linalg.norm(csr_query_embeddings, axis=1), 1.0, atol=1e-5)
assert csr_query_metadata_df["Question_ID"].astype(int).tolist() == authoritative_question_ids
print(f"Query embedding shape: {csr_query_embeddings.shape}")
print(CSR_CACHE_MANIFEST_FILE)

# %% [notebook cell 19]
# ============================================================
# Cell 8: Run Deterministic Dense + CSR Top-100 Retrieval
# ============================================================

similarity_matrix = np.matmul(csr_query_embeddings, csr_passage_embeddings.T)
dense_csr_records = []

for query_index, query_row in csr_query_metadata_df.iterrows():
    scores = np.asarray(similarity_matrix[query_index], dtype=np.float64)
    top_indices = np.lexsort((passage_ids, -scores))[:TOP_K]

    for rank_position, passage_index in enumerate(top_indices, start=1):
        dense_csr_records.append(
            {
                "System": "DenseE5+CSR",
                "Question_ID": int(query_row["Question_ID"]),
                "Question": query_row["Question"],
                "CSR_Query_Text": query_row["CSR_Text"],
                "Passage_ID": passage_ids[passage_index],
                "Dense_CSR_Score": float(scores[passage_index]),
                "Rank": rank_position,
            }
        )

dense_csr_top100_df = pd.DataFrame(dense_csr_records)
DENSE_CSR_TOP100_FILE = CORRECTED_OUTPUT_DIR / "Dense_CSR_Top100_Corrected.csv"
dense_csr_top100_df.to_csv(
    DENSE_CSR_TOP100_FILE, index=False, encoding="utf-8-sig"
)

print("DENSE + CSR RETRIEVAL COMPLETE")
print(f"Questions: {dense_csr_top100_df['Question_ID'].nunique():,}")
print(f"Rows: {len(dense_csr_top100_df):,}")
print(DENSE_CSR_TOP100_FILE)

# %% [notebook cell 21]
# Historical local evaluation/reporting code is intentionally disabled in the
# controlled retrieval-regeneration notebook. The original code is preserved
# byte-for-byte in the pre-correction Notebook 10D backup.

# %% [notebook cell 23]
# Historical local evaluation/reporting code is intentionally disabled in the
# controlled retrieval-regeneration notebook. The original code is preserved
# byte-for-byte in the pre-correction Notebook 10D backup.

# %% [notebook cell 25]
# Historical local evaluation/reporting code is intentionally disabled in the
# controlled retrieval-regeneration notebook. The original code is preserved
# byte-for-byte in the pre-correction Notebook 10D backup.

# %% [notebook cell 27]
# Historical local evaluation/reporting code is intentionally disabled in the
# controlled retrieval-regeneration notebook. The original code is preserved
# byte-for-byte in the pre-correction Notebook 10D backup.

# %% [notebook cell 29]
# Historical local evaluation/reporting code is intentionally disabled in the
# controlled retrieval-regeneration notebook. The original code is preserved
# byte-for-byte in the pre-correction Notebook 10D backup.

# %% [notebook cell 31]
# Historical local evaluation/reporting code is intentionally disabled in the
# controlled retrieval-regeneration notebook. The original code is preserved
# byte-for-byte in the pre-correction Notebook 10D backup.

# %% [notebook cell 33]
# Historical local evaluation/reporting code is intentionally disabled in the
# controlled retrieval-regeneration notebook. The original code is preserved
# byte-for-byte in the pre-correction Notebook 10D backup.

# %% [notebook cell 35]
# ============================================================
# Final Cell: Structural and Qrel-Independence Validation
# ============================================================


def validate_top100(df: pd.DataFrame, score_column: str, system_name: str) -> dict:
    qids = set(df["Question_ID"].astype(int))
    passage_values = df["Passage_ID"].astype(str)
    scores = pd.to_numeric(df[score_column], errors="coerce")

    depth = df.groupby("Question_ID", sort=False).size()
    rank_lists = df.groupby("Question_ID", sort=False)["Rank"].apply(list)
    missing_qids = sorted(authoritative_question_id_set - qids)
    unexpected_qids = sorted(qids - authoritative_question_id_set)
    invalid_passage_ids = sorted(set(passage_values) - official_passage_id_set)
    duplicate_pairs = int(df.duplicated(["Question_ID", "Passage_ID"]).sum())
    malformed_scores = int(scores.isna().sum())
    nonfinite_scores = int((~np.isfinite(scores.to_numpy(dtype=float))).sum())
    rank_violations = int(sum(ranks != list(range(1, TOP_K + 1)) for ranks in rank_lists))

    order_violations = 0
    tie_order_violations = 0
    for _, group in df.groupby("Question_ID", sort=False):
        group = group.sort_values("Rank", kind="mergesort")
        group_scores = group[score_column].to_numpy(dtype=float)
        group_ids = group["Passage_ID"].astype(str).tolist()
        if np.any(group_scores[:-1] < group_scores[1:]):
            order_violations += 1
        for i in range(len(group_scores) - 1):
            if group_scores[i] == group_scores[i + 1] and group_ids[i] > group_ids[i + 1]:
                tie_order_violations += 1

    result = {
        "system": system_name,
        "rows": int(len(df)),
        "question_count": int(len(qids)),
        "missing_question_ids": missing_qids,
        "unexpected_question_ids": unexpected_qids,
        "qid_504_present": 504 in qids,
        "minimum_depth": int(depth.min()),
        "maximum_depth": int(depth.max()),
        "invalid_passage_ids": invalid_passage_ids,
        "duplicate_question_passage_pairs": duplicate_pairs,
        "malformed_scores": malformed_scores,
        "nonfinite_scores": nonfinite_scores,
        "rank_continuity_violations": rank_violations,
        "descending_score_order_violations": order_violations,
        "tie_order_violations": tie_order_violations,
    }

    assert result["rows"] == 25100
    assert result["question_count"] == 251
    assert not missing_qids and not unexpected_qids
    assert result["qid_504_present"]
    assert result["minimum_depth"] == TOP_K == result["maximum_depth"]
    assert not invalid_passage_ids
    assert duplicate_pairs == 0
    assert malformed_scores == 0 and nonfinite_scores == 0
    assert rank_violations == 0 and order_violations == 0 and tie_order_violations == 0
    return result


# Reload exported files so validation covers serialized artifacts.
bm25_reload_df = pd.read_csv(BM25_CSR_TOP100_FILE, encoding="utf-8-sig")
dense_reload_df = pd.read_csv(DENSE_CSR_TOP100_FILE, encoding="utf-8-sig")

validation = {
    "query_csr_question_count": len(authoritative_question_id_set),
    "qrels_loaded_or_used": False,
    "retrieval_eligibility_source": "QuranQA_Query_CSR_Corrected.csv",
    "bm25": validate_top100(
        bm25_reload_df, "BM25_CSR_Score", "BM25+CSR"
    ),
    "dense": validate_top100(
        dense_reload_df, "Dense_CSR_Score", "DenseE5+CSR"
    ),
    "bm25_output_sha256": sha256_file(BM25_CSR_TOP100_FILE),
    "dense_output_sha256": sha256_file(DENSE_CSR_TOP100_FILE),
}

VALIDATION_FILE = CORRECTED_OUTPUT_DIR / "Notebook_10D_Structural_Validation.json"
VALIDATION_FILE.write_text(
    json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(json.dumps(validation, ensure_ascii=False, indent=2))
print("Corrected retrieval generation uses no qrel information.")
