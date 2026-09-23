#!/usr/bin/env python3
"""Authoritative Step-10 zero-shot CrossEncoder reranking and evaluation.

Phases are deliberately separated so qrels cannot enter prediction construction:
  prepare  verify frozen inputs and write the predeclared configuration;
  infer    score/rerank the frozen fixed-RALSR candidates and freeze official runs;
  evaluate load qrels only after prediction artifacts are frozen, then evaluate/analyze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(r"${THESIS_WORKSPACE}")
CONTROL = Path(r"${THESIS_CONTROL_ROOT}")
OUTPUT = CONTROL / "CrossEncoder_Final_Evaluation"
SCRIPTS = OUTPUT / "Scripts"
CHECKPOINTS = OUTPUT / "Inference_Checkpoints"
VALIDATION_LOGS = OUTPUT / "Validation_Logs"
EVALUATOR_LOGS = OUTPUT / "Evaluator_Logs"
SCORING_RUNS = OUTPUT / "Scoring_Runs"
FIGURES = OUTPUT / "Figures"
TABLES = OUTPUT / "Tables"

DATA = WORKSPACE / "02_Methodology_Evidence" / "Data and Resources" / "QuranQA"
RALSR = WORKSPACE / "02_Methodology_Evidence" / "RALSR" / "Corrected_Outputs" / "RALSR_Split_Safe_Normalization_No_Answer" / "RALSR_Top100_Split_Safe_With_No_Answer.csv"
QPC = DATA / "QQA23_TaskA_QPC_v1.1.tsv"
QUESTIONS = {s: DATA / f"QQA23_TaskA_ayatec_v1.2_{s}.tsv" for s in ("train", "dev", "test")}
QRELS = {s: DATA / f"QQA23_TaskA_ayatec_v1.2_qrels_{s}.gold" for s in ("train", "dev", "test")}
ANSWERABLE_QRELS = {s: CONTROL / "No_Answer_Audit" / f"qrels_{s}_answerable_only.gold" for s in ("train", "dev", "test")}

MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
MODEL_DIR = Path(r"${THESIS_MODEL_ROOT}\mmarco-mMiniLMv2-L12-H384-v1")
MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
HISTORICAL_NOTEBOOK = WORKSPACE / "02_Methodology_Evidence" / "Cross-Encoder" / "Notebook 11 – Transformer Cross-Encoder Re-ranking.ipynb"

OFFICIAL_ROOT = CONTROL / "Official_QuranQA_TaskA_Evaluation"
ORGANIZER = OFFICIAL_ROOT / "01_Official_Organizer_Files"
PIPELINE = OFFICIAL_ROOT / "02_Evaluation_Pipeline"
STRICT_VALIDATOR = PIPELINE / "quranqa_taskA_strict_validator.py"
OFFICIAL_CHECKER = ORGANIZER / "QQA23_TaskA_submission_checker.py"
OFFICIAL_SCORER = ORGANIZER / "QQA23_TaskA_eval.py"
INDEPENDENT_EVALUATOR = PIPELINE / "quranqa_taskA_independent_eval.py"
SCORER_ADAPTER = CONTROL / "Final_Core_Evaluation" / "Official_Scorer_Compatibility"

STEP7 = CONTROL / "Final_Core_Evaluation"
STEP8 = CONTROL / "Core_Error_Analysis"
STEP9 = CONTROL / "CSR_Final_Evaluation"

EXPECTED = {
    "baseline": "5EE78669B8BD3E544873CD71946CE9841A87D25F6B82182B899C7DC2CBC2AD15",
    "ralsr": "7B70FBEACBF4E1AB11888CCB1798E9C48349C2977795F871601866A1D04D4D6C",
    "qpc": "0A86C33C465AB6CF9321924D2C03B23ED72F8360134AE92BA4BD4A90C93BE08C",
    "answerable_train": "FD8D351896507C33021EB8B1F5BAE403F09105495E51D7ECDA4AF44B19F93176",
    "answerable_dev": "4D372EEF4655E27662F1F017557B01F003C6A16C57E16074621F1E227E31EAF6",
    "answerable_test": "E799BEC96EA33B7ECDFC0B8285B85B4F16521B6C2FDD7DD02AA7AFE0F608872B",
    "qrels_train": "48E64E24A715BD77B824D9A4C17863C39614FE198A0A44FED23E76A6BEA0CFE6",
    "qrels_dev": "6F74218F1259AA144C795F19D0E4FB99095F927511EE708FA7EE4125F37A3590",
    "qrels_test": "E23E4CF0628EB2FF39562852A5632DE0D948C8F643B5EB8E08B1D8B69CBA0332",
    "model_config": "CC2CFE51AA3FD759D21D21ACF5DFD6994AA67A3C9210636D22E143699D336C77",
    "model_safetensors": "5DAECA2481A76B5976A2BDC32F0A78532B6716DA4F8CD3FF59460EF8D2F359B4",
    "pytorch_model": "1ABC209E54D70BBCB08C1B5111A924FB99C0428F51CAB1659310CCDCAB69DC03",
    "tokenizer_json": "62C24CDC13D4C9952D63718D6C9FA4C287974249E16B7ADE6D5A85E7BBB75626",
    "sentencepiece": "CFC8146ABE2A0488E9E2A0C56DE7952F7C11AB059ECA145A0A727AFCE0DB2865",
}

NO_ANSWER_QIDS = ["102", "108", "110", "137", "141", "143", "212", "234", "235", "252", "258", "536", "613"]
BATCH_SIZE = 16
TAG = "RALSR_CrossEncoder"

INTERNAL_RUN = OUTPUT / "RALSR_CrossEncoder_Top100_With_No_Answer.csv"
SCORE_CHECKPOINT = CHECKPOINTS / "crossencoder_raw_scores.csv"
CONFIG_FREEZE = OUTPUT / "CROSSENCODER_CONFIGURATION_FREEZE.json"
PRE_EVAL_FREEZE = OUTPUT / "CROSSENCODER_PRE_EVALUATION_FREEZE.json"
OFFICIAL_RUNS = {
    "train": OUTPUT / "thesis_RALSRCEtr.tsv",
    "dev": OUTPUT / "thesis_RALSRCEdv.tsv",
    "test": OUTPUT / "thesis_RALSRCEte.tsv",
}


def ensure_dirs() -> None:
    for path in (OUTPUT, SCRIPTS, CHECKPOINTS, VALIDATION_LOGS, EVALUATOR_LOGS, SCORING_RUNS, FIGURES, TABLES):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_hash(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"Identity mismatch: {path}\nexpected={expected}\nactual={actual}")
    return actual


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def read_questions() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    frames = []
    for split, path in QUESTIONS.items():
        df = pd.read_csv(path, sep="\t", header=None, names=["QID", "Question"], dtype=str, encoding="utf-8-sig")
        df["Split"] = split
        frames.append(df)
    questions = pd.concat(frames, ignore_index=True)
    if len(questions) != 251 or questions["QID"].nunique() != 251:
        raise RuntimeError("Official question set is not 251 unique qids")
    return questions, dict(zip(questions.QID, questions.Question)), dict(zip(questions.QID, questions.Split))


def read_qpc() -> pd.DataFrame:
    frame = pd.read_csv(QPC, sep="\t", header=None, names=["Passage_ID", "Passage_Text"], dtype=str, encoding="utf-8-sig")
    if len(frame) != 1266 or frame.Passage_ID.nunique() != 1266:
        raise RuntimeError("QPC is not 1,266 unique passages")
    return frame


def pid_key(pid: str) -> tuple[int, int, int, str]:
    try:
        surah, verses = pid.split(":", 1)
        start, end = verses.split("-", 1)
        return int(surah), int(start), int(end), pid
    except Exception:
        return 10**9, 10**9, 10**9, pid


def load_and_validate_candidates() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    questions, qmap, split_map = read_questions()
    qpc = read_qpc()
    pmap = dict(zip(qpc.Passage_ID, qpc.Passage_Text))
    frame = pd.read_csv(RALSR, dtype={"Question_ID": str, "Passage_ID": str}, encoding="utf-8-sig")
    normal = frame.loc[frame.Passage_ID != "-1"].copy()
    no_answer = frame.loc[frame.Passage_ID == "-1"].copy()
    facts = {
        "rows": int(len(frame)), "qids": int(frame.Question_ID.nunique()),
        "normal_rows": int(len(normal)), "candidate_bearing_qids": int(normal.Question_ID.nunique()),
        "structural_rows": int(len(no_answer)),
        "structural_qids": sorted(no_answer.Question_ID.tolist(), key=int),
    }
    expected = {"rows": 20161, "qids": 251, "normal_rows": 20148, "candidate_bearing_qids": 238, "structural_rows": 13}
    for key, value in expected.items():
        if facts[key] != value:
            raise RuntimeError(f"Candidate structural mismatch {key}: {facts[key]} != {value}")
    if facts["structural_qids"] != NO_ANSWER_QIDS:
        raise RuntimeError("Structural no-answer qid mismatch")
    if set(frame.Question_ID) != set(questions.QID):
        raise RuntimeError("Candidate qid membership differs from official questions")
    if normal.duplicated(["Question_ID", "Passage_ID"]).any():
        raise RuntimeError("Duplicate normal candidate pair")
    missing_q = sorted(set(normal.Question_ID) - set(qmap), key=int)
    missing_p = sorted(set(normal.Passage_ID) - set(pmap))
    if missing_q or missing_p:
        raise RuntimeError(f"Text join failure: qids={missing_q[:5]} passages={missing_p[:5]}")
    normal["Authoritative_Question"] = normal.Question_ID.map(qmap)
    normal["Authoritative_Passage_Text"] = normal.Passage_ID.map(pmap)
    normal["Official_Split"] = normal.Question_ID.map(split_map)
    if normal.Authoritative_Question.isna().any() or normal.Authoritative_Passage_Text.isna().any():
        raise RuntimeError("Missing authoritative text after join")
    if no_answer.Passage_ID.ne("-1").any() or no_answer.groupby("Question_ID").size().ne(1).any():
        raise RuntimeError("Malformed structural no-answer rows")
    return normal.reset_index(drop=True), no_answer.reset_index(drop=True), facts


def model_identity() -> dict:
    files = {
        "config.json": EXPECTED["model_config"],
        "model.safetensors": EXPECTED["model_safetensors"],
        "pytorch_model.bin": EXPECTED["pytorch_model"],
        "tokenizer.json": EXPECTED["tokenizer_json"],
        "sentencepiece.bpe.model": EXPECTED["sentencepiece"],
    }
    evidence = {}
    for name, expected in files.items():
        path = MODEL_DIR / name
        evidence[name] = {"path": str(path), "sha256": verify_hash(path, expected), "bytes": path.stat().st_size}
    metadata = MODEL_DIR / ".cache" / "huggingface" / "download" / "config.json.metadata"
    resolved = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    if resolved != MODEL_REVISION:
        raise RuntimeError(f"Model revision mismatch: {resolved}")
    tokenizer_config = json.loads((MODEL_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
    return {
        "identifier": MODEL_ID, "local_path": str(MODEL_DIR), "resolved_revision": resolved,
        "files": evidence, "tokenizer_class": tokenizer_config.get("tokenizer_class"),
        "tokenizer_model_max_length": tokenizer_config.get("model_max_length"),
        "architecture": config.get("architectures", [None])[0],
        "hidden_size": config.get("hidden_size"), "layers": config.get("num_hidden_layers"),
        "default_activation": config.get("sbert_ce_default_activation_function"),
    }


def phase_prepare() -> dict:
    ensure_dirs()
    verify_hash(WORKSPACE.with_suffix(".rar"), EXPECTED["baseline"])
    verify_hash(RALSR, EXPECTED["ralsr"])
    verify_hash(QPC, EXPECTED["qpc"])
    normal, no_answer, facts = load_and_validate_candidates()
    model = model_identity()
    historical_hash = sha256(HISTORICAL_NOTEBOOK)
    config = {
        "status": "frozen_before_qrel_access",
        "frozen_at_local": datetime.now().astimezone().isoformat(),
        "research_question": "Can a zero-shot CrossEncoder improve the ranking of relevant passages already retrieved by fixed RALSR?",
        "model": model,
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "sentence_transformers": package_version("sentence-transformers"),
            "transformers": package_version("transformers"), "torch": package_version("torch"),
            "numpy": package_version("numpy"), "pandas": package_version("pandas"),
            "device": "cpu", "batch_size": BATCH_SIZE,
        },
        "loading": {
            "class": "sentence_transformers.CrossEncoder", "local_only": True,
            "fine_tuning": False, "quranqa_training": False, "model_selection": False,
            "max_length_behavior": "not overridden; tokenizer model_max_length=512",
            "input_pair_order": ["original_question_text", "original_qpc_passage_text"],
            "score_type": "raw single-logit CrossEncoder score; Identity activation",
        },
        "ranking": {
            "primary": "CrossEncoder_Score descending", "exact_tie": "Passage_ID ascending by numeric surah/start/end",
            "score_fusion": False, "candidate_membership_change": False,
        },
        "candidate_source": {"path": str(RALSR), "sha256": EXPECTED["ralsr"], **facts},
        "text_sources": {"questions": {s: {"path": str(p), "sha256": sha256(p)} for s, p in QUESTIONS.items()}, "qpc": {"path": str(QPC), "sha256": EXPECTED["qpc"]}},
        "structural_no_answer": {"qids": NO_ANSWER_QIDS, "rule": "preserve sole -1 row; no CrossEncoder inference", "score_threshold": None},
        "qid_504": "candidate-bearing and reranked normally; excluded later from scoring because qrel absent",
        "historical_notebook": {
            "path": str(HISTORICAL_NOTEBOOK), "sha256": historical_hash,
            "recovered_valid_details": {"model": MODEL_ID, "class": "CrossEncoder", "pair_order": "question, passage", "batch_size": 16, "device": "cpu", "max_length_explicit": False, "score": "raw model score"},
            "obsolete_details_excluded": {"question_subset": 213, "candidate_branches": ["BM25", "Dense", "historical RALSR"], "tie_rule": "Original_Rank", "evaluation": "historical local evaluator"},
        },
        "qrels_loaded_during_prepare": False,
    }
    if CONFIG_FREEZE.exists() and json.loads(CONFIG_FREEZE.read_text(encoding="utf-8"))["candidate_source"]["sha256"] != EXPECTED["ralsr"]:
        raise RuntimeError("Existing configuration freeze conflicts with authoritative source")
    write_json(CONFIG_FREEZE, config)
    identity_md = OUTPUT / "CROSSENCODER_MODEL_IDENTITY.md"
    lines = ["# CrossEncoder model identity", "", f"- Model: `{MODEL_ID}`", f"- Resolved revision: `{MODEL_REVISION}`", f"- Local path: `{MODEL_DIR}`", f"- Device: CPU", f"- Batch size: {BATCH_SIZE}", f"- Tokenizer: {model['tokenizer_class']}; maximum length 512", f"- Architecture: {model['architecture']}; 12 layers; hidden size 384", "- Score: raw single logit with Identity activation; not a calibrated probability.", "", "## Major files", ""]
    for name, item in model["files"].items():
        lines.append(f"- `{name}` — `{item['sha256']}`")
    identity_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUTPUT / "CROSSENCODER_PREPARATION_VALIDATION.json", {"status": "PASS", "candidate_facts": facts, "missing_question_text": 0, "missing_passage_text": 0, "invalid_passage_ids": 0, "duplicate_qpc_ids": 0, "qrels_loaded": False})
    print(json.dumps({"phase": "prepare", "status": "PASS", "candidate_facts": facts, "config_sha256": sha256(CONFIG_FREEZE)}, indent=2))
    return config


def load_checkpoint(expected_keys: list[tuple[str, str]]) -> list[float]:
    if not SCORE_CHECKPOINT.exists():
        return []
    saved = pd.read_csv(SCORE_CHECKPOINT, dtype={"QID": str, "Passage_ID": str})
    keys = list(zip(saved.QID, saved.Passage_ID))
    if keys != expected_keys[: len(keys)]:
        raise RuntimeError("Inference checkpoint is not a valid prefix of the frozen candidate order")
    scores = saved.CrossEncoder_Score.astype(float).tolist()
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite score in checkpoint")
    return scores


def append_score_checkpoint(rows: list[dict], first_write: bool) -> None:
    pd.DataFrame(rows).to_csv(SCORE_CHECKPOINT, mode="w" if first_write else "a", header=first_write, index=False, encoding="utf-8", lineterminator="\n")


def build_internal_and_runs(normal: pd.DataFrame, no_answer: pd.DataFrame, scores: list[float]) -> dict:
    if len(scores) != len(normal):
        raise RuntimeError("Incomplete CrossEncoder score set")
    scored = pd.DataFrame({
        "QID": normal.Question_ID.astype(str), "Passage_ID": normal.Passage_ID.astype(str),
        "CrossEncoder_Score": np.asarray(scores, dtype=float),
        "Original_RALSR_Rank": normal.Rank.astype(int), "Original_RALSR_Score": normal.RALSR_Score.astype(float),
        "Split": normal.Official_Split.astype(str),
    })
    if not np.isfinite(scored.CrossEncoder_Score).all():
        raise RuntimeError("Non-finite CrossEncoder scores")
    ranked_parts = []
    exact_ties = 0
    for qid, group in scored.groupby("QID", sort=False):
        records = group.to_dict("records")
        records.sort(key=lambda r: (-r["CrossEncoder_Score"], pid_key(r["Passage_ID"])))
        exact_ties += sum(records[i]["CrossEncoder_Score"] == records[i + 1]["CrossEncoder_Score"] for i in range(len(records) - 1))
        for rank, record in enumerate(records, 1):
            record["Rank"] = rank
            record["Output_Type"] = "normal"
            record["Structural_No_Answer"] = False
        ranked_parts.extend(records)
    structural = []
    for row in no_answer.itertuples(index=False):
        structural.append({
            "QID": str(row.Question_ID), "Passage_ID": "-1", "CrossEncoder_Score": 0.0,
            "Original_RALSR_Rank": 1, "Original_RALSR_Score": 0.0, "Split": str(row.Split).lower(),
            "Rank": 1, "Output_Type": "structural_no_answer", "Structural_No_Answer": True,
        })
    internal = pd.DataFrame(ranked_parts + structural)
    internal["_qid"] = internal.QID.astype(int)
    internal = internal.sort_values(["_qid", "Rank"]).drop(columns="_qid").reset_index(drop=True)
    write_csv(INTERNAL_RUN, internal)
    normal_out = internal[internal.Passage_ID != "-1"]
    source_pairs = set(zip(normal.Question_ID.astype(str), normal.Passage_ID.astype(str)))
    output_pairs = set(zip(normal_out.QID, normal_out.Passage_ID))
    if source_pairs != output_pairs:
        raise RuntimeError("Candidate membership mismatch after reranking")
    qids = set(internal.QID)
    if len(internal) != 20161 or len(qids) != 251 or internal.duplicated(["QID", "Passage_ID"]).any():
        raise RuntimeError("Internal structural validation failed")
    rank_bad = 0
    order_bad = 0
    tie_bad = 0
    for qid, group in normal_out.groupby("QID"):
        group = group.sort_values("Rank")
        if group.Rank.tolist() != list(range(1, len(group) + 1)):
            rank_bad += 1
        recs = group.to_dict("records")
        for i in range(len(recs) - 1):
            if recs[i]["CrossEncoder_Score"] < recs[i + 1]["CrossEncoder_Score"]:
                order_bad += 1
            if recs[i]["CrossEncoder_Score"] == recs[i + 1]["CrossEncoder_Score"] and pid_key(recs[i]["Passage_ID"]) > pid_key(recs[i + 1]["Passage_ID"]):
                tie_bad += 1
    if rank_bad or order_bad or tie_bad:
        raise RuntimeError(f"Ranking validation failure: rank={rank_bad} order={order_bad} tie={tie_bad}")

    qsets = {s: set(pd.read_csv(QUESTIONS[s], sep="\t", header=None, dtype=str)[0]) for s in ("train", "dev", "test")}
    validation = {}
    for split, destination in OFFICIAL_RUNS.items():
        subset = internal[internal.QID.isin(qsets[split])].copy()
        lines = []
        for qid, group in subset.groupby("QID", sort=False):
            ordered = group.sort_values("Rank")
            if ordered.iloc[0].Passage_ID == "-1":
                rows = ordered.iloc[:1]
            else:
                rows = ordered[ordered.Rank <= 10]
            for row in rows.itertuples(index=False):
                official_score = 0.0 if row.Passage_ID == "-1" else float(11 - int(row.Rank))
                lines.append([str(row.QID), "Q0", str(row.Passage_ID), str(int(row.Rank)), f"{official_score:.6f}", TAG])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerows(lines)
        strict_json = VALIDATION_LOGS / f"{destination.name}.strict.json"
        strict = subprocess.run([sys.executable, str(STRICT_VALIDATOR), "--run", str(destination), "--questions", str(QUESTIONS[split]), "--qpc", str(QPC), "--json-report", str(strict_json)], text=True, capture_output=True, encoding="utf-8", errors="replace")
        (VALIDATION_LOGS / f"{destination.name}.strict.log").write_text(strict.stdout + "\n[stderr]\n" + strict.stderr, encoding="utf-8")
        if strict.returncode != 0:
            raise RuntimeError(f"Strict validator failed: {destination}")
        checker = subprocess.run([sys.executable, str(OFFICIAL_CHECKER), "--model-prediction", str(destination)], text=True, capture_output=True, encoding="utf-8", errors="replace")
        (VALIDATION_LOGS / f"{destination.name}.organizer_checker.log").write_text(checker.stdout + "\n[stderr]\n" + checker.stderr, encoding="utf-8")
        if checker.returncode != 0 or "Format check: Passed" not in checker.stdout:
            raise RuntimeError(f"Organizer checker failed: {destination}")
        parsed = pd.read_csv(destination, sep="\t", header=None, names=["QID", "Q0", "Passage_ID", "Rank", "Score", "Tag"], dtype={"QID": str, "Passage_ID": str})
        mismatch = 0
        for qid, group in parsed.groupby("QID"):
            official_pids = group.sort_values("Rank").Passage_ID.tolist()
            expected = subset[subset.QID == qid].sort_values("Rank")
            expected_pids = (["-1"] if expected.iloc[0].Passage_ID == "-1" else expected[expected.Rank <= 10].Passage_ID.tolist())
            mismatch += int(official_pids != expected_pids)
        if mismatch:
            raise RuntimeError(f"Official ranking mismatch: {split}")
        validation[split] = {"path": str(destination), "sha256": sha256(destination), "rows": len(parsed), "qids": parsed.QID.nunique(), "qid_504_present": "504" in set(parsed.QID), "strict_validator": "PASS", "organizer_checker": "PASS", "ranking_mismatches": mismatch}
    facts = {
        "rows": len(internal), "qids": internal.QID.nunique(), "candidate_bearing_qids": normal_out.QID.nunique(),
        "structural_no_answer_qids": sorted(internal[internal.Passage_ID == "-1"].QID.tolist(), key=int),
        "candidate_membership_mismatches": len(source_pairs.symmetric_difference(output_pairs)),
        "added_candidates": len(output_pairs - source_pairs), "removed_candidates": len(source_pairs - output_pairs),
        "duplicates": int(internal.duplicated(["QID", "Passage_ID"]).sum()), "nonfinite_scores": int((~np.isfinite(normal_out.CrossEncoder_Score)).sum()),
        "rank_violations": rank_bad, "score_order_violations": order_bad, "tie_order_violations": tie_bad, "exact_adjacent_score_ties": exact_ties,
        "score_min": float(normal_out.CrossEncoder_Score.min()), "score_max": float(normal_out.CrossEncoder_Score.max()),
        "score_mean": float(normal_out.CrossEncoder_Score.mean()), "score_median": float(normal_out.CrossEncoder_Score.median()),
    }
    write_json(OUTPUT / "CROSSENCODER_INTERNAL_RUN_VALIDATION.json", facts)
    write_json(OUTPUT / "CROSSENCODER_OFFICIAL_RUN_VALIDATION.json", validation)
    return {"internal": facts, "official": validation}


def phase_infer() -> dict:
    ensure_dirs()
    if not CONFIG_FREEZE.exists():
        raise RuntimeError("Run prepare before infer")
    verify_hash(RALSR, EXPECTED["ralsr"])
    verify_hash(QPC, EXPECTED["qpc"])
    model_identity()
    normal, no_answer, facts = load_and_validate_candidates()
    keys = list(zip(normal.Question_ID.astype(str), normal.Passage_ID.astype(str)))
    scores = load_checkpoint(keys)
    if len(scores) < len(normal):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from sentence_transformers import CrossEncoder
        torch.manual_seed(0)
        model = CrossEncoder(str(MODEL_DIR), device="cpu")
        start = len(scores)
        total_batches = math.ceil((len(normal) - start) / BATCH_SIZE)
        print(f"Inference resume row {start:,}/{len(normal):,}; remaining batches={total_batches:,}", flush=True)
        for batch_number, begin in enumerate(range(start, len(normal), BATCH_SIZE), 1):
            end = min(begin + BATCH_SIZE, len(normal))
            pairs = list(zip(normal.iloc[begin:end].Authoritative_Question.astype(str), normal.iloc[begin:end].Authoritative_Passage_Text.astype(str)))
            batch_scores = np.asarray(model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False), dtype=float).reshape(-1)
            if len(batch_scores) != end - begin or not np.isfinite(batch_scores).all():
                raise RuntimeError("Invalid inference batch")
            rows = [{"Source_Row_Index": i, "QID": keys[i][0], "Passage_ID": keys[i][1], "CrossEncoder_Score": format(float(batch_scores[i - begin]), ".17g")} for i in range(begin, end)]
            append_score_checkpoint(rows, first_write=(begin == 0 and not SCORE_CHECKPOINT.exists()))
            scores.extend(batch_scores.astype(float).tolist())
            if batch_number == 1 or batch_number % 25 == 0 or end == len(normal):
                print(f"INFERENCE_PROGRESS rows={end}/{len(normal)} batches={batch_number}/{total_batches}", flush=True)
    validation = build_internal_and_runs(normal, no_answer, scores)
    prefreeze = {
        "status": "predictions_frozen_before_qrel_access", "frozen_at_local": datetime.now().astimezone().isoformat(),
        "qrels_loaded_during_generation": False, "configuration": {"path": str(CONFIG_FREEZE), "sha256": sha256(CONFIG_FREEZE)},
        "candidate_source": {"path": str(RALSR), "sha256": sha256(RALSR)},
        "raw_score_checkpoint": {"path": str(SCORE_CHECKPOINT), "sha256": sha256(SCORE_CHECKPOINT), "rows": len(scores)},
        "internal_run": {"path": str(INTERNAL_RUN), "sha256": sha256(INTERNAL_RUN)},
        "official_runs": validation["official"], "internal_validation": validation["internal"],
        "script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    }
    write_json(PRE_EVAL_FREEZE, prefreeze)
    print(json.dumps({"phase": "infer", "status": "PASS", "internal_sha256": sha256(INTERNAL_RUN), "official_runs": validation["official"]}, indent=2))
    return prefreeze


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    result = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row:
                result[str(row[0])][str(row[2])] = int(row[3])
    return dict(result)


def run_command(command: list[str], log_path: Path, env=None) -> subprocess.CompletedProcess:
    completed = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace", env=env, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")
    return completed


def filter_run(source: Path, destination: Path, qids: set[str]) -> None:
    lines = [line for line in source.read_text(encoding="utf-8-sig").splitlines() if line.split("\t", 1)[0] in qids]
    seen = {line.split("\t", 1)[0] for line in lines}
    if seen != qids:
        raise RuntimeError(f"Filtered run qid mismatch: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def score_official(run: Path, qrels: Path, prefix: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SCORER_ADAPTER), str(ORGANIZER), env.get("PYTHONPATH", "")])
    metrics = prefix.with_suffix(".metrics.tsv")
    completed = run_command([sys.executable, str(PIPELINE / "run_official_taskA_evaluation.py"), "--run", str(run), "--qrels", str(qrels), "--official-dir", str(ORGANIZER), "--output", str(metrics)], prefix.with_suffix(".scorer.log"), env=env)
    if "Format check: Passed" not in completed.stdout:
        raise RuntimeError("Official scorer did not confirm format")
    frame = pd.read_csv(metrics, sep="\t")
    return {"MAP_at_10": float(frame.loc[0, "map_cut_10"]), "MRR_at_10": float(frame.loc[0, "recip_rank"]), "metrics_path": str(metrics)}


def score_independent(run: Path, qrels: Path, prefix: Path) -> dict:
    output = prefix.with_suffix(".json")
    run_command([sys.executable, str(INDEPENDENT_EVALUATOR), "--run", str(run), "--qrels", str(qrels), "--output", str(output)], prefix.with_suffix(".log"))
    return json.loads(output.read_text(encoding="utf-8"))


def load_frozen_comparators() -> pd.DataFrame:
    step7 = pd.read_csv(STEP7 / "CORE_TEST_RESULTS_MAIN_TABLE.csv")
    step9 = pd.read_csv(STEP9 / "CSR_TEST_COMPARISON.csv")
    step7 = step7.rename(columns={
        "Official_Judged_MAP_at_10": "Official_MAP_at_10",
        "Official_Judged_MRR_at_10": "Official_MRR_at_10",
        "Answerable_Only_MAP_at_10": "Answerable_MAP_at_10",
        "Answerable_Only_MRR_at_10": "Answerable_MRR_at_10",
    })
    name_col = "System"
    required = ["BM25", "Dense E5", "Fixed RALSR", "Tuned RALSR"]
    core = step7[step7[name_col].isin(required)].copy()
    csr = step9[step9[name_col].eq("Dense E5 + CSR")].copy()
    columns = [name_col, "Official_MAP_at_10", "Official_MRR_at_10", "Answerable_MAP_at_10", "Answerable_MRR_at_10"]
    return pd.concat([core[columns], csr[columns]], ignore_index=True)


def tex_escape(value: str) -> str:
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def phase_evaluate() -> dict:
    ensure_dirs()
    if not PRE_EVAL_FREEZE.exists():
        raise RuntimeError("Predictions must be frozen before evaluation")
    prefreeze = json.loads(PRE_EVAL_FREEZE.read_text(encoding="utf-8"))
    if prefreeze["internal_run"]["sha256"] != sha256(INTERNAL_RUN):
        raise RuntimeError("Frozen internal prediction identity changed")
    for split in ("train", "dev", "test"):
        verify_hash(QRELS[split], EXPECTED[f"qrels_{split}"])
        verify_hash(ANSWERABLE_QRELS[split], EXPECTED[f"answerable_{split}"])

    results = []
    independent_records = {}
    max_repeat = 0.0
    max_independent = 0.0
    for population, qrel_paths in (("Official judged", QRELS), ("Answerable-only", ANSWERABLE_QRELS)):
        for split in ("train", "dev", "test"):
            qrel_map = read_qrels(qrel_paths[split])
            run = OFFICIAL_RUNS[split]
            filtered = SCORING_RUNS / population.replace(" ", "_") / split / run.name
            filter_run(run, filtered, set(qrel_map))
            prefix1 = EVALUATOR_LOGS / f"{population.replace(' ', '_')}_{split}_pass1"
            prefix2 = EVALUATOR_LOGS / f"{population.replace(' ', '_')}_{split}_pass2"
            first = score_official(filtered, qrel_paths[split], prefix1)
            second = score_official(filtered, qrel_paths[split], prefix2)
            repeat = max(abs(first["MAP_at_10"] - second["MAP_at_10"]), abs(first["MRR_at_10"] - second["MRR_at_10"]))
            max_repeat = max(max_repeat, repeat)
            independent = score_independent(filtered, qrel_paths[split], EVALUATOR_LOGS / f"{population.replace(' ', '_')}_{split}_independent")
            diff = max(abs(first["MAP_at_10"] - independent["overall"]["map_cut_10"]), abs(first["MRR_at_10"] - independent["overall"]["recip_rank"]))
            max_independent = max(max_independent, diff)
            independent_records[(population, split)] = independent
            results.append({"Split": split, "System": "RALSR + CrossEncoder", "Population": population, "QID_Count": len(qrel_map), "MAP_at_10": first["MAP_at_10"], "MRR_at_10": first["MRR_at_10"], "Run_Path": str(filtered), "Qrels_Path": str(qrel_paths[split])})
    results_df = pd.DataFrame(results)
    write_csv(OUTPUT / "CROSSENCODER_RESULTS_ALL_SPLITS.csv", results_df)
    write_csv(OUTPUT / "CROSSENCODER_RESULTS_OFFICIAL_JUDGED.csv", results_df[results_df.Population == "Official judged"])
    write_csv(OUTPUT / "CROSSENCODER_RESULTS_ANSWERABLE_ONLY.csv", results_df[results_df.Population == "Answerable-only"])

    test_off = results_df[(results_df.Split == "test") & (results_df.Population == "Official judged")].iloc[0]
    test_ans = results_df[(results_df.Split == "test") & (results_df.Population == "Answerable-only")].iloc[0]
    ce_values = [float(test_off.MAP_at_10), float(test_off.MRR_at_10), float(test_ans.MAP_at_10), float(test_ans.MRR_at_10)]
    comparison = load_frozen_comparators()
    ce_row = pd.DataFrame([{"System": "RALSR + CrossEncoder", "Official_MAP_at_10": ce_values[0], "Official_MRR_at_10": ce_values[1], "Answerable_MAP_at_10": ce_values[2], "Answerable_MRR_at_10": ce_values[3]}])
    comparison = pd.concat([comparison, ce_row], ignore_index=True)
    write_csv(OUTPUT / "FINAL_EXTENSION_TEST_COMPARISON.csv", comparison)
    delta_rows = []
    for system in ("Fixed RALSR", "Dense E5", "Dense E5 + CSR"):
        base = comparison[comparison.System == system].iloc[0]
        delta_rows.append({"Comparison": f"RALSR + CrossEncoder - {system}", "Delta_Official_MAP_at_10": ce_values[0] - float(base.Official_MAP_at_10), "Delta_Official_MRR_at_10": ce_values[1] - float(base.Official_MRR_at_10), "Delta_Answerable_MAP_at_10": ce_values[2] - float(base.Answerable_MAP_at_10), "Delta_Answerable_MRR_at_10": ce_values[3] - float(base.Answerable_MRR_at_10)})
    deltas = pd.DataFrame(delta_rows)
    write_csv(OUTPUT / "CROSSENCODER_TEST_DELTAS.csv", deltas)

    official_test = independent_records[("Official judged", "test")]
    qrels_test = read_qrels(QRELS["test"])
    internal = pd.read_csv(INTERNAL_RUN, dtype={"QID": str, "Passage_ID": str})
    top_predictions = internal.sort_values(["QID", "Rank"]).groupby("QID").first().Passage_ID.to_dict()
    step8_master = pd.read_csv(STEP8 / "TEST_QUERY_ERROR_MASTER_TABLE.csv", dtype={"QID": str})
    step8_map = step8_master.set_index("QID").to_dict("index")
    per_query = []
    for qid in sorted(qrels_test, key=int):
        gold_docs = {pid for pid, rel in qrels_test[qid].items() if rel > 0 and pid != "-1"}
        gold_status = "no-answer" if "-1" in {pid for pid, rel in qrels_test[qid].items() if rel > 0} else "answerable"
        group = internal[internal.QID == qid].sort_values("Rank")
        best_ce = min((int(r.Rank) for r in group.itertuples() if r.Passage_ID in gold_docs), default=None)
        fixed_best = step8_map.get(qid, {}).get("Best_Gold_Fixed_RALSR_Candidate_Rank", "")
        score = official_test["per_qid"][qid]
        per_query.append({"QID": qid, "Gold_Status": gold_status, "AP_at_10": score["map_cut_10"], "Reciprocal_Rank": score["recip_rank"], "Top_Prediction": top_predictions.get(qid, ""), "Top_Relevant_Rank": best_ce if best_ce is not None and best_ce <= 10 else "", "Fixed_RALSR_Top_Relevant_Rank": fixed_best, "CrossEncoder_Top_Relevant_Rank": best_ce if best_ce is not None else "", "Output_Type": group.iloc[0].Output_Type if len(group) else "missing", "Candidate_Set_Status": "relevant_present" if best_ce is not None else ("structural_zero_candidates" if group.iloc[0].Passage_ID == "-1" else "relevant_absent")})
    per_query_df = pd.DataFrame(per_query)
    write_csv(OUTPUT / "PER_QUERY_TEST_RALSR_CROSSENCODER.csv", per_query_df)

    answerable = step8_master.copy()
    ce_lookup = per_query_df.set_index("QID").to_dict("index")
    fixed_comparison = []
    for row in answerable.itertuples(index=False):
        qid = str(row.QID)
        ce = ce_lookup[qid]
        fixed_ap, fixed_rr = float(row.Fixed_RALSR_AP_at_10), float(row.Fixed_RALSR_RR)
        ce_ap, ce_rr = float(ce["AP_at_10"]), float(ce["Reciprocal_Rank"])
        fixed_rank = None if pd.isna(row.Fixed_RALSR_Best_Relevant_Top10_Rank) else int(float(row.Fixed_RALSR_Best_Relevant_Top10_Rank))
        ce_rank = None if ce["CrossEncoder_Top_Relevant_Rank"] == "" else int(ce["CrossEncoder_Top_Relevant_Rank"])
        fixed_comparison.append({"QID": qid, "Fixed_AP_at_10": fixed_ap, "CrossEncoder_AP_at_10": ce_ap, "AP_Change": "improved" if ce_ap > fixed_ap + 1e-15 else ("worsened" if ce_ap < fixed_ap - 1e-15 else "unchanged"), "Fixed_RR": fixed_rr, "CrossEncoder_RR": ce_rr, "RR_Change": "improved" if ce_rr > fixed_rr + 1e-15 else ("worsened" if ce_rr < fixed_rr - 1e-15 else "unchanged"), "Fixed_Top10_Relevant": fixed_rank is not None, "CrossEncoder_Top10_Relevant": ce_rank is not None and ce_rank <= 10, "Newly_Entered_Top10": fixed_rank is None and ce_rank is not None and ce_rank <= 10, "Left_Top10": fixed_rank is not None and (ce_rank is None or ce_rank > 10), "Reached_Rank1": ce_rank == 1})
    fixed_comparison_df = pd.DataFrame(fixed_comparison)
    write_csv(OUTPUT / "CROSSENCODER_VS_FIXED_QUERY_COMPARISON.csv", fixed_comparison_df)

    ranking_failure_qids = set(step8_master[step8_master.Primary_RALSR_Outcome == "Candidate present but ranking failure"].QID)
    rescue_rows = []
    for qid in sorted(ranking_failure_qids, key=int):
        row = step8_map[qid]
        fixed_rank = int(float(row["Best_Gold_Fixed_RALSR_Candidate_Rank"]))
        ce_rank_raw = ce_lookup[qid]["CrossEncoder_Top_Relevant_Rank"]
        ce_rank = int(ce_rank_raw) if ce_rank_raw != "" else None
        if ce_rank is not None and ce_rank <= 10:
            category = "rescued_into_top10"
        elif ce_rank is not None and ce_rank < fixed_rank:
            category = "improved_but_below_top10"
        elif ce_rank == fixed_rank:
            category = "unchanged"
        else:
            category = "worsened"
        rescue_rows.append({"QID": qid, "Fixed_Relevant_Rank": fixed_rank, "CrossEncoder_Relevant_Rank": ce_rank if ce_rank is not None else "", "Rescue_Category": category, "Moved_to_Rank1": ce_rank == 1})
    rescue_df = pd.DataFrame(rescue_rows)
    if len(rescue_df) != 21:
        raise RuntimeError("Step-8 ranking-failure set is not 21 qids")
    write_csv(OUTPUT / "STEP8_RANKING_FAILURE_RESCUE_ANALYSIS.csv", rescue_df)

    outcome = pd.read_csv(STEP8 / "DENSE_VS_RALSR_OUTCOME_GROUPS.csv")
    dense_only_text = outcome[outcome.Outcome_Group == "Dense E5-only success"].iloc[0].QIDs
    dense_only_qids = dense_only_text.split(";")
    dense_rows = []
    for qid in dense_only_qids:
        row = step8_map[qid]
        fixed_rank_raw = row["Best_Gold_Fixed_RALSR_Candidate_Rank"]
        fixed_rank = None if fixed_rank_raw == "" or pd.isna(fixed_rank_raw) else int(float(fixed_rank_raw))
        ce_rank_raw = ce_lookup[qid]["CrossEncoder_Top_Relevant_Rank"]
        ce_rank = int(ce_rank_raw) if ce_rank_raw != "" else None
        if fixed_rank is None:
            category = "candidate_unrecoverable"
        elif ce_rank is not None and ce_rank <= 10:
            category = "rescued_into_top10"
        elif ce_rank is not None and ce_rank < fixed_rank:
            category = "improved_but_below_top10"
        elif ce_rank == fixed_rank:
            category = "unchanged"
        else:
            category = "worsened"
        dense_rows.append({"QID": qid, "Rerankable": fixed_rank is not None, "Fixed_Relevant_Rank": fixed_rank if fixed_rank is not None else "", "CrossEncoder_Relevant_Rank": ce_rank if ce_rank is not None else "", "Outcome": category})
    dense_df = pd.DataFrame(dense_rows)
    write_csv(OUTPUT / "DENSE_ONLY_CASE_RESCUE_ANALYSIS.csv", dense_df)

    ralsr_only_text = outcome[outcome.Outcome_Group == "Fixed RALSR-only success"].iloc[0].QIDs
    complementary_rows = []
    for qid in ralsr_only_text.split(";"):
        fixed_ap = float(step8_map[qid]["Fixed_RALSR_AP_at_10"])
        ce_ap = float(ce_lookup[qid]["AP_at_10"])
        effect = "improves" if ce_ap > fixed_ap + 1e-15 else ("harms_loses_success" if ce_ap == 0 else "preserves_success")
        complementary_rows.append({"QID": qid, "Fixed_AP_at_10": fixed_ap, "CrossEncoder_AP_at_10": ce_ap, "Effect": effect})
    complementary_df = pd.DataFrame(complementary_rows)
    write_csv(OUTPUT / "RALSR_COMPLEMENTARY_CASE_PRESERVATION.csv", complementary_df)

    candidate_ceiling = pd.DataFrame([
        {"Category": "Structural zero-candidate failures", "Count": 2, "Reranker_Can_Resolve": False},
        {"Category": "Nonempty candidates without relevant passage", "Count": 6, "Reranker_Can_Resolve": False},
        {"Category": "Relevant candidate below Top 10", "Count": 21, "Reranker_Can_Resolve": True},
        {"Category": "Existing fixed-RALSR Top-10 successes", "Count": 15, "Reranker_Can_Resolve": "preserve_or_improve"},
    ])

    table_a = comparison.copy()
    table_b = pd.DataFrame([{"Metric": col.replace("_at_10", "@10").replace("Official_", "Official ").replace("Answerable_", "Answerable-only "), "Fixed RALSR": float(comparison[comparison.System == "Fixed RALSR"].iloc[0][col]), "RALSR + CrossEncoder": float(ce_row.iloc[0][col]), "Absolute Delta": float(ce_row.iloc[0][col]) - float(comparison[comparison.System == "Fixed RALSR"].iloc[0][col])} for col in ["Official_MAP_at_10", "Official_MRR_at_10", "Answerable_MAP_at_10", "Answerable_MRR_at_10"]])
    rescue_counts = rescue_df.Rescue_Category.value_counts().to_dict()
    table_c = pd.DataFrame([{"Measure": "Total frozen Step-8 ranking failures", "Count": 21, "Category_Relationship": "total"}, {"Measure": "Rescued into Top 10", "Count": rescue_counts.get("rescued_into_top10", 0), "Category_Relationship": "mutually exclusive"}, {"Measure": "Moved to rank 1", "Count": int(rescue_df.Moved_to_Rank1.sum()), "Category_Relationship": "annotation overlapping rescued"}, {"Measure": "Improved but below Top 10", "Count": rescue_counts.get("improved_but_below_top10", 0), "Category_Relationship": "mutually exclusive"}, {"Measure": "Unchanged", "Count": rescue_counts.get("unchanged", 0), "Category_Relationship": "mutually exclusive"}, {"Measure": "Worsened", "Count": rescue_counts.get("worsened", 0), "Category_Relationship": "mutually exclusive"}])
    thesis_tables = {"TABLE_FINAL_SYSTEM_COMPARISON": table_a, "TABLE_CROSSENCODER_VS_FIXED_RALSR": table_b, "TABLE_RANKING_FAILURE_RESCUE": table_c, "TABLE_CANDIDATE_GENERATION_CEILING": candidate_ceiling}
    for name, frame in thesis_tables.items():
        write_csv(TABLES / f"{name}.csv", frame)

    reproducibility = {"conditions": 6, "repeat_maximum_difference": max_repeat, "independent_maximum_difference": max_independent, "repeat_passed": max_repeat <= 1e-15, "independent_passed": max_independent <= 1e-12}
    write_json(OUTPUT / "CROSSENCODER_SCORING_REPRODUCIBILITY.json", reproducibility)
    summary = {
        "ce_values": ce_values, "results": results, "deltas": delta_rows,
        "answerable_query_changes": {"ap": fixed_comparison_df.AP_Change.value_counts().to_dict(), "rr": fixed_comparison_df.RR_Change.value_counts().to_dict(), "new_top10": fixed_comparison_df[fixed_comparison_df.Newly_Entered_Top10].QID.tolist(), "left_top10": fixed_comparison_df[fixed_comparison_df.Left_Top10].QID.tolist(), "rank1": fixed_comparison_df[fixed_comparison_df.Reached_Rank1].QID.tolist()},
        "ranking_failure_rescue": {"counts": rescue_counts, "rank1": rescue_df[rescue_df.Moved_to_Rank1].QID.tolist()},
        "dense_only": {"total": len(dense_df), "rerankable": int(dense_df.Rerankable.sum()), "outcomes": dense_df.Outcome.value_counts().to_dict()},
        "ralsr_complementary": complementary_df.Effect.value_counts().to_dict(),
        "candidate_ceiling": {"structural": 2, "candidate_generation": 6, "rerankable_failures": 21, "existing_successes": 15},
        "reproducibility": reproducibility,
    }
    write_json(OUTPUT / "CROSSENCODER_ANALYSIS_SUMMARY.json", summary)
    print(json.dumps({"phase": "evaluate", "status": "PASS", **summary}, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "infer", "evaluate", "all"])
    args = parser.parse_args()
    if args.phase in ("prepare", "all"):
        phase_prepare()
    if args.phase in ("infer", "all"):
        phase_infer()
    if args.phase in ("evaluate", "all"):
        phase_evaluate()


if __name__ == "__main__":
    main()
