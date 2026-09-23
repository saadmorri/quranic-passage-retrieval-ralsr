#!/usr/bin/env python3
"""Step 9: freeze and evaluate the pre-existing Dense E5 + CSR extension.

The authoritative Dense+CSR Top-100 ranking is read-only. This script freezes
its recovered configuration before scoring, converts stored ranks to the
already validated official format, validates the derivatives, and evaluates
the frozen run under the Step-4 populations. It never encodes text, regenerates
retrieval scores, or changes the source ranking.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


EXPECTED = {
    "baseline": "5EE78669B8BD3E544873CD71946CE9841A87D25F6B82182B899C7DC2CBC2AD15",
    "query_csr": "6972313CF9A0A8897DA4DD0DF64DB8132A7B68ED766F2C199072ECCBF3D10987",
    "passage_csr": "7D839476B922628E9123A538DBBDCFF6227D7A762B7866E4A151A8E11964A745",
    "internal_run": "025DAB1CCD7C604888C0834193FC9DB5C5620747CB566B77038823945A959489",
    "qpc": "0A86C33C465AB6CF9321924D2C03B23ED72F8360134AE92BA4BD4A90C93BE08C",
    "question_train": "3B707F9A857A668B71A2DADEA9AA6545F038E656F6092E6D78EC0C4C99A3A0A9",
    "question_dev": "BAAE0A5DFE7ADC3E300DC79F504527897E84C2E3DACB19789638ECB29E97229B",
    "question_test": "33C82BD0918F2742360FCCB7598E655241FB5CFEEEECB7F2B009EC59B16E1155",
    "qrels_train": "48E64E24A715BD77B824D9A4C17863C39614FE198A0A44FED23E76A6BEA0CFE6",
    "qrels_dev": "6F74218F1259AA144C795F19D0E4FB99095F927511EE708FA7EE4125F37A3590",
    "qrels_test": "E23E4CF0628EB2FF39562852A5632DE0D948C8F643B5EB8E08B1D8B69CBA0332",
    "answerable_train": "FD8D351896507C33021EB8B1F5BAE403F09105495E51D7ECDA4AF44B19F93176",
    "answerable_dev": "4D372EEF4655E27662F1F017557B01F003C6A16C57E16074621F1E227E31EAF6",
    "answerable_test": "E799BEC96EA33B7ECDFC0B8285B85B4F16521B6C2FDD7DD02AA7AFE0F608872B",
    "converter": "BA8BAB8EBA4FA2AFFA51A3A1A815C7D4AF471042E4291A52550B3E7354B302C5",
    "strict_validator": "9FBE397509F1BDCE2A0922115635A01251C63330F9B5F79296F3354A9EB01FBF",
    "official_checker": "EA322F186C7EA9A31B05F88E5D4800A53540F270A3E893070D4538BB1F5B745F",
    "official_scorer": "39F98576783FBB41222360F2319BC011A90400075122334728BC68BBA55DBE1A",
    "independent_evaluator": "DABB1B45B0905B589E8A8AF07967B7CE1184CED8D88ADA75BEE9BE4F80239253",
    "notebook_9d": "4C33191EF9F45AC2C8208CB31F8F35FAE7C3815B29015825BFEE7925082D855F",
    "notebook_10d": "1987EC3DC7E736D74A58F97AF01AD3D22261300B6A2300DF72070F311D6970A5",
    "model_config": "9DAB198F24C8C0879E481CF7822005D5ECBCEEDBACB390FFAFA594E28D31BAC4",
    "model_modules": "C6E29747481E8B5DD2B58401966AEAC910DE39092F90CDA9A704B1545F902B04",
    "model_pooling": "F586AB6C734AF7B8D14D898BFA7FAA886F23B46F86DD1730514309893AF13D75",
    "model_weights": "A18A44FAD1D0B46DED15928144138CFF1135D5CC8233BDD90BE5F18822DE09A7",
    "step7_freeze": "25C375F39B33A075D3E6800ECF37CDAAA3D9A54BCC3CA46245D5BCF3A79E1484",
    "adapter": "BF3CB90EA6F5B30AB42E37CA409256EC5F26186F92F7ED37B3625E17CB7347E1",
}

CORE_TEST = {
    "BM25": (0.073356767544, 0.174229691877, 0.085027162380, 0.201948051948),
    "Dense E5": (0.099526538915, 0.289542483660, 0.115360306470, 0.335606060606),
    "Fixed RALSR": (0.042896916937, 0.135154061625, 0.049721426449, 0.156655844156),
    "Tuned RALSR": (0.040843540030, 0.135862122627, 0.047341375944, 0.157476551227),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_question_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        qids = [str(row[0]).strip() for row in csv.reader(handle, delimiter="\t") if row]
    if len(qids) != len(set(qids)):
        raise RuntimeError(f"Duplicate qid in {path}")
    return qids


def read_qpc_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row[0]).strip() for row in csv.reader(handle, delimiter="\t") if row}


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) != 4:
                raise RuntimeError(f"Malformed qrel row: {path}: {row}")
            qid, _, docid, relevance = row
            result[str(qid).strip()][str(docid).strip()] = int(relevance)
    return dict(result)


def run_command(command: list[str], log_path: Path, env=None) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")
    return completed


def parse_trec(path: Path) -> dict[str, list[tuple[int, str, float, str]]]:
    grouped: dict[str, list[tuple[int, str, float, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) != 6:
                raise RuntimeError(f"{path}:{line_number}: expected six columns")
            qid, q0, docid, rank_text, score_text, tag = row
            if q0 != "Q0" or not tag:
                raise RuntimeError(f"{path}:{line_number}: malformed Q0/tag")
            rank, score = int(rank_text), float(score_text)
            if rank < 1 or not math.isfinite(score):
                raise RuntimeError(f"{path}:{line_number}: invalid rank/score")
            grouped[qid].append((rank, docid, score, tag))
    for qid, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        if [row[0] for row in rows] != list(range(1, len(rows) + 1)):
            raise RuntimeError(f"Rank continuity failure for {qid} in {path}")
        if len(rows) > 10:
            raise RuntimeError(f"Top-10 depth exceeded for {qid} in {path}")
        if len({row[1] for row in rows}) != len(rows):
            raise RuntimeError(f"Duplicate qid/docid for {qid} in {path}")
        if any(rows[i][2] <= rows[i + 1][2] for i in range(len(rows) - 1)):
            raise RuntimeError(f"Strict score ordering failure for {qid} in {path}")
    return dict(grouped)


def filter_run_exact(source: Path, destination: Path, qids: set[str]) -> dict:
    kept, seen = [], set()
    for line in source.read_bytes().splitlines(keepends=True):
        qid = line.split(b"\t", 1)[0].decode("utf-8-sig").strip()
        if qid in qids:
            kept.append(line)
            seen.add(qid)
    if seen != qids:
        raise RuntimeError(f"Filtered run mismatch for {source}: missing={sorted(qids-seen)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = b"".join(kept)
    if destination.is_file():
        if destination.read_bytes() != expected_bytes:
            raise RuntimeError(f"Existing filtered run differs from deterministic derivative: {destination}")
    else:
        destination.write_bytes(expected_bytes)
    return {"rows": len(kept), "qids": len(seen), "sha256": sha256(destination)}


def score_official(run: Path, qrels: Path, scorer: Path, organizer: Path, adapter: Path, prefix: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(adapter), str(organizer), env.get("PYTHONPATH", "")])
    metrics = prefix.with_suffix(".metrics.tsv")
    completed = run_command(
        [sys.executable, str(scorer), "--run", str(run), "--qrels", str(qrels), "--output", str(metrics)],
        prefix.with_suffix(".scorer.log"),
        env=env,
    )
    if "Format check: Passed" not in completed.stdout:
        raise RuntimeError(f"Organizer scorer format check failed for {run}")
    frame = pd.read_csv(metrics, sep="\t")
    return {
        "map_cut_10": float(frame.loc[0, "map_cut_10"]),
        "recip_rank": float(frame.loc[0, "recip_rank"]),
        "metrics_path": str(metrics),
    }


def score_independent(run: Path, qrels: Path, evaluator: Path, prefix: Path) -> dict:
    output = prefix.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    run_command(
        [sys.executable, str(evaluator), "--run", str(run), "--qrels", str(qrels), "--output", str(output)],
        prefix.with_suffix(".log"),
    )
    return json.loads(output.read_text(encoding="utf-8"))


def load_metrics(path: Path) -> dict:
    frame = pd.read_csv(path, sep="\t")
    return {
        "map_cut_10": float(frame.loc[0, "map_cut_10"]),
        "recip_rank": float(frame.loc[0, "recip_rank"]),
        "metrics_path": str(path),
    }


def main() -> None:
    output = Path(r"${THESIS_CONTROL_ROOT}\CSR_Final_Evaluation")
    if (output / "CSR_RESULT_FREEZE_MANIFEST.json").exists():
        raise SystemExit("Refusing to overwrite a completed Step-9 freeze")

    codex = Path(r"${THESIS_CODEX_ROOT}")
    workspace = codex / "Master Thesis - Quranic Passage Retrieval - Writing Workspace"
    correction = codex / "Master Thesis - Quranic Passage Retrieval - Correction Control"
    quranqa = workspace / "02_Methodology_Evidence" / "Data and Resources" / "QuranQA"
    csr_input = workspace / "02_Methodology_Evidence" / "CSR" / "Corrected_Outputs" / "CSR_Authoritative_Lineage"
    csr_retrieval = workspace / "02_Methodology_Evidence" / "CSR" / "Corrected_Outputs" / "Notebook_10D_CSR_Retrieval"
    csr_evidence = correction / "01_Corrective_Workspace" / "CSR_Authoritative_Lineage"
    retrieval_evidence = correction / "01_Corrective_Workspace" / "Notebook_10D_CSR_Retrieval"
    official = correction / "Official_QuranQA_TaskA_Evaluation"
    organizer = official / "01_Official_Organizer_Files"
    pipeline = official / "02_Evaluation_Pipeline"
    step4 = correction / "No_Answer_Audit"
    step7 = correction / "Final_Core_Evaluation"
    model = Path(r"${THESIS_MODEL_ROOT}\multilingual-e5-base")

    paths = {
        "baseline": workspace.with_suffix(".rar"),
        "query_csr": csr_input / "QuranQA_Query_CSR_Corrected.csv",
        "passage_csr": csr_input / "QuranQA_Passage_CSR_Corrected.csv",
        "internal_run": csr_retrieval / "Dense_CSR_Top100_Corrected.csv",
        "cache_manifest": csr_retrieval / "Dense_CSR_Cache" / "CSR_Embedding_Cache_Manifest.json",
        "notebook_9d": csr_evidence / "02_Executed_Notebook" / "Notebook_9D_CSR_Authoritative_Lineage_Corrected_Executed.ipynb",
        "notebook_10d": retrieval_evidence / "02_Executed_Notebook" / "Notebook_10D_Corrected_BM25_CSR_Dense_CSR_Executed.ipynb",
        "qrel_independence_report": retrieval_evidence / "03_Validation_Reports" / "QUESTION_ID_QREL_INDEPENDENCE_VALIDATION_REPORT.md",
        "determinism_report": retrieval_evidence / "03_Validation_Reports" / "NOTEBOOK_10D_DETERMINISM_REPORT.md",
        "qpc": quranqa / "QQA23_TaskA_QPC_v1.1.tsv",
        "converter": pipeline / "quranqa_taskA_convert_run.py",
        "strict_validator": pipeline / "quranqa_taskA_strict_validator.py",
        "official_checker": organizer / "QQA23_TaskA_submission_checker.py",
        "official_scorer": organizer / "QQA23_TaskA_eval.py",
        "independent_evaluator": pipeline / "quranqa_taskA_independent_eval.py",
        "adapter": step7 / "Official_Scorer_Compatibility" / "pytrec_eval.py",
        "step7_freeze": step7 / "CORE_RESULT_FREEZE_MANIFEST.json",
        "step7_main": step7 / "CORE_TEST_RESULTS_MAIN_TABLE.csv",
        "step7_dense_per_query": step7 / "PER_QUERY_TEST_DENSE.csv",
        "model_config": model / "config.json",
        "model_modules": model / "modules.json",
        "model_pooling": model / "1_Pooling" / "config.json",
        "model_weights": model / "model.safetensors",
    }
    questions = {split: quranqa / f"QQA23_TaskA_ayatec_v1.2_{split}.tsv" for split in ["train", "dev", "test"]}
    official_qrels = {split: quranqa / f"QQA23_TaskA_ayatec_v1.2_qrels_{split}.gold" for split in ["train", "dev", "test"]}
    answerable_qrels = {split: step4 / f"qrels_{split}_answerable_only.gold" for split in ["train", "dev", "test"]}
    answerable_masks = {split: step4 / f"ANSWERABLE_QIDS_{split}.txt" for split in ["train", "dev", "test"]}

    hash_targets = {
        "query_csr": paths["query_csr"], "passage_csr": paths["passage_csr"], "internal_run": paths["internal_run"],
        "qpc": paths["qpc"], "converter": paths["converter"], "strict_validator": paths["strict_validator"],
        "official_checker": paths["official_checker"], "official_scorer": paths["official_scorer"],
        "independent_evaluator": paths["independent_evaluator"], "notebook_9d": paths["notebook_9d"],
        "notebook_10d": paths["notebook_10d"], "model_config": paths["model_config"],
        "model_modules": paths["model_modules"], "model_pooling": paths["model_pooling"],
        "model_weights": paths["model_weights"], "step7_freeze": paths["step7_freeze"],
        "adapter": paths["adapter"],
    }
    for split in ["train", "dev", "test"]:
        hash_targets[f"question_{split}"] = questions[split]
        hash_targets[f"qrels_{split}"] = official_qrels[split]
        hash_targets[f"answerable_{split}"] = answerable_qrels[split]
    for key, path in hash_targets.items():
        actual = sha256(path)
        if actual != EXPECTED[key]:
            raise RuntimeError(f"Frozen identity mismatch for {key}: {actual} != {EXPECTED[key]}")

    step7_freeze = json.loads(paths["step7_freeze"].read_text(encoding="utf-8"))
    step7_artifacts = {Path(item["path"]).name: item["sha256"] for item in step7_freeze["artifacts"]}
    for key in ["CORE_TEST_RESULTS_MAIN_TABLE.csv", "PER_QUERY_TEST_DENSE.csv"]:
        path = step7 / key
        if sha256(path) != step7_artifacts[key]:
            raise RuntimeError(f"Step-7 comparator identity mismatch: {key}")

    split_ids = {split: read_question_ids(path) for split, path in questions.items()}
    if {split: len(qids) for split, qids in split_ids.items()} != {"train": 174, "dev": 25, "test": 52}:
        raise RuntimeError("Official split counts do not equal 174/25/52")
    if "504" not in split_ids["test"]:
        raise RuntimeError("qid 504 missing from official test questions")
    all_qids = [qid for split in ["train", "dev", "test"] for qid in split_ids[split]]
    qpc_ids = read_qpc_ids(paths["qpc"])
    if len(qpc_ids) != 1266:
        raise RuntimeError("Official QPC does not contain 1,266 passage IDs")

    query_df = pd.read_csv(paths["query_csr"], encoding="utf-8-sig", dtype={"Question_ID": "string"})
    passage_df = pd.read_csv(paths["passage_csr"], encoding="utf-8-sig", dtype={"Passage_ID": "string"})
    run_df = pd.read_csv(paths["internal_run"], encoding="utf-8-sig", dtype={"Question_ID": "string", "Passage_ID": "string"})
    run_df["Question_ID"] = run_df["Question_ID"].str.strip()
    run_df["Passage_ID"] = run_df["Passage_ID"].str.strip()
    run_df["Rank"] = pd.to_numeric(run_df["Rank"], errors="raise").astype(int)
    run_df["Dense_CSR_Score"] = pd.to_numeric(run_df["Dense_CSR_Score"], errors="raise")

    internal_errors = []
    if len(query_df) != 251 or query_df["Question_ID"].nunique() != 251:
        internal_errors.append("query CSR count")
    if len(passage_df) != 1266 or passage_df["Passage_ID"].nunique() != 1266:
        internal_errors.append("passage CSR count")
    if len(run_df) != 25100 or run_df["Question_ID"].nunique() != 251:
        internal_errors.append("internal run count")
    if set(run_df["Question_ID"]) != set(all_qids):
        internal_errors.append("qid membership")
    if "504" not in set(run_df["Question_ID"]):
        internal_errors.append("qid 504")
    invalid_qpc = int((~run_df["Passage_ID"].isin(qpc_ids)).sum())
    duplicate_pairs = int(run_df.duplicated(["Question_ID", "Passage_ID"]).sum())
    nonfinite_scores = int((~run_df["Dense_CSR_Score"].map(math.isfinite)).sum())
    rank_violations = score_violations = tie_violations = 0
    depth_counts = Counter()
    adjacent_score_ties = 0
    source_by_qid: dict[str, list[dict[str, str]]] = {}
    for qid, group in run_df.groupby("Question_ID", sort=False):
        group = group.sort_values("Rank", kind="stable")
        depth_counts[len(group)] += 1
        ranks = group["Rank"].tolist()
        scores = group["Dense_CSR_Score"].tolist()
        docs = group["Passage_ID"].tolist()
        if ranks != list(range(1, len(group) + 1)):
            rank_violations += 1
        for i in range(len(group) - 1):
            if scores[i] < scores[i + 1]:
                score_violations += 1
            if scores[i] == scores[i + 1]:
                adjacent_score_ties += 1
                if docs[i] > docs[i + 1]:
                    tie_violations += 1
        source_by_qid[str(qid)] = [
            {"Passage_ID": doc, "Rank": str(rank), "Dense_CSR_Score": str(score)}
            for rank, doc, score in zip(ranks, docs, scores)
        ]
    if invalid_qpc or duplicate_pairs or nonfinite_scores or rank_violations or score_violations or tie_violations:
        internal_errors.append("structural/order validation")
    if depth_counts != Counter({100: 251}):
        internal_errors.append("depth distribution")
    if (run_df["Passage_ID"] == "-1").any():
        internal_errors.append("unexpected no-answer row")
    if internal_errors:
        raise RuntimeError(f"Authoritative CSR run failed validation: {internal_errors}")

    cache = json.loads(paths["cache_manifest"].read_text(encoding="utf-8"))
    expected_cache = {
        "query_csr_sha256": EXPECTED["query_csr"],
        "passage_csr_sha256": EXPECTED["passage_csr"],
        "model_config_sha256": EXPECTED["model_config"],
        "model_modules_sha256": EXPECTED["model_modules"],
        "model_pooling_sha256": EXPECTED["model_pooling"],
        "model_name": "multilingual-e5-base",
        "passage_prefix": "passage: ",
        "query_prefix": "query: ",
        "batch_size": 16,
        "normalize_embeddings": True,
    }
    if cache != expected_cache:
        raise RuntimeError("Dense+CSR cache identity differs from the frozen configuration")

    # Freeze the complete pre-existing configuration before conversion/scoring.
    configuration = {
        "status": "frozen_before_scoring",
        "frozen_at_local": datetime.now().astimezone().isoformat(),
        "system_name": "Dense E5 + Compact Semantic Representation (Dense E5 + CSR)",
        "experimental_role": "preplanned thesis extension",
        "csr_definition": {
            "text_order": ["original text", "authoritative roots", "selected semantic keywords"],
            "semantic_source": "successful exact Maqayis matches only",
            "normalization": [
                "remove Arabic diacritics U+0617-U+061A, U+064B-U+0652, superscript alef, and tatweel",
                "map أ/إ/آ to ا, ى to ي, ؤ to و, ئ to ي, and ة to ه",
                "replace non-Arabic-letter characters with spaces and collapse whitespace",
            ],
            "token_rule": "Arabic-only normalized token, minimum length 3, not in the frozen Arabic/formulaic stopword set after common-prefix variants",
            "selection_order": "meanings in authoritative order; first eligible tokens per meaning; first-occurrence deduplication; then global cap",
            "query_cap": {"per_meaning": 5, "overall": 20},
            "passage_cap": {"per_meaning": 3, "overall": 60},
            "root_rule": "query roots are ordered-unique Notebook-7 accepted roots; passage roots are copied exactly from Notebook 9A",
            "valid_root_without_maqayis": "root retained; no semantic keyword invented",
            "unresolved_query_term": "no root or semantic keyword invented; original question text remains",
        },
        "dense_configuration": {
            "model_id": "intfloat/multilingual-e5-base",
            "model_revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
            "framework": "SentenceTransformers over XLM-RoBERTa",
            "pooling": "attention-mask-aware mean-token pooling",
            "embedding_dimension": 768,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "batch_size": 16,
            "normalized_embeddings": True,
            "similarity": "dot product of L2-normalized embeddings (cosine-equivalent)",
            "retrieval_depth": 100,
            "ranking": "similarity descending; exact ties by Passage_ID ascending",
            "no_answer_behavior": "none; every query receives normal passage rankings",
            "model_weights_sha256": EXPECTED["model_weights"],
        },
        "inputs": {
            "query_csr": {"path": str(paths["query_csr"]), "sha256": EXPECTED["query_csr"], "rows": 251},
            "passage_csr": {"path": str(paths["passage_csr"]), "sha256": EXPECTED["passage_csr"], "rows": 1266},
        },
        "authoritative_internal_run": {"path": str(paths["internal_run"]), "sha256": EXPECTED["internal_run"], "rows": 25100, "qids": 251},
        "generating_evidence": {
            "notebook_9d": {"path": str(paths["notebook_9d"]), "sha256": EXPECTED["notebook_9d"]},
            "notebook_10d": {"path": str(paths["notebook_10d"]), "sha256": EXPECTED["notebook_10d"]},
            "cache_manifest": {"path": str(paths["cache_manifest"]), "sha256": sha256(paths["cache_manifest"]), "content": cache},
            "determinism_report": {"path": str(paths["determinism_report"]), "sha256": sha256(paths["determinism_report"])},
        },
        "qrel_independence": {
            "result": "PASS",
            "construction_and_retrieval_use_qrels": False,
            "evidence_report": str(paths["qrel_independence_report"]),
            "evidence_report_sha256": sha256(paths["qrel_independence_report"]),
        },
    }
    config_path = output / "CSR_CONFIGURATION_FREEZE.json"
    if config_path.is_file():
        existing_configuration = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            existing_configuration.get("system_name") != configuration["system_name"]
            or existing_configuration.get("authoritative_internal_run", {}).get("sha256") != EXPECTED["internal_run"]
            or existing_configuration.get("inputs", {}).get("query_csr", {}).get("sha256") != EXPECTED["query_csr"]
            or existing_configuration.get("inputs", {}).get("passage_csr", {}).get("sha256") != EXPECTED["passage_csr"]
        ):
            raise RuntimeError("Existing pre-scoring configuration freeze is inconsistent")
    else:
        write_json(config_path, configuration)
    frozen_config_hash = sha256(config_path)

    official_runs = {}
    validation = {}
    conversion_logs = output / "Validation_Logs"
    filenames = {"train": "thesis_CSRtr.tsv", "dev": "thesis_CSRdv.tsv", "test": "thesis_CSRte.tsv"}
    with tempfile.TemporaryDirectory(prefix="quranqa_step9_") as temp_name:
        temp = Path(temp_name)
        for split in ["train", "dev", "test"]:
            split_set = set(split_ids[split])
            subset = run_df.loc[run_df["Question_ID"].isin(split_set)].copy()
            temp_input = temp / f"DenseCSR_{split}.csv"
            destination = output / filenames[split]
            if not destination.is_file():
                subset.to_csv(temp_input, index=False, encoding="utf-8", lineterminator="\n")
                run_command(
                    [
                        sys.executable, str(paths["converter"]), "--input", str(temp_input), "--output", str(destination),
                        "--questions", str(questions[split]), "--tag", "DenseCSR", "--qid-col", "Question_ID",
                        "--docid-col", "Passage_ID", "--score-col", "Dense_CSR_Score", "--rank-col", "Rank",
                        "--top-k", "10", "--score-mode", "rank-derived",
                    ],
                    conversion_logs / f"{filenames[split]}.conversion.log",
                )
            if destination.exists() is False:
                raise RuntimeError(f"Converter did not create {destination}")
            strict_json = conversion_logs / f"{filenames[split]}.strict.json"
            if not strict_json.is_file():
                run_command(
                    [sys.executable, str(paths["strict_validator"]), "--run", str(destination), "--questions", str(questions[split]), "--qpc", str(paths["qpc"]), "--json-report", str(strict_json)],
                    conversion_logs / f"{filenames[split]}.strict.log",
                )
            strict = json.loads(strict_json.read_text(encoding="utf-8"))
            checker_log = conversion_logs / f"{filenames[split]}.organizer_checker.log"
            if not checker_log.is_file():
                checker = run_command(
                    [sys.executable, str(paths["official_checker"]), "--model-prediction", str(destination)],
                    checker_log,
                )
                checker_text = checker.stdout
            else:
                checker_text = checker_log.read_text(encoding="utf-8")
            if not strict.get("passed") or "Format check: Passed" not in checker_text:
                raise RuntimeError(f"Official-format validation failed: {split}")
            grouped = parse_trec(destination)
            if set(grouped) != split_set:
                raise RuntimeError(f"Official qid membership mismatch: {split}")
            if any(len(rows) != 10 for rows in grouped.values()):
                raise RuntimeError(f"Official depth mismatch: {split}")
            mismatches = 0
            invalid = 0
            duplicates = 0
            for qid in split_ids[split]:
                actual_docs = [row[1] for row in grouped[qid]]
                expected_docs = [row["Passage_ID"] for row in source_by_qid[qid][:10]]
                if actual_docs != expected_docs:
                    mismatches += 1
                invalid += sum(doc not in qpc_ids for doc in actual_docs)
                duplicates += len(actual_docs) - len(set(actual_docs))
            if mismatches or invalid or duplicates:
                raise RuntimeError(f"Frozen ranking mismatch/invalid output: {split}")
            official_runs[split] = destination
            validation[split] = {
                "path": str(destination), "sha256": sha256(destination), "rows": sum(len(v) for v in grouped.values()),
                "qids": len(grouped), "qid_504_present": "504" in grouped, "validator_passed": True,
                "organizer_checker_passed": True, "ranking_mismatches": mismatches, "invalid_qpc_ids": invalid,
                "duplicate_pairs": duplicates, "no_answer_rows": sum(row[1] == "-1" for rows in grouped.values() for row in rows),
                "score_mode": "rank-derived official-format score; internal Dense_CSR_Score unchanged",
            }

    internal_validation = {
        "passed": True,
        "query_csr_rows": len(query_df), "passage_csr_rows": len(passage_df), "run_rows": len(run_df),
        "run_qids": run_df["Question_ID"].nunique(), "depth_distribution": dict(depth_counts),
        "qid_504_present": "504" in set(run_df["Question_ID"]), "invalid_qpc_ids": invalid_qpc,
        "duplicate_pairs": duplicate_pairs, "nonfinite_scores": nonfinite_scores,
        "rank_violations": rank_violations, "score_order_violations": score_violations,
        "tie_order_violations": tie_violations, "adjacent_equal_score_pairs": adjacent_score_ties,
        "no_answer_rows": int((run_df["Passage_ID"] == "-1").sum()), "qrels_used": False,
    }
    internal_report_path = output / "CSR_INTERNAL_RUN_VALIDATION.md"
    internal_report_text = (
        "# Dense E5 + CSR Internal Run Validation\n\n"
        f"The frozen source run is `{paths['internal_run']}` with SHA-256 `{EXPECTED['internal_run']}`. "
        "It was audited read-only; no embedding or retrieval operation was executed.\n\n"
        "| Check | Result |\n|---|---:|\n"
        f"| Query CSR rows | {len(query_df)} |\n| Passage CSR rows | {len(passage_df)} |\n"
        f"| Run rows / qids | {len(run_df):,} / {run_df['Question_ID'].nunique()} |\n"
        f"| Depth | 100 for all 251 qids |\n| qid 504 | Present |\n"
        f"| Invalid QPC IDs | {invalid_qpc} |\n| Duplicate pairs | {duplicate_pairs} |\n"
        f"| Non-finite scores | {nonfinite_scores} |\n| Rank / score / tie-order violations | {rank_violations} / {score_violations} / {tie_violations} |\n"
        f"| Structural `-1` rows | 0 |\n\n"
        "The active Notebook 9D/10D evidence loads no qrels. Query eligibility comes only from the 251-row corrected Query CSR, and passage eligibility comes only from the corrected Passage CSR reconciled to the official QPC.\n",
    )
    if not internal_report_path.is_file():
        internal_report_path.write_text(internal_report_text, encoding="utf-8")
    official_validation_path = output / "CSR_OFFICIAL_RUN_VALIDATION.json"
    if not official_validation_path.is_file():
        write_json(official_validation_path, {"internal": internal_validation, "official_runs": validation, "ranking_mismatch_total": sum(v["ranking_mismatches"] for v in validation.values())})

    # Scoring begins only after the configuration and official files are frozen.
    qrels = {split: read_qrels(path) for split, path in official_qrels.items()}
    answerable = {split: read_qrels(path) for split, path in answerable_qrels.items()}
    if {split: len(v) for split, v in qrels.items()} != {"train": 174, "dev": 25, "test": 51}:
        raise RuntimeError("Official judged population mismatch")
    if {split: len(v) for split, v in answerable.items()} != {"train": 148, "dev": 21, "test": 44}:
        raise RuntimeError("Answerable population mismatch")
    if "504" in qrels["test"] or "504" in answerable["test"]:
        raise RuntimeError("qid 504 was improperly classified in qrels")

    scoring_runs = {}
    filtering = []
    for population, qrel_map in [("Official judged", qrels), ("Answerable-only", answerable)]:
        scoring_runs[population] = {}
        pop_slug = "Official_Judged" if population == "Official judged" else "Answerable_Only"
        for split in ["train", "dev", "test"]:
            destination = output / "Scoring_Runs" / pop_slug / split / filenames[split]
            stats = filter_run_exact(official_runs[split], destination, set(qrel_map[split]))
            if set(parse_trec(destination)) != set(qrel_map[split]):
                raise RuntimeError(f"Scoring-run membership mismatch: {population} {split}")
            stats.update({"population": population, "split": split, "source": str(official_runs[split]), "filtered": str(destination)})
            filtering.append(stats)
            scoring_runs[population][split] = destination

    results = []
    reproducibility = []
    crosschecks = []
    max_repeat_delta = 0.0
    max_crosscheck_delta = 0.0
    independent_records = {}
    for population, qrel_paths in [("Official judged", official_qrels), ("Answerable-only", answerable_qrels)]:
        for split in ["train", "dev", "test"]:
            run_path = scoring_runs[population][split]
            base = f"{population.replace('-', '_').replace(' ', '_')}_Dense_CSR_{split}"
            first_prefix = output / "Evaluator_Logs" / (base + "_pass1")
            second_prefix = output / "Evaluator_Logs" / (base + "_pass2")
            first = load_metrics(first_prefix.with_suffix(".metrics.tsv")) if first_prefix.with_suffix(".metrics.tsv").is_file() else score_official(run_path, qrel_paths[split], paths["official_scorer"], organizer, paths["adapter"].parent, first_prefix)
            second = load_metrics(second_prefix.with_suffix(".metrics.tsv")) if second_prefix.with_suffix(".metrics.tsv").is_file() else score_official(run_path, qrel_paths[split], paths["official_scorer"], organizer, paths["adapter"].parent, second_prefix)
            repeat_delta = max(abs(first["map_cut_10"] - second["map_cut_10"]), abs(first["recip_rank"] - second["recip_rank"]))
            max_repeat_delta = max(max_repeat_delta, repeat_delta)
            if repeat_delta > 1e-15:
                raise RuntimeError(f"Repeated scoring mismatch: {population} {split}")
            independent = score_independent(run_path, qrel_paths[split], paths["independent_evaluator"], output / "Evaluator_Logs" / "Independent_Crosschecks" / base)
            independent_records[(population, split)] = independent
            cross_delta = max(
                abs(first["map_cut_10"] - float(independent["overall"]["map_cut_10"])),
                abs(first["recip_rank"] - float(independent["overall"]["recip_rank"])),
            )
            max_crosscheck_delta = max(max_crosscheck_delta, cross_delta)
            if cross_delta > 1e-12:
                raise RuntimeError(f"Independent evaluator mismatch: {population} {split}: {cross_delta}")
            results.append({
                "Split": split, "System": "Dense E5 + CSR", "Condition_Role": "extension",
                "Population": population, "QID_Count": len(qrels[split] if population == "Official judged" else answerable[split]),
                "MAP_at_10": f"{first['map_cut_10']:.15f}", "MRR_at_10": f"{first['recip_rank']:.15f}",
                "Run_Path": str(run_path), "Qrels_Path": str(qrel_paths[split]),
            })
            reproducibility.append({"Population": population, "Split": split, "Repeat_Max_Delta": repeat_delta})
            crosschecks.append({"Population": population, "Split": split, "Official_MAP_at_10": first["map_cut_10"], "Independent_MAP_at_10": independent["overall"]["map_cut_10"], "Official_MRR_at_10": first["recip_rank"], "Independent_MRR_at_10": independent["overall"]["recip_rank"], "Maximum_Difference": cross_delta})

    fields = list(results[0])
    write_csv(output / "CSR_RESULTS_ALL_SPLITS.csv", fields, results)
    official_rows = [row for row in results if row["Population"] == "Official judged"]
    answerable_rows = [row for row in results if row["Population"] == "Answerable-only"]
    write_csv(output / "CSR_RESULTS_OFFICIAL_JUDGED.csv", fields, official_rows)
    write_csv(output / "CSR_RESULTS_ANSWERABLE_ONLY.csv", fields, answerable_rows)

    def result(population: str, split: str) -> dict:
        return next(row for row in results if row["Population"] == population and row["Split"] == split)

    csr_test = (
        float(result("Official judged", "test")["MAP_at_10"]),
        float(result("Official judged", "test")["MRR_at_10"]),
        float(result("Answerable-only", "test")["MAP_at_10"]),
        float(result("Answerable-only", "test")["MRR_at_10"]),
    )
    test_comparison = []
    metric_names = ["Official_MAP_at_10", "Official_MRR_at_10", "Answerable_MAP_at_10", "Answerable_MRR_at_10"]
    for system, values in list(CORE_TEST.items()) + [("Dense E5 + CSR", csr_test)]:
        test_comparison.append({"System": system, **{name: f"{value:.15f}" for name, value in zip(metric_names, values)}})
    write_csv(output / "CSR_TEST_COMPARISON.csv", list(test_comparison[0]), test_comparison)

    delta_rows = []
    for comparator in ["Dense E5", "BM25", "Fixed RALSR"]:
        row = {"Comparison": f"Dense E5 + CSR - {comparator}"}
        for name, csr_value, base_value in zip(metric_names, csr_test, CORE_TEST[comparator]):
            row[f"Delta_{name}"] = f"{csr_value - base_value:.15f}"
        delta_rows.append(row)
    write_csv(output / "CSR_TEST_DELTAS.csv", list(delta_rows[0]), delta_rows)

    official_test_independent = independent_records[("Official judged", "test")]
    test_run = parse_trec(scoring_runs["Official judged"]["test"])
    per_query = []
    for qid in sorted(qrels["test"], key=int):
        judgments = qrels["test"][qid]
        gold_no_answer = "-1" in judgments
        rows = test_run[qid]
        top_prediction = rows[0][1]
        relevant = {docid for docid, rel in judgments.items() if rel > 0 and docid != "-1"}
        top_relevant_rank = next((rank for rank, docid, _, _ in rows if docid in relevant), None)
        score = official_test_independent["per_qid"][qid]
        per_query.append({
            "QID": qid, "Gold_Status": "no-answer" if gold_no_answer else "answerable",
            "Gold_Relevant_Passage_Count": 1 if gold_no_answer else len(relevant),
            "AP_at_10": f"{float(score['map_cut_10']):.15f}", "Reciprocal_Rank": f"{float(score['recip_rank']):.15f}",
            "Top_Prediction": top_prediction, "Top_Relevant_Rank": "" if top_relevant_rank is None else top_relevant_rank,
            "Output_Type": "normal", "Answerable": str(not gold_no_answer),
        })
    write_csv(output / "PER_QUERY_TEST_CSR.csv", list(per_query[0]), per_query)
    if abs(sum(float(r["AP_at_10"]) for r in per_query) / 51 - csr_test[0]) > 1e-12:
        raise RuntimeError("Per-query MAP reconciliation failed")
    if abs(sum(float(r["Reciprocal_Rank"]) for r in per_query) / 51 - csr_test[1]) > 1e-12:
        raise RuntimeError("Per-query MRR reconciliation failed")

    dense_rows = {row["QID"]: row for row in csv.DictReader(paths["step7_dense_per_query"].open("r", encoding="utf-8", newline="")) if row["Gold_Status"] == "answerable"}
    csr_answerable = {row["QID"]: row for row in per_query if row["Gold_Status"] == "answerable"}
    if set(dense_rows) != set(csr_answerable) or len(csr_answerable) != 44:
        raise RuntimeError("CSR/Dense answerable-test qid mismatch")
    comparison_rows = []
    group_counts = Counter()
    improved, worsened, unchanged = [], [], []
    for qid in sorted(csr_answerable, key=int):
        csr_ap = float(csr_answerable[qid]["AP_at_10"])
        csr_rr = float(csr_answerable[qid]["Reciprocal_Rank"])
        dense_ap = float(dense_rows[qid]["AP_at_10"])
        dense_rr = float(dense_rows[qid]["Reciprocal_Rank"])
        csr_success, dense_success = csr_ap > 0, dense_ap > 0
        if csr_success and not dense_success:
            group = "CSR-only success"
        elif dense_success and not csr_success:
            group = "Dense-only success"
        elif csr_success and dense_success:
            group = "Both succeed"
        else:
            group = "Both fail"
        group_counts[group] += 1
        difference = csr_ap - dense_ap
        if difference > 1e-15:
            change = "improved"; improved.append(qid)
        elif difference < -1e-15:
            change = "worsened"; worsened.append(qid)
        else:
            change = "unchanged"; unchanged.append(qid)
        comparison_rows.append({
            "QID": qid, "CSR_AP_at_10": f"{csr_ap:.15f}", "Dense_AP_at_10": f"{dense_ap:.15f}",
            "AP_Difference_CSR_minus_Dense": f"{difference:.15f}", "CSR_Reciprocal_Rank": f"{csr_rr:.15f}",
            "Dense_Reciprocal_Rank": f"{dense_rr:.15f}", "RR_Difference_CSR_minus_Dense": f"{csr_rr-dense_rr:.15f}",
            "Outcome_Group": group, "AP_Change": change,
        })
    write_csv(output / "CSR_VS_DENSE_QUERY_COMPARISON.csv", list(comparison_rows[0]), comparison_rows)

    answerable_summary = [{
        "Answerable_Test_QIDs": 44,
        "AP_at_10_Positive": sum(float(row["AP_at_10"]) > 0 for row in csr_answerable.values()),
        "Reciprocal_Rank_Positive": sum(float(row["Reciprocal_Rank"]) > 0 for row in csr_answerable.values()),
        "Relevant_at_Rank_1": sum(str(row["Top_Relevant_Rank"]) == "1" for row in csr_answerable.values()),
        "Relevant_in_Top_10": sum(row["Top_Relevant_Rank"] != "" for row in csr_answerable.values()),
        "No_Relevant_in_Top_10": sum(row["Top_Relevant_Rank"] == "" for row in csr_answerable.values()),
        "CSR_Only_Success": group_counts["CSR-only success"], "Dense_Only_Success": group_counts["Dense-only success"],
        "Both_Succeed": group_counts["Both succeed"], "Both_Fail": group_counts["Both fail"],
        "CSR_AP_Improved_QIDs": len(improved), "CSR_AP_Worsened_QIDs": len(worsened), "CSR_AP_Unchanged_QIDs": len(unchanged),
    }]
    write_csv(output / "CSR_ANSWERABLE_TEST_SUMMARY.csv", list(answerable_summary[0]), answerable_summary)

    reproducibility_record = {
        "official_scoring_conditions": 6, "official_runs_per_condition": 2,
        "maximum_repeat_difference": max_repeat_delta, "repeat_passed": max_repeat_delta <= 1e-15,
        "independent_crosschecks": crosschecks, "maximum_independent_difference": max_crosscheck_delta,
        "independent_crosscheck_passed": max_crosscheck_delta <= 1e-12,
    }
    write_json(output / "CSR_SCORING_REPRODUCIBILITY.json", reproducibility_record)

    lineage = {
        "stage": "Step 9 - CSR Final Official Evaluation and Extension Result Freeze",
        "system": "Dense E5 + CSR",
        "configuration_freeze": {"path": str(config_path), "sha256": frozen_config_hash, "written_before_scoring": True},
        "source": {"path": str(paths["internal_run"]), "sha256": EXPECTED["internal_run"]},
        "official_runs": validation,
        "conversion": {"script": str(paths["converter"]), "sha256": EXPECTED["converter"], "score_mode": "rank-derived", "ranking_changed": False},
        "scoring": {"official_scorer": str(paths["official_scorer"]), "official_scorer_sha256": EXPECTED["official_scorer"], "independent_evaluator": str(paths["independent_evaluator"]), "independent_evaluator_sha256": EXPECTED["independent_evaluator"]},
        "populations": {"Official judged": {"train": 174, "dev": 25, "test": 51}, "Answerable-only": {"train": 148, "dev": 21, "test": 44}},
        "qid_504": "retained in complete test run; absent from published qrels and both scoring populations",
        "qrels_used_for_construction_or_retrieval": False,
        "retrieval_regenerated": False,
    }
    write_json(output / "CSR_EVALUATION_LINEAGE_MANIFEST.json", lineage)

    dense_delta = tuple(csr_test[i] - CORE_TEST["Dense E5"][i] for i in range(4))
    relation = "higher" if csr_test[0] > CORE_TEST["Dense E5"][0] else ("lower" if csr_test[0] < CORE_TEST["Dense E5"][0] else "equal")
    report = []
    report.append("# Dense E5 + CSR Final Evaluation Report\n\n")
    report.append("## Status\n\nThe pre-existing Dense E5 + CSR Top-100 artifact passed identity and structural checks and was frozen unchanged. No CSR construction, embedding, or retrieval operation was executed. The configuration record was written before any qrel was loaded for scoring.\n\n")
    report.append("## CSR identity\n\nCSR concatenates original text, authoritative roots, and compact Maqāyīs keywords in that order. Eligible definition tokens are normalized Arabic-only tokens of length at least three after the frozen stopword/formulaic filter. Query selection keeps at most five tokens per meaning and 20 overall; passage selection keeps at most three per meaning and 60 overall. First occurrence determines order after deduplication. Valid roots without exact Maqāyīs evidence remain as roots only, and unresolved query terms contribute no invented root or semantic content.\n\n")
    report.append("Dense E5 + CSR uses `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a`, `query:`/`passage:` prefixes, mean pooling, normalized 768-dimensional embeddings, batch size 16, and dot-product similarity. Rankings are similarity descending with Passage_ID ascending for exact ties.\n\n")
    report.append("## Internal and official-run validation\n\n")
    report.append(f"- Internal run: `{paths['internal_run']}`\n- SHA-256: `{EXPECTED['internal_run']}`\n- 25,100 rows, 251 qids, exactly depth 100, qid 504 present.\n- Invalid IDs, duplicate pairs, non-finite scores, rank/order/tie violations, and no-answer rows: 0.\n- Qrel independence: PASS.\n\n")
    report.append("| Split | Official rows | Qids | qid 504 | Strict validator | Organizer checker | Ranking mismatches | SHA-256 |\n|---|---:|---:|---:|---:|---:|---:|---|\n")
    for split in ["train", "dev", "test"]:
        v = validation[split]
        report.append(f"| {split} | {v['rows']} | {v['qids']} | {'yes' if v['qid_504_present'] else 'no'} | PASS | PASS | {v['ranking_mismatches']} | `{v['sha256']}` |\n")
    report.append("\n## Metrics\n\n| Split | Official MAP@10 | Official MRR@10 | Answerable-only MAP@10 | Answerable-only MRR@10 |\n|---|---:|---:|---:|---:|\n")
    for split in ["train", "dev", "test"]:
        oj, ao = result("Official judged", split), result("Answerable-only", split)
        report.append(f"| {split} | {float(oj['MAP_at_10']):.12f} | {float(oj['MRR_at_10']):.12f} | {float(ao['MAP_at_10']):.12f} | {float(ao['MRR_at_10']):.12f} |\n")
    report.append("\n## Held-out test comparison\n\n| System | Official MAP@10 | Official MRR@10 | Answerable MAP@10 | Answerable MRR@10 |\n|---|---:|---:|---:|---:|\n")
    for row in test_comparison:
        report.append(f"| {row['System']} | {float(row['Official_MAP_at_10']):.4f} | {float(row['Official_MRR_at_10']):.4f} | {float(row['Answerable_MAP_at_10']):.4f} | {float(row['Answerable_MRR_at_10']):.4f} |\n")
    report.append("\n## Test deltas\n\n| Comparison | Δ official MAP | Δ official MRR | Δ answerable MAP | Δ answerable MRR |\n|---|---:|---:|---:|---:|\n")
    for row in delta_rows:
        report.append(f"| {row['Comparison']} | {float(row['Delta_Official_MAP_at_10']):+.4f} | {float(row['Delta_Official_MRR_at_10']):+.4f} | {float(row['Delta_Answerable_MAP_at_10']):+.4f} | {float(row['Delta_Answerable_MRR_at_10']):+.4f} |\n")
    report.append("\n## Query-level Dense comparison\n\n")
    report.append(f"Among 44 answerable test qids, Dense E5 + CSR alone succeeds on {group_counts['CSR-only success']}, Dense E5 alone succeeds on {group_counts['Dense-only success']}, both succeed on {group_counts['Both succeed']}, and both fail on {group_counts['Both fail']}. CSR AP@10 is higher for {len(improved)} qids, lower for {len(worsened)}, and unchanged for {len(unchanged)}.\n\n")
    report.append("## No-answer behavior\n\nDense E5 + CSR has no abstention mechanism and emits no `-1` predictions. It returns ten normal passages for every qid in each complete official split file. Consequently, it receives no explicit abstention credit on gold no-answer qids. Test qid 504 remains in the complete test prediction file but is excluded from both scoring populations because it has no published qrel.\n\n")
    report.append("## Reproducibility\n\n")
    report.append(f"Each of six split/population conditions was scored twice with maximum difference `{max_repeat_delta}`. The independent evaluator cross-check had maximum difference `{max_crosscheck_delta}`.\n\n")
    report.append("## Thesis-ready results paragraph\n\n")
    report.append(f"The Dense E5 + CSR extension augmented each query and passage with the frozen compact representation while retaining the same multilingual-E5 encoder used by the dense baseline. On the held-out test judgments, it achieved MAP@10 of {csr_test[0]:.4f} and MRR@10 of {csr_test[1]:.4f}; on the secondary answerable-only population, it achieved {csr_test[2]:.4f} and {csr_test[3]:.4f}. Its official test MAP@10 was {relation} than Dense E5 by {dense_delta[0]:+.4f}, and its official MRR@10 differed by {dense_delta[1]:+.4f}.\n\n")
    report.append("## Thesis-ready discussion note\n\n")
    if dense_delta[0] > 0 and dense_delta[1] > 0:
        report.append("Under the frozen settings, compact root and Maqāyīs information added measurable value to the dense representation on both official test metrics. The query-level comparison nevertheless shows mixed cases, so the result supports incremental complementarity rather than a uniform benefit for every query.\n\n")
    elif dense_delta[0] < 0 and dense_delta[1] < 0:
        report.append("Under the frozen settings, adding compact root and Maqāyīs information did not improve the dense baseline on the held-out test metrics. The extra tokens may provide useful evidence for some queries, but the query-level gains were insufficient to offset cases where the expanded representation changed the embedding unfavorably. This is a descriptive interpretation; no post-test setting was changed.\n\n")
    else:
        report.append("The frozen extension produced a mixed result across MAP@10 and MRR@10. This indicates that compact lexical-semantic augmentation changed which relevant passages were retrieved or promoted without providing a uniform improvement across ranking objectives. No post-test setting was changed.\n\n")
    report.append("## Verdict\n\n**STEP 9 COMPLETE — CSR FINAL RESULTS FROZEN**\n")
    report_path = output / "CSR_FINAL_EVALUATION_REPORT.md"
    report_path.write_text("".join(report), encoding="utf-8")

    freeze = {
        "status": "frozen",
        "frozen_at_local": datetime.now().astimezone().isoformat(),
        "system": "Dense E5 + CSR",
        "configuration_sha256": frozen_config_hash,
        "source_sha256": EXPECTED["internal_run"],
        "official_runs": validation,
        "results": results,
        "test_comparison": test_comparison,
        "test_deltas": delta_rows,
        "query_level_summary": answerable_summary[0],
        "improved_qids": improved, "worsened_qids": worsened, "unchanged_qids": unchanged,
        "reproducibility": reproducibility_record,
        "integrity": {
            "retrieval_regenerated": False, "csr_representation_changed": False, "qrel_leakage": False,
            "source_run_unchanged": sha256(paths["internal_run"]) == EXPECTED["internal_run"],
            "query_csr_unchanged": sha256(paths["query_csr"]) == EXPECTED["query_csr"],
            "passage_csr_unchanged": sha256(paths["passage_csr"]) == EXPECTED["passage_csr"],
            "step7_core_results_unchanged": sha256(paths["step7_freeze"]) == EXPECTED["step7_freeze"],
            "baseline_unchanged": sha256(paths["baseline"]) == EXPECTED["baseline"],
        },
    }
    write_json(output / "CSR_RESULT_FREEZE_MANIFEST.json", freeze)

    checksum_lines = []
    for path in sorted((p for p in output.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"), key=lambda p: str(p.relative_to(output)).lower()):
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n")
    (output / "SHA256SUMS.txt").write_text("".join(checksum_lines), encoding="utf-8")

    print(json.dumps({
        "status": "STEP 9 COMPLETE — CSR FINAL RESULTS FROZEN",
        "configuration_sha256": frozen_config_hash,
        "internal": internal_validation,
        "official_runs": validation,
        "results": results,
        "csr_test": {name: value for name, value in zip(metric_names, csr_test)},
        "deltas_vs_dense": {name: value for name, value in zip(metric_names, dense_delta)},
        "query_level_groups": dict(group_counts),
        "improved_qids": improved, "worsened_qids": worsened, "unchanged_qids": unchanged,
        "max_repeat_delta": max_repeat_delta, "max_crosscheck_delta": max_crosscheck_delta,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
