#!/usr/bin/env python3
"""Training-selected, development-checked RALSR weight ablation.

This script changes only the six non-negative feature weights. It reads the
frozen split-safe feature table and normalization artifact, never reads test
qrels, never produces a test prediction, and never edits any upstream file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "Root_Coverage",
    "Root_Overlap",
    "Semantic_Overlap",
    "Jaccard_Root_Similarity",
    "Passage_Root_Count",
    "Semantic_Count",
]
NORM_COLUMNS = [f"{feature}_Norm" for feature in FEATURES]
FIXED_UNITS = np.array([6, 5, 4, 3, 1, 1], dtype=np.int16)
FIXED_WEIGHTS = FIXED_UNITS.astype(np.float64) / 20.0
EXPECTED_TRAIN_ZERO = {"102", "108", "110", "137", "141", "143", "212", "235", "252", "258"}
EXPECTED_DEV_ZERO = {"234"}
EXPECTED_SOURCE_HASHES = {
    "feature_table": "7953EA8C5FA5AA85F1B9E3C1A0856463C1C72390F101AAF541DFACCF1092EAF3",
    "normalization": "B59BDCC6E29AB973266C0623841FDAFAB5D765D4BFCDB1F1825875BD6C57A851",
    "split_mapping": "A94DB69216F61D36E2E93730DE0DF5904E66C71F78B674E8D7539A73CA9A699A",
    "fixed_source": "7B70FBEACBF4E1AB11888CCB1798E9C48349C2977795F871601866A1D04D4D6C",
    "train_questions": "3B707F9A857A668B71A2DADEA9AA6545F038E656F6092E6D78EC0C4C99A3A0A9",
    "dev_questions": "BAAE0A5DFE7ADC3E300DC79F504527897E84C2E3DACB19789638ECB29E97229B",
    "train_qrels": "48E64E24A715BD77B824D9A4C17863C39614FE198A0A44FED23E76A6BEA0CFE6",
    "dev_qrels": "6F74218F1259AA144C795F19D0E4FB99095F927511EE708FA7EE4125F37A3590",
    "fixed_step3_train": "8914022F74E143B5B3FB0187F3B444D551CEFBF7B8926AEEF16D5E7E69A83124",
    "fixed_step3_dev": "7BFC6E2A33B080D81AE4F80C7BFD710D378E0A761147BEEF0F5D30D4C8C4A883",
    "fixed_step3_test": "6455AFF49056D7DC92901B70880E9CB0D587AA6DA6C261F5CBD00F31218CC019",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_question_ids(path: Path) -> list[str]:
    result: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row and str(row[0]).strip():
                result.append(str(row[0]).strip())
    return result


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    # The test-qrel file is intentionally neither named nor discoverable here.
    if "test" in path.name.lower():
        raise RuntimeError("Test-set firewall: refusing a test qrel path")
    result: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) != 4:
                raise ValueError(f"Malformed qrel row in {path}: {row}")
            qid, _, docid, relevance = row
            result[str(qid).strip()][str(docid).strip()] = int(relevance)
    return dict(result)


def read_qpc_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if row:
                result.add(str(row[0]).strip())
    return result


def compositions(total: int, parts: int, prefix: tuple[int, ...] = ()):
    if parts == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from compositions(total - value, parts - 1, prefix + (value,))


def load_features(path: Path, train_qids: set[str], dev_qids: set[str]) -> pd.DataFrame:
    usecols = ["Question_ID", "Passage_ID", "Split"] + NORM_COLUMNS
    wanted = train_qids | dev_qids
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"Question_ID": "string", "Passage_ID": "string", "Split": "string"},
        chunksize=10_000,
    ):
        chunk["Question_ID"] = chunk["Question_ID"].str.strip()
        chunk["Passage_ID"] = chunk["Passage_ID"].str.strip()
        selected = chunk.loc[chunk["Question_ID"].isin(wanted)].copy()
        if not selected.empty:
            chunks.append(selected)
    result = pd.concat(chunks, ignore_index=True)
    if result.duplicated(["Question_ID", "Passage_ID"]).any():
        raise RuntimeError("Duplicate candidate pairs in frozen feature input")
    if result[NORM_COLUMNS].isna().any().any():
        raise RuntimeError("Missing normalized features")
    expected_split = result["Question_ID"].map(
        lambda qid: "train" if qid in train_qids else "dev"
    )
    if not (result["Split"].str.lower() == expected_split).all():
        raise RuntimeError("Frozen feature Split column disagrees with official question files")
    return result


def prepare_groups(frame: pd.DataFrame, qids: list[str]):
    groups: dict[str, tuple[list[str], np.ndarray]] = {}
    for qid in qids:
        group = frame.loc[frame["Question_ID"] == qid, ["Passage_ID"] + NORM_COLUMNS]
        if group.empty:
            continue
        group = group.sort_values("Passage_ID", kind="stable")
        docids = group["Passage_ID"].astype(str).tolist()
        matrix = group[NORM_COLUMNS].to_numpy(dtype=np.float64, copy=True)
        groups[qid] = (docids, matrix)
    return groups


def top10_order(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return exact frozen top-10 order for one or more weight vectors.

    Candidate rows are already sorted by Passage_ID ascending. Scores are
    rounded to the established six decimals before ranking. A sub-micro-score
    deterministic offset makes Passage_ID order explicit for ties, allowing
    partial selection without changing any non-tied six-decimal score order.
    """
    scores = np.round(matrix @ weights.T, 6)
    depth = min(10, len(matrix))
    tie_offset = (len(matrix) - np.arange(len(matrix), dtype=np.float64)) * 1e-12
    adjusted = scores + tie_offset[:, None]
    if len(matrix) <= depth:
        return np.argsort(-adjusted, axis=0, kind="stable")[:depth, :]
    selected = np.argpartition(-adjusted, kth=depth - 1, axis=0)[:depth, :]
    selected_scores = np.take_along_axis(adjusted, selected, axis=0)
    within = np.argsort(-selected_scores, axis=0, kind="stable")
    return np.take_along_axis(selected, within, axis=0)


def evaluate_grid(
    units: np.ndarray,
    qids: list[str],
    groups,
    qrels: dict[str, dict[str, int]],
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    if set(qids) != set(qrels):
        raise RuntimeError("Official question qids and qrel qids do not match")
    n_config = len(units)
    maps = np.zeros(n_config, dtype=np.float64)
    mrrs = np.zeros(n_config, dtype=np.float64)
    constant_map = 0.0
    constant_mrr = 0.0
    scored_groups = []
    for qid in qids:
        judgments = qrels[qid]
        is_zero_answer = "-1" in judgments
        if qid not in groups:
            if is_zero_answer:
                constant_map += 1.0
                constant_mrr += 1.0
            continue
        if is_zero_answer:
            # Candidate-bearing questions never abstain under the frozen rule.
            continue
        relevant = {docid for docid, rel in judgments.items() if rel > 0 and docid != "-1"}
        docids, matrix = groups[qid]
        rel_vector = np.array([docid in relevant for docid in docids], dtype=np.bool_)
        scored_groups.append((matrix, rel_vector, len(relevant)))

    for start in range(0, n_config, batch_size):
        stop = min(start + batch_size, n_config)
        weights = units[start:stop].astype(np.float64) / 20.0
        map_sum = np.full(stop - start, constant_map, dtype=np.float64)
        mrr_sum = np.full(stop - start, constant_mrr, dtype=np.float64)
        for matrix, rel_vector, relevant_count in scored_groups:
            order = top10_order(matrix, weights)
            hits = rel_vector[order]
            cumulative = np.cumsum(hits, axis=0)
            precision = cumulative / np.arange(1, hits.shape[0] + 1, dtype=np.float64)[:, None]
            if relevant_count:
                map_sum += (precision * hits).sum(axis=0) / relevant_count
            has_hit = hits.any(axis=0)
            first_hit = np.argmax(hits, axis=0)
            mrr_sum += np.where(has_hit, 1.0 / (first_hit + 1), 0.0)
        maps[start:stop] = map_sum / len(qids)
        mrrs[start:stop] = mrr_sum / len(qids)
        if start == 0 or stop == n_config or (start // batch_size) % 10 == 0:
            print(f"evaluated {stop:,}/{n_config:,} configurations", flush=True)
    return maps, mrrs


def evaluate_one(weights, qids, groups, qrels):
    units = np.rint(np.asarray(weights, dtype=np.float64) * 20).astype(np.int16)[None, :]
    maps, mrrs = evaluate_grid(units, qids, groups, qrels, batch_size=1)
    return float(maps[0]), float(mrrs[0])


def rank_rows(qids: list[str], groups, weights: np.ndarray, tag: str):
    rows = []
    sequences = {}
    for qid in qids:
        if qid not in groups:
            rows.append([qid, "Q0", "-1", 1, "1", tag])
            sequences[qid] = ["-1"]
            continue
        docids, matrix = groups[qid]
        scores = np.round(matrix @ weights, 6)
        order = np.argsort(-scores, kind="stable")[:10]
        ranked_docids = [docids[int(index)] for index in order]
        sequences[qid] = ranked_docids
        for rank, docid in enumerate(ranked_docids, 1):
            # Rank-derived evaluator score protects the already-selected order.
            rows.append([qid, "Q0", docid, rank, f"{11-rank:.6f}", tag])
    return rows, sequences


def write_trec(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def read_trec_sequences(path: Path):
    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            grouped[str(row[0])].append((int(row[3]), str(row[2])))
    return {qid: [docid for _, docid in sorted(rows)] for qid, rows in grouped.items()}


def compare_sequences(left, right):
    all_qids = set(left) | set(right)
    return [qid for qid in sorted(all_qids, key=int) if left.get(qid) != right.get(qid)]


def run_command(command, log_path: Path, env=None):
    completed = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace", env=env)
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(map(str, command))}")
    return completed


def validate_run(
    run_path: Path,
    question_path: Path,
    qpc_path: Path,
    strict_validator: Path,
    official_checker: Path,
    logs_dir: Path,
):
    strict_json = logs_dir / f"{run_path.name}.strict.json"
    strict = run_command(
        [sys.executable, str(strict_validator), "--run", str(run_path), "--questions", str(question_path), "--qpc", str(qpc_path), "--json-report", str(strict_json)],
        logs_dir / f"{run_path.name}.strict.log",
    )
    strict_result = json.loads(strict_json.read_text(encoding="utf-8"))
    if not strict_result.get("passed"):
        raise RuntimeError(f"Strict validation failed for {run_path.name}")
    checker = run_command(
        [sys.executable, str(official_checker), "--model-prediction", str(run_path)],
        logs_dir / f"{run_path.name}.organizer_checker.log",
    )
    if "Format check: Passed" not in checker.stdout:
        raise RuntimeError(f"Organizer checker failed for {run_path.name}")
    return {"strict_validator_passed": True, "organizer_checker_passed": True, "strict_result": strict_result}


def run_official_scorer(
    run_path: Path,
    qrels_path: Path,
    official_scorer: Path,
    organizer_dir: Path,
    pytrec_dir: Path,
    output_prefix: Path,
):
    metric_output = output_prefix.with_suffix(".official_metrics.tsv")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(pytrec_dir), str(organizer_dir), env.get("PYTHONPATH", "")])
    run_command(
        [sys.executable, str(official_scorer), "--run", str(run_path), "--qrels", str(qrels_path), "--output", str(metric_output)],
        output_prefix.with_suffix(".official_scorer.log"),
        env=env,
    )
    metric_frame = pd.read_csv(metric_output, sep="\t")
    return {
        "map_cut_10": float(metric_frame.loc[0, "map_cut_10"]),
        "recip_rank": float(metric_frame.loc[0, "recip_rank"]),
        "metrics_file": str(metric_output),
    }


def structural_audit(path: Path, qids: list[str], zero_qids: set[str], qpc_ids: set[str]):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) != 6:
                raise RuntimeError(f"Six-column violation at {path}:{line_number}")
            qid, q0, docid, rank_text, score_text, tag = row
            rank = int(rank_text)
            score = float(score_text)
            if not math.isfinite(score):
                raise RuntimeError("Non-finite score")
            rows.append((qid, q0, docid, rank, score, tag))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[0]].append(row)
    errors = []
    if set(grouped) != set(qids):
        errors.append("qid membership mismatch")
    duplicate_pairs = len(rows) - len({(row[0], row[2]) for row in rows})
    invalid_ids = sum(1 for row in rows if row[2] != "-1" and row[2] not in qpc_ids)
    for qid, qrows in grouped.items():
        qrows.sort(key=lambda row: row[3])
        if [row[3] for row in qrows] != list(range(1, len(qrows) + 1)):
            errors.append(f"rank continuity: {qid}")
        if any(qrows[i][4] < qrows[i + 1][4] for i in range(len(qrows) - 1)):
            errors.append(f"score order: {qid}")
        docids = [row[2] for row in qrows]
        if qid in zero_qids:
            if docids != ["-1"]:
                errors.append(f"no-answer structure: {qid}")
        elif "-1" in docids:
            errors.append(f"unexpected no-answer: {qid}")
    return {
        "rows": len(rows),
        "qids": len(grouped),
        "zero_answer_qids": sorted(zero_qids, key=int),
        "duplicate_pairs": duplicate_pairs,
        "invalid_qpc_ids": invalid_ids,
        "errors": errors,
        "passed": not errors and duplicate_pairs == 0 and invalid_ids == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--pytrec-dir", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = Path(args.output).resolve()
    pytrec_dir = Path(args.pytrec_dir).resolve()
    if output.exists() and any(path.is_file() for path in output.rglob("*")):
        raise SystemExit(f"Refusing to overwrite output folder containing files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logs_dir = output / "Validation_Logs"
    cross_dir = output / "Metric_Cross_Checks"
    sample_dir = cross_dir / "Sampled_Runs"
    scripts_dir = output / "Scripts"
    for directory in [logs_dir, cross_dir, sample_dir, scripts_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    codex = Path(r"${THESIS_CODEX_ROOT}")
    workspace = codex / "Master Thesis - Quranic Passage Retrieval - Writing Workspace"
    correction = codex / "Master Thesis - Quranic Passage Retrieval - Correction Control"
    ralsr_dir = workspace / "02_Methodology_Evidence" / "RALSR" / "Corrected_Outputs" / "RALSR_Split_Safe_Normalization_No_Answer"
    quranqa = workspace / "02_Methodology_Evidence" / "Data and Resources" / "QuranQA"
    official_pipeline = correction / "Official_QuranQA_TaskA_Evaluation"
    organizer_dir = official_pipeline / "01_Official_Organizer_Files"
    custom_dir = official_pipeline / "02_Evaluation_Pipeline"
    step3 = correction / "Core_Retrieval_Frozen_Runs"

    paths = {
        "feature_table": ralsr_dir / "RALSR_LEM_Features_Split_Safe.csv",
        "normalization": ralsr_dir / "RALSR_Train_Only_Normalization_Parameters.csv",
        "split_mapping": ralsr_dir / "QuranQA_Question_Split_Mapping.csv",
        "fixed_source": ralsr_dir / "RALSR_Top100_Split_Safe_With_No_Answer.csv",
        "train_questions": quranqa / "QQA23_TaskA_ayatec_v1.2_train.tsv",
        "dev_questions": quranqa / "QQA23_TaskA_ayatec_v1.2_dev.tsv",
        "train_qrels": quranqa / "QQA23_TaskA_ayatec_v1.2_qrels_train.gold",
        "dev_qrels": quranqa / "QQA23_TaskA_ayatec_v1.2_qrels_dev.gold",
        "qpc": quranqa / "QQA23_TaskA_QPC_v1.1.tsv",
        "fixed_step3_train": step3 / "Official_Top10" / "thesis_RALSRtr.tsv",
        "fixed_step3_dev": step3 / "Official_Top10" / "thesis_RALSRdv.tsv",
        "fixed_step3_test": step3 / "Official_Top10" / "thesis_RALSRte.tsv",
        "strict_validator": custom_dir / "quranqa_taskA_strict_validator.py",
        "official_checker": organizer_dir / "QQA23_TaskA_submission_checker.py",
        "official_scorer": organizer_dir / "QQA23_TaskA_eval.py",
    }
    for key, expected in EXPECTED_SOURCE_HASHES.items():
        actual = sha256(paths[key])
        if actual != expected:
            raise RuntimeError(f"Source hash mismatch for {key}: {actual} != {expected}")
    before_fixed = {key: sha256(paths[key]) for key in ["fixed_source", "fixed_step3_train", "fixed_step3_dev", "fixed_step3_test"]}

    train_qids = read_question_ids(paths["train_questions"])
    dev_qids = read_question_ids(paths["dev_questions"])
    if len(train_qids) != 174 or len(dev_qids) != 25 or set(train_qids) & set(dev_qids):
        raise RuntimeError("Official train/dev question membership failed")
    train_qrels = read_qrels(paths["train_qrels"])
    dev_qrels = read_qrels(paths["dev_qrels"])
    if set(train_qrels) != set(train_qids) or set(dev_qrels) != set(dev_qids):
        raise RuntimeError("Question/qrel membership mismatch")

    feature_frame = load_features(paths["feature_table"], set(train_qids), set(dev_qids))
    train_frame = feature_frame.loc[feature_frame["Question_ID"].isin(train_qids)].copy()
    dev_frame = feature_frame.loc[feature_frame["Question_ID"].isin(dev_qids)].copy()
    if len(train_frame) != 38_118 or len(dev_frame) != 5_335:
        raise RuntimeError(f"Candidate counts changed: train={len(train_frame)}, dev={len(dev_frame)}")
    train_groups = prepare_groups(train_frame, train_qids)
    dev_groups = prepare_groups(dev_frame, dev_qids)
    train_zero = set(train_qids) - set(train_groups)
    dev_zero = set(dev_qids) - set(dev_groups)
    if train_zero != EXPECTED_TRAIN_ZERO or dev_zero != EXPECTED_DEV_ZERO:
        raise RuntimeError(f"Structural no-answer set changed: train={train_zero}, dev={dev_zero}")

    grid_units = np.asarray(list(compositions(20, 6)), dtype=np.int16)
    if grid_units.shape != (53_130, 6) or len(np.unique(grid_units, axis=0)) != 53_130:
        raise RuntimeError(f"Invalid grid shape/uniqueness: {grid_units.shape}")
    fixed_match = np.flatnonzero(np.all(grid_units == FIXED_UNITS, axis=1))
    if len(fixed_match) != 1:
        raise RuntimeError("Fixed heuristic weight vector not uniquely present in grid")
    fixed_index = int(fixed_match[0])

    # Prove that the faster partial-selection kernel is exactly equivalent to
    # the original stable full sort for deterministic representative samples.
    kernel_sample_indices = [0, 1, 10_625, 26_564, 39_847, 53_129, fixed_index]
    kernel_weights = grid_units[kernel_sample_indices].astype(np.float64) / 20.0
    kernel_checks = 0
    for qid in list(train_groups)[:20] + list(dev_groups)[:10]:
        matrix = (train_groups.get(qid) or dev_groups[qid])[1]
        scores = np.round(matrix @ kernel_weights.T, 6)
        expected = np.argsort(-scores, axis=0, kind="stable")[:min(10, len(matrix)), :]
        actual = top10_order(matrix, kernel_weights)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"Optimized top-10 kernel changed frozen ranking for qid {qid}")
        kernel_checks += len(kernel_sample_indices)
    print(f"top-10 kernel equivalence checks passed: {kernel_checks}", flush=True)

    train_map, train_mrr = evaluate_grid(grid_units, train_qids, train_groups, train_qrels)
    l1_units = np.abs(grid_units - FIXED_UNITS).sum(axis=1)
    order = np.lexsort(
        (
            grid_units[:, 5], grid_units[:, 4], grid_units[:, 3],
            grid_units[:, 2], grid_units[:, 1], grid_units[:, 0],
            l1_units, -train_mrr, -train_map,
        )
    )
    selection_rank = np.empty(len(order), dtype=np.int32)
    selection_rank[order] = np.arange(1, len(order) + 1)
    selected_index = int(order[0])
    selected_units = grid_units[selected_index]
    selected_weights = selected_units.astype(np.float64) / 20.0

    fixed_train = (float(train_map[fixed_index]), float(train_mrr[fixed_index]))
    tuned_train = (float(train_map[selected_index]), float(train_mrr[selected_index]))
    fixed_dev = evaluate_one(FIXED_WEIGHTS, dev_qids, dev_groups, dev_qrels)
    tuned_dev = evaluate_one(selected_weights, dev_qids, dev_groups, dev_qrels)

    grid_frame = pd.DataFrame({
        "Configuration_ID": [f"G{i+1:05d}" for i in range(len(grid_units))],
        **{f"{feature}_Weight": grid_units[:, i] / 20.0 for i, feature in enumerate(FEATURES)},
        "Train_MAP_at_10": train_map,
        "Train_MRR_at_10": train_mrr,
        "L1_Distance_to_Fixed": l1_units / 20.0,
        "Selection_Rank": selection_rank,
    })
    grid_path = output / "TUNED_RALSR_GRID_RESULTS.csv"
    grid_frame.to_csv(grid_path, index=False, float_format="%.12f", encoding="utf-8")
    top_path = output / "TUNED_RALSR_TOP_CONFIGURATIONS.csv"
    grid_frame.iloc[order[:20]].to_csv(top_path, index=False, float_format="%.12f", encoding="utf-8")

    comparison_rows = []
    for condition, weights, train_metrics, dev_metrics in [
        ("Fixed heuristic RALSR", FIXED_WEIGHTS, fixed_train, fixed_dev),
        ("Training-selected tuned RALSR", selected_weights, tuned_train, tuned_dev),
    ]:
        comparison_rows.append({
            "Condition": condition,
            **{f"{feature}_Weight": float(weights[i]) for i, feature in enumerate(FEATURES)},
            "Train_MAP_at_10": train_metrics[0],
            "Train_MRR_at_10": train_metrics[1],
            "Dev_MAP_at_10": dev_metrics[0],
            "Dev_MRR_at_10": dev_metrics[1],
            "Train_MAP_Delta_vs_Fixed": train_metrics[0] - fixed_train[0],
            "Train_MRR_Delta_vs_Fixed": train_metrics[1] - fixed_train[1],
            "Dev_MAP_Delta_vs_Fixed": dev_metrics[0] - fixed_dev[0],
            "Dev_MRR_Delta_vs_Fixed": dev_metrics[1] - fixed_dev[1],
        })
    comparison_path = output / "TUNED_RALSR_TRAIN_DEV_COMPARISON.csv"
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False, float_format="%.12f", encoding="utf-8")

    train_rows, train_sequences = rank_rows(train_qids, train_groups, selected_weights, "RALSRtuned")
    dev_rows, dev_sequences = rank_rows(dev_qids, dev_groups, selected_weights, "RALSRtuned")
    tuned_train_path = output / "thesis_TunedTR.tsv"
    tuned_dev_path = output / "thesis_TunedDV.tsv"
    write_trec(tuned_train_path, train_rows)
    write_trec(tuned_dev_path, dev_rows)

    fixed_train_rows, fixed_train_sequences = rank_rows(train_qids, train_groups, FIXED_WEIGHTS, "RALSRFixed")
    fixed_dev_rows, fixed_dev_sequences = rank_rows(dev_qids, dev_groups, FIXED_WEIGHTS, "RALSRFixed")
    fixed_sequence_mismatches = {
        "train": compare_sequences(fixed_train_sequences, read_trec_sequences(paths["fixed_step3_train"])),
        "dev": compare_sequences(fixed_dev_sequences, read_trec_sequences(paths["fixed_step3_dev"])),
    }
    if fixed_sequence_mismatches["train"] or fixed_sequence_mismatches["dev"]:
        raise RuntimeError(f"Fixed feature ranking does not reproduce frozen fixed runs: {fixed_sequence_mismatches}")

    qpc_ids = read_qpc_ids(paths["qpc"])
    run_validation = {
        "train": {
            **structural_audit(tuned_train_path, train_qids, train_zero, qpc_ids),
            **validate_run(tuned_train_path, paths["train_questions"], paths["qpc"], paths["strict_validator"], paths["official_checker"], logs_dir),
        },
        "dev": {
            **structural_audit(tuned_dev_path, dev_qids, dev_zero, qpc_ids),
            **validate_run(tuned_dev_path, paths["dev_questions"], paths["qpc"], paths["strict_validator"], paths["official_checker"], logs_dir),
        },
    }
    if not run_validation["train"]["passed"] or not run_validation["dev"]["passed"]:
        raise RuntimeError("Tuned run structural validation failed")

    cross_checks = []
    def cross_check(label, run_path, qrels_path, optimized):
        official = run_official_scorer(
            run_path, qrels_path, paths["official_scorer"], organizer_dir, pytrec_dir,
            cross_dir / label,
        )
        delta_map = official["map_cut_10"] - optimized[0]
        delta_mrr = official["recip_rank"] - optimized[1]
        passed = abs(delta_map) <= 1e-12 and abs(delta_mrr) <= 1e-12
        record = {
            "label": label,
            "run": str(run_path),
            "optimized_map_cut_10": optimized[0],
            "optimized_recip_rank": optimized[1],
            "official_map_cut_10": official["map_cut_10"],
            "official_recip_rank": official["recip_rank"],
            "map_delta": delta_map,
            "mrr_delta": delta_mrr,
            "tolerance": 1e-12,
            "passed": passed,
            "official_metrics_file": official["metrics_file"],
        }
        cross_checks.append(record)
        if not passed:
            raise RuntimeError(f"Official metric cross-check failed: {record}")

    cross_check("fixed_train", paths["fixed_step3_train"], paths["train_qrels"], fixed_train)
    cross_check("fixed_dev", paths["fixed_step3_dev"], paths["dev_qrels"], fixed_dev)
    cross_check("tuned_train", tuned_train_path, paths["train_qrels"], tuned_train)
    cross_check("tuned_dev", tuned_dev_path, paths["dev_qrels"], tuned_dev)

    sample_indices = [0, 10_625, 26_564, 39_847, 53_129]
    used_samples = set()
    for index in sample_indices:
        if index in {fixed_index, selected_index} or index in used_samples:
            continue
        used_samples.add(index)
        sample_rows, _ = rank_rows(train_qids, train_groups, grid_units[index] / 20.0, "RALSRaudit")
        sample_path = sample_dir / f"audit_S{index+1:05d}.tsv"
        write_trec(sample_path, sample_rows)
        cross_check(
            f"sample_G{index+1:05d}_train",
            sample_path,
            paths["train_qrels"],
            (float(train_map[index]), float(train_mrr[index])),
        )

    # Repeat the selected ranking/output construction in memory and require an
    # exact match; the full grid is deterministic integer enumeration.
    repeat_train_rows, repeat_train_sequences = rank_rows(train_qids, train_groups, selected_weights, "RALSRtuned")
    repeat_dev_rows, repeat_dev_sequences = rank_rows(dev_qids, dev_groups, selected_weights, "RALSRtuned")
    determinism = {
        "train_rows_identical": repeat_train_rows == train_rows,
        "dev_rows_identical": repeat_dev_rows == dev_rows,
        "train_sequences_identical": repeat_train_sequences == train_sequences,
        "dev_sequences_identical": repeat_dev_sequences == dev_sequences,
    }
    if not all(determinism.values()):
        raise RuntimeError("Selected tuned ranking is nondeterministic")

    weights_document = {
        "status": "frozen_for_future_final_evaluation",
        "condition_name": "training-selected tuned RALSR ablation",
        "feature_order": FEATURES,
        "selected_weights": {feature: float(selected_weights[i]) for i, feature in enumerate(FEATURES)},
        "selected_configuration_id": f"G{selected_index+1:05d}",
        "selected_training_rank": int(selection_rank[selected_index]),
        "grid": {"values": "0.00, 0.05, ..., 1.00", "constraint": "non-negative; sum exactly 1.00", "configurations": 53_130},
        "selection_rule": [
            "highest training MAP@10",
            "highest training MRR@10",
            "smallest L1 distance from fixed heuristic weights",
            "lexicographically smallest tuple in feature order",
        ],
        "metrics": {
            "tuned_train": {"map_cut_10": tuned_train[0], "recip_rank": tuned_train[1]},
            "tuned_dev": {"map_cut_10": tuned_dev[0], "recip_rank": tuned_dev[1]},
            "fixed_train": {"map_cut_10": fixed_train[0], "recip_rank": fixed_train[1]},
            "fixed_dev": {"map_cut_10": fixed_dev[0], "recip_rank": fixed_dev[1]},
        },
        "source_feature_table": {"path": str(paths["feature_table"]), "sha256": sha256(paths["feature_table"])},
        "normalization_artifact": {"path": str(paths["normalization"]), "sha256": sha256(paths["normalization"])},
        "frozen_at_local": datetime.now().astimezone().isoformat(),
        "test_qrels_accessed": False,
        "test_metrics_computed": False,
        "tuned_test_predictions_generated": False,
    }
    weights_path = output / "TUNED_RALSR_WEIGHTS.json"
    weights_path.write_text(json.dumps(weights_document, ensure_ascii=False, indent=2), encoding="utf-8")

    cross_path = output / "TUNED_RALSR_METRIC_CROSSCHECK.json"
    cross_path.write_text(json.dumps({"checks": cross_checks}, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_path = output / "TUNED_RALSR_RUN_VALIDATION.json"
    validation_path.write_text(json.dumps(run_validation, ensure_ascii=False, indent=2), encoding="utf-8")

    after_fixed = {key: sha256(paths[key]) for key in before_fixed}
    fixed_integrity = {key: before_fixed[key] == after_fixed[key] == EXPECTED_SOURCE_HASHES[key] for key in before_fixed}
    if not all(fixed_integrity.values()):
        raise RuntimeError(f"Frozen fixed RALSR integrity failed: {fixed_integrity}")

    outputs_for_lineage = [grid_path, top_path, comparison_path, tuned_train_path, tuned_dev_path, weights_path, cross_path, validation_path]
    lineage = {
        "experiment": "Tuned-RALSR weight-only ablation",
        "primary_system_unchanged": "fixed heuristic RALSR",
        "sources": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items() if path.is_file()},
        "candidate_pairs": {"train": len(train_frame), "dev": len(dev_frame)},
        "structural_no_answer_qids": {"train": sorted(train_zero, key=int), "dev": sorted(dev_zero, key=int)},
        "fixed_ranking_reproduction_mismatches": {key: len(value) for key, value in fixed_sequence_mismatches.items()},
        "fixed_integrity_after_experiment": fixed_integrity,
        "qrels_accessed": [str(paths["train_qrels"]), str(paths["dev_qrels"])],
        "test_qrels_accessed": False,
        "test_metrics_computed": False,
        "tuned_test_predictions_generated": False,
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs_for_lineage],
    }
    lineage_path = output / "TUNED_RALSR_LINEAGE_MANIFEST.json"
    lineage_path.write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "official_scorer": {"path": str(paths["official_scorer"]), "sha256": sha256(paths["official_scorer"])},
        "pytrec_eval_provider": "pytrec-eval-terrier 0.5.7 (isolated compatibility runtime exposing pytrec_eval)",
        "pytrec_eval_path": str(pytrec_dir),
    }
    environment_path = output / "TUNED_RALSR_ENVIRONMENT.json"
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__).resolve(), scripts_dir / Path(__file__).name)
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete",
        "grid_expected": 53_130,
        "grid_evaluated": len(grid_units),
        "fixed_configuration_id": f"G{fixed_index+1:05d}",
        "fixed_training_rank": int(selection_rank[fixed_index]),
        "fixed_weights": {feature: float(FIXED_WEIGHTS[i]) for i, feature in enumerate(FEATURES)},
        "selected_configuration_id": f"G{selected_index+1:05d}",
        "selected_weights": {feature: float(selected_weights[i]) for i, feature in enumerate(FEATURES)},
        "metrics": weights_document["metrics"],
        "deltas": {
            "train_map": tuned_train[0] - fixed_train[0],
            "train_mrr": tuned_train[1] - fixed_train[1],
            "dev_map": tuned_dev[0] - fixed_dev[0],
            "dev_mrr": tuned_dev[1] - fixed_dev[1],
        },
        "candidate_pairs": {"train": len(train_frame), "dev": len(dev_frame)},
        "zero_candidate_qids": {"train": sorted(train_zero, key=int), "dev": sorted(dev_zero, key=int)},
        "run_validation": {split: {key: value for key, value in result.items() if key != "strict_result"} for split, result in run_validation.items()},
        "official_metric_crosschecks": len(cross_checks),
        "official_metric_crosschecks_passed": all(item["passed"] for item in cross_checks),
        "determinism": determinism,
        "fixed_integrity": fixed_integrity,
        "elapsed_seconds": elapsed,
        "output_hashes": {
            path.name: sha256(path)
            for path in [grid_path, top_path, comparison_path, tuned_train_path, tuned_dev_path, weights_path, cross_path, validation_path, lineage_path, environment_path]
        },
        "test_firewall": {
            "test_qrels_accessed": False,
            "test_metrics_computed": False,
            "test_performance_inspected": False,
            "test_used_for_selection": False,
            "tuned_test_predictions_generated": False,
        },
    }
    summary_path = output / "TUNED_RALSR_ABLATION_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
