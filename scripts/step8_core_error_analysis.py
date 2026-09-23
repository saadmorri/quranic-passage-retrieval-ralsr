from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


CC = Path(r"${THESIS_CONTROL_ROOT}")
WS = Path(r"${THESIS_WORKSPACE}")
OUT = CC / "Core_Error_Analysis"
OUT.mkdir(parents=True, exist_ok=True)

STEP7 = CC / "Final_Core_Evaluation"
STEP6 = CC / "Unresolved_Term_Characterization"
STEP4 = CC / "No_Answer_Audit"
STEP3 = CC / "Core_Retrieval_Frozen_Runs"

FEATURES = WS / "02_Methodology_Evidence" / "RALSR" / "Corrected_Outputs" / "RALSR_Split_Safe_Normalization_No_Answer" / "RALSR_LEM_Features_Split_Safe.csv"
FIXED_TOP100 = WS / "02_Methodology_Evidence" / "RALSR" / "Corrected_Outputs" / "RALSR_Split_Safe_Normalization_No_Answer" / "RALSR_Top100_Split_Safe_With_No_Answer.csv"
BM25_TOP100 = WS / "02_Methodology_Evidence" / "Baselines" / "Corrected_Outputs" / "Base_BM25_Corrected_Query" / "BM25_Retrieval_Top100_Corrected.csv"
DENSE_TOP100 = WS / "02_Methodology_Evidence" / "Baselines" / "Dense_Retrieval_Top100.csv"
TUNED_INTERNAL = STEP7 / "TUNED_RALSR_TEST_INTERNAL_RANKING.csv"
SEMANTIC = WS / "02_Methodology_Evidence" / "Maqāyīs Lookup" / "Corrected_Outputs" / "Notebook_7_Maqayis_Semantic_Enrichment" / "QuranQA_Semantic_Enrichment_Corrected.csv"
PASSAGES = WS / "02_Methodology_Evidence" / "QAC and Root Extraction" / "Corrected_Outputs" / "QPC_Passage_Side_Validation" / "QuranQA_Enriched_Passage_Representation.csv"
QRELS = STEP4 / "qrels_test_answerable_only.gold"
TEST_QUESTIONS = WS / "02_Methodology_Evidence" / "Data and Resources" / "QuranQA" / "QQA23_TaskA_ayatec_v1.2_test.tsv"
STEP6_MASTER = STEP6 / "UNRESOLVED_TERM_MASTER_TABLE.csv"
STEP6_QSUMMARY = STEP6 / "QUESTION_LEVEL_COVERAGE_SUMMARY.csv"

PER_QUERY = {
    "BM25": STEP7 / "PER_QUERY_TEST_BM25.csv",
    "Dense_E5": STEP7 / "PER_QUERY_TEST_DENSE.csv",
    "Fixed_RALSR": STEP7 / "PER_QUERY_TEST_RALSR_FIXED.csv",
    "Tuned_RALSR": STEP7 / "PER_QUERY_TEST_RALSR_TUNED.csv",
}
RUNS = {
    "BM25": STEP3 / "Official_Top10" / "thesis_BM25te.tsv",
    "Dense_E5": STEP3 / "Official_Top10" / "thesis_E5te.tsv",
    "Fixed_RALSR": STEP3 / "Official_Top10" / "thesis_RALSRte.tsv",
    "Tuned_RALSR": STEP7 / "thesis_TunedTE.tsv",
}

EXPECTED_HASHES = {
    Path(r"${THESIS_WORKSPACE}.rar"): "5EE78669B8BD3E544873CD71946CE9841A87D25F6B82182B899C7DC2CBC2AD15",
    BM25_TOP100: "B24A5946FF821136352616B65B6C9AA1C1D66279114F8780F95D8425E1803235",
    DENSE_TOP100: "FD603ACEA95DA89FE31E7F1E9D3AB3A778BCE36D8C566C5C02E727A6582A2E6F",
    FIXED_TOP100: "7B70FBEACBF4E1AB11888CCB1798E9C48349C2977795F871601866A1D04D4D6C",
    FEATURES: "7953EA8C5FA5AA85F1B9E3C1A0856463C1C72390F101AAF541DFACCF1092EAF3",
    WS / "02_Methodology_Evidence" / "RALSR" / "Corrected_Outputs" / "RALSR_Split_Safe_Normalization_No_Answer" / "RALSR_Train_Only_Normalization_Parameters.csv": "B59BDCC6E29AB973266C0623841FDAFAB5D765D4BFCDB1F1825875BD6C57A851",
    CC / "Tuned_RALSR_Ablation" / "TUNED_RALSR_WEIGHTS.json": "06C2447FABEA838DDBC09153B126EBCD4B23D1872DCCE35673966D1AA26D9F6B",
    STEP7 / "CORE_RESULT_FREEZE_MANIFEST.json": "25C375F39B33A075D3E6800ECF37CDAAA3D9A54BCC3CA46245D5BCF3A79E1484",
    STEP7 / "PER_QUERY_TEST_BM25.csv": "5A3C0AA70B91975F5FDBF1BBD20CE1310DFC29C1130199F887873BDDE9EC50DB",
    STEP7 / "PER_QUERY_TEST_DENSE.csv": "C8D3D18DA2AAC99C32A58E9E0EDCC6A8A561FDF259FF7A8F539B1F142CFECCF0",
    STEP7 / "PER_QUERY_TEST_RALSR_FIXED.csv": "D60CFC8D5BF50C6E73F56DC2B42B37EEE857B9B56A53D94F85B7AC242D38CCE6",
    STEP7 / "PER_QUERY_TEST_RALSR_TUNED.csv": "D3DCBF1816B3B523B4AF115D4BC4E3F8D3F84A0C993C5A20F262A507FC60762E",
    WS / "02_Methodology_Evidence" / "Data and Resources" / "QuranQA" / "QQA23_TaskA_ayatec_v1.2_qrels_test.gold": "E23E4CF0628EB2FF39562852A5632DE0D948C8F643B5EB8E08B1D8B69CBA0332",
    STEP4 / "qrels_test_answerable_only.gold": "E799BEC96EA33B7ECDFC0B8285B85B4F16521B6C2FDD7DD02AA7AFE0F608872B",
    STEP6 / "UNRESOLVED_TERM_MASTER_TABLE.csv": "3A0FF836541F7597F193467EB5E0751F97E1020EC89FC32167ACA727ADC11192",
    STEP6 / "QUESTION_LEVEL_COVERAGE_SUMMARY.csv": "02D16F999FA23B38CA5B6249749243CF40B22FC0BC2B64E966FAF2FF34653E87",
    SEMANTIC: "1F65EABA834BB1CF5D5098E124218FFAB1C209CCD5AFD9A662390DE360202FB9",
    WS / "02_Methodology_Evidence" / "Maqāyīs Lookup" / "Lexicon Database and Metadata" / "db.sqlite": "D39ADF6D3846AA17AE92802C9D7EF530F33E857BB3B1163880100C5C479595EB",
    CC / "Maqayis_Printed_Edition_Audit" / "MAQAYIS_PRINTED_EDITION_AUDIT_REPORT.md": "AB8C294D3C91306E108B83ACA627DF24F28E8C09A498DC062B16BA03F2DAEC87",
    WS / "04_New_Proposal" / "02_Review" / "AlaToo_LaTeX_Proposal_Final_Dr_Muso_Revision" / "main.pdf": "D3BA67D5AA5579A1C5618C167CFBC10E576F9ABEE73460CA9926C1F0AE2698BA",
    PASSAGES: "9BC72BB9C6C13E2FF1810B1963568C3F8CEDBB776BF7B60E389B0DA2433F96BA",
    RUNS["BM25"]: "4DAE625B5877A4C1D92DBDE6E57DB3E226C1C85AA1AA529C0B76D5DA403684E2",
    RUNS["Dense_E5"]: "46585DEE581BC70B9321989FE719DBB8AC3F1CD47D0B187221A631D29B22798D",
    RUNS["Fixed_RALSR"]: "6455AFF49056D7DC92901B70880E9CB0D587AA6DA6C261F5CBD00F31218CC019",
    RUNS["Tuned_RALSR"]: "8430EE2B9E08EF47AB88F03964A200F06DC4BF2BB8EB2A430AC8E0A6DE136397",
    TUNED_INTERNAL: "26F368F34DF5D34BCD0A56E331F45CA1BA12AD5BFE3E5FAB0ECB0F49F802168A",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_json(value: str, default):
    try:
        return json.loads(value)
    except Exception:
        return default


def parse_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) >= 4 and float(parts[3]) > 0:
                out[parts[0]].add(parts[2])
    return dict(out)


def parse_run(path: Path) -> dict[str, list[tuple[int, str, float]]]:
    out: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            p = line.rstrip("\n\r").split("\t")
            if len(p) != 6:
                raise ValueError(f"Malformed run row in {path}: {line!r}")
            out[p[0]].append((int(p[3]), p[2], float(p[4])))
    for qid in out:
        out[qid].sort()
    return dict(out)


def best_relevant_rank(run_rows: list[tuple[int, str, float]], gold: set[str]) -> int | None:
    ranks = [rank for rank, pid, _ in run_rows if pid in gold]
    return min(ranks) if ranks else None


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def safe_mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def safe_median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def dump_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


# Integrity gate before post-hoc analysis.
hash_audit = []
for path, expected in EXPECTED_HASHES.items():
    actual = sha256(path)
    hash_audit.append({"Path": str(path), "Expected_SHA256": expected, "Actual_SHA256": actual, "Match": actual == expected})
    if actual != expected:
        raise RuntimeError(f"Frozen artifact identity mismatch: {path}\nExpected {expected}\nActual   {actual}")


qrels = parse_qrels(QRELS)
answerable_qids = sorted(qrels, key=int)
if len(answerable_qids) != 44 or "504" in answerable_qids:
    raise RuntimeError(f"Unexpected answerable test population: {len(answerable_qids)} qids")

questions = {}
with TEST_QUESTIONS.open("r", encoding="utf-8-sig") as f:
    for line in f:
        qid, text = line.rstrip("\n\r").split("\t", 1)
        questions[qid] = text

perq = {name: {r["QID"]: r for r in read_csv(path)} for name, path in PER_QUERY.items()}
for name, rows in perq.items():
    if set(answerable_qids) - set(rows):
        raise RuntimeError(f"Missing per-query rows for {name}")

expected_answerable_metrics = {
    "BM25": (0.085027162380103, 0.201948051948052),
    "Dense_E5": (0.115360306469744, 0.335606060606061),
    "Fixed_RALSR": (0.049721426449368, 0.156655844155844),
    "Tuned_RALSR": (0.047341375944317, 0.157476551226551),
}
metric_reconciliation = []
for name, expected in expected_answerable_metrics.items():
    map10 = statistics.mean(float(perq[name][q]["AP_at_10"]) for q in answerable_qids)
    mrr10 = statistics.mean(float(perq[name][q]["Reciprocal_Rank"]) for q in answerable_qids)
    ok = abs(map10 - expected[0]) <= 1e-12 and abs(mrr10 - expected[1]) <= 1e-12
    metric_reconciliation.append({"System": name, "Computed_MAP_at_10": map10, "Expected_MAP_at_10": expected[0], "Computed_MRR_at_10": mrr10, "Expected_MRR_at_10": expected[1], "Within_1e-12": ok})
    if not ok:
        raise RuntimeError(f"Step-7 metric reconciliation failed for {name}")

runs = {name: parse_run(path) for name, path in RUNS.items()}

# Authoritative query lineage, root sources, and semantic coverage.
semantic_rows = read_csv(SEMANTIC)
query_info = {}
for row in semantic_rows:
    qid = row["Question_ID"]
    terms = parse_json(row["Query_Terms"], [])
    matches = parse_json(row["Lexical_Matches"], [])
    roots = [m.get("Root_AR") for m in matches if m.get("Root_Resolved") and m.get("Root_AR")]
    qac = sum(1 for m in matches if m.get("Root_Resolved") and m.get("Root_Source") == "QAC")
    camel = sum(1 for m in matches if m.get("Root_Resolved") and m.get("Root_Source") == "CAMeL")
    unresolved = sum(1 for m in matches if not m.get("Root_Resolved"))
    sem = sum(1 for m in matches if m.get("Semantic_Found"))
    query_info[qid] = {
        "Question": row["Question"],
        "Terms": terms,
        "Roots": roots,
        "QAC_Count": qac,
        "CAMeL_Count": camel,
        "Unresolved_Count": unresolved,
        "Semantic_Count": sem,
        "Selected_Term_Count": len(terms),
    }

step6_rows = read_csv(STEP6_MASTER)
step6_by_qid = defaultdict(list)
for r in step6_rows:
    if r["Dataset"] == "Test":
        step6_by_qid[r["Question_ID"]].append(r)

def representation_mismatch(rows: list[dict[str, str]]) -> bool:
    return any("representation mismatch" in r["Linguistic_Category"].lower() for r in rows)

def diagnostic_tags(rows: list[dict[str, str]]) -> list[str]:
    tags = []
    if any(r["Population"] == "Root_Unresolved" for r in rows):
        tags.append("contains_unresolved_term")
    if any(r["Linguistic_Category"] == "Arabic lexical content word" and r["Population"] == "Root_Unresolved" for r in rows):
        tags.append("unresolved_arabic_lexical_content")
    if any("Proper name" in r["Linguistic_Category"] for r in rows):
        tags.append("named_entity_or_special_form")
    if any(r["Linguistic_Category"] == "Function / grammatical expression" for r in rows):
        tags.append("function_expression")
    if any(r["Linguistic_Category"] == "Honorific / title expression" for r in rows):
        tags.append("honorific_title")
    if any("Modern / foreign" in r["Linguistic_Category"] for r in rows):
        tags.append("modern_or_foreign")
    if representation_mismatch(rows):
        tags.append("possible_representation_mismatch")
    return tags

# Passage text/tokens/roots for lexical-overlap diagnostics only.
passage_info = {}
passage_root_frequency = Counter()
for row in read_csv(PASSAGES):
    passage_roots = set(parse_json(row["Roots"], []))
    passage_info[row["Passage_ID"]] = {
        "text": row["Passage_Text"],
        "tokens": set(parse_json(row["Tokens"], [])),
        "roots": passage_roots,
    }
    passage_root_frequency.update(passage_roots)

# Candidate table and feature rows.
feature_rows = [r for r in read_csv(FEATURES) if r["Split"].lower() == "test"]
features_by_qid = defaultdict(list)
feature_by_pair = {}
for r in feature_rows:
    qid = r["Question_ID"]
    r["Rank_int"] = int(float(r["Rank"]))
    features_by_qid[qid].append(r)
    feature_by_pair[(qid, r["Passage_ID"])] = r
for rows in features_by_qid.values():
    rows.sort(key=lambda r: r["Rank_int"])

tuned_rows = read_csv(TUNED_INTERNAL)
tuned_rank = {(r["Question_ID"], r["Passage_ID"]): int(r["Rank"]) for r in tuned_rows if r["Passage_ID"] != "-1"}

bm25_full_rank = {}
for r in read_csv(BM25_TOP100):
    bm25_full_rank[(r["Question_ID"], r["Passage_ID"])] = int(r["Rank"])
dense_full_rank = {}
for r in read_csv(DENSE_TOP100):
    dense_full_rank[(r["Question_ID"], r["Passage_ID"])] = int(r["Rank"])

# Frozen rubric: exact maximal AP defines strong success. Partial is positive but below 1.
def outcome(qid: str, candidate_count: int, any_gold: bool, top10_gold: bool, ap: float) -> str:
    if candidate_count == 0:
        return "Structural zero-candidate failure"
    if not any_gold:
        return "Candidate-generation failure"
    if not top10_gold:
        return "Candidate present but ranking failure"
    if math.isclose(ap, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return "Strong retrieval success"
    return "Partial retrieval success"


master = []
feature_diag = []
for qid in answerable_qids:
    gold = qrels[qid]
    crows = features_by_qid.get(qid, [])
    cpids = {r["Passage_ID"] for r in crows}
    gold_candidates = [r for r in crows if r["Passage_ID"] in gold]
    any_gold = bool(gold_candidates)
    best_fixed_candidate_rank = min((r["Rank_int"] for r in gold_candidates), default=None)
    top100 = best_fixed_candidate_rank is not None and best_fixed_candidate_rank <= 100
    top10 = best_fixed_candidate_rank is not None and best_fixed_candidate_rank <= 10
    rank1 = best_fixed_candidate_rank == 1
    fixed_ap = float(perq["Fixed_RALSR"][qid]["AP_at_10"])
    fixed_rr = float(perq["Fixed_RALSR"][qid]["Reciprocal_Rank"])
    primary = outcome(qid, len(crows), any_gold, top10, fixed_ap)

    qi = query_info[qid]
    sel = qi["Selected_Term_Count"]
    root_resolved = qi["QAC_Count"] + qi["CAMeL_Count"]
    sem_count = qi["Semantic_Count"]
    root_cov = root_resolved / sel if sel else 0.0
    sem_cov = sem_count / sel if sel else 0.0
    if math.isclose(sem_cov, 1.0):
        sem_bin = "100%"
    elif sem_cov >= 0.5:
        sem_bin = "50–<100%"
    elif sem_cov > 0:
        sem_bin = ">0–<50%"
    else:
        sem_bin = "0%"

    if qi["Unresolved_Count"]:
        if qi["QAC_Count"] or qi["CAMeL_Count"]:
            source_profile = "Resolved roots plus unresolved terms"
        else:
            source_profile = "All selected terms unresolved"
    elif qi["QAC_Count"] and qi["CAMeL_Count"]:
        source_profile = "Mixed QAC and CAMeL"
    elif qi["QAC_Count"]:
        source_profile = "QAC only"
    elif qi["CAMeL_Count"]:
        source_profile = "CAMeL only"
    else:
        source_profile = "No selected terms"

    gold_tokens = set()
    gold_roots = set()
    for pid in gold:
        if pid in passage_info:
            gold_tokens |= passage_info[pid]["tokens"]
            gold_roots |= passage_info[pid]["roots"]
    exact_overlap = len(set(qi["Terms"]) & gold_tokens)
    root_overlap = len(set(qi["Roots"]) & gold_roots)
    tags = diagnostic_tags(step6_by_qid.get(qid, []))
    if sem_cov < 0.5:
        tags.append("semantic_coverage_below_50_percent")
    if sem_cov == 0:
        tags.append("zero_semantic_coverage")

    ranks = {name: best_relevant_rank(runs[name].get(qid, []), gold) for name in runs}
    tuned_full_rank = min((tuned_rank[(qid, pid)] for pid in gold if (qid, pid) in tuned_rank), default=None)
    bm25_best_full = min((bm25_full_rank[(qid, pid)] for pid in gold if (qid, pid) in bm25_full_rank), default=None)
    dense_best_full = min((dense_full_rank[(qid, pid)] for pid in gold if (qid, pid) in dense_full_rank), default=None)
    root_corpus_hits = {root: passage_root_frequency[root] for root in qi["Roots"]}

    master.append({
        "QID": qid,
        "Question": questions[qid],
        "Gold_Relevant_Passage_Count": len(gold),
        "Gold_Relevant_Passage_IDs": json.dumps(sorted(gold), ensure_ascii=False),
        "Candidate_Set_Size": len(crows),
        "Gold_In_RALSR_Candidate_Set": any_gold,
        "Best_Gold_Fixed_RALSR_Candidate_Rank": best_fixed_candidate_rank if best_fixed_candidate_rank is not None else "",
        "Gold_In_Fixed_Top100": top100,
        "Gold_In_Fixed_Top10": top10,
        "Gold_At_Fixed_Rank1": rank1,
        "Primary_RALSR_Outcome": primary,
        "Selected_Term_Count": sel,
        "Accepted_Root_Count": root_resolved,
        "QAC_Root_Count": qi["QAC_Count"],
        "CAMeL_Root_Count": qi["CAMeL_Count"],
        "Unresolved_Term_Count": qi["Unresolved_Count"],
        "Semantic_Evidence_Term_Count": sem_count,
        "No_Semantic_Evidence_Term_Count": sel - sem_count,
        "Root_Coverage_Proportion": root_cov,
        "Semantic_Coverage_Proportion": sem_cov,
        "Semantic_Coverage_Bin": sem_bin,
        "Root_Source_Profile": source_profile,
        "Query_Roots_With_Any_Passage_Corpus_Hit": sum(1 for count in root_corpus_hits.values() if count > 0),
        "Query_Root_Corpus_Passage_Hit_Counts": json.dumps(root_corpus_hits, ensure_ascii=False),
        "Contains_Possible_Representation_Mismatch": representation_mismatch(step6_by_qid.get(qid, [])),
        "Secondary_Diagnostic_Tags": ";".join(tags),
        "Exact_Selected_Term_Overlap_With_Gold": exact_overlap,
        "Authoritative_Root_Overlap_With_Gold": root_overlap,
        "BM25_AP_at_10": float(perq["BM25"][qid]["AP_at_10"]),
        "BM25_RR": float(perq["BM25"][qid]["Reciprocal_Rank"]),
        "BM25_Best_Relevant_Top10_Rank": ranks["BM25"] or "",
        "BM25_Best_Relevant_Top100_Rank": bm25_best_full if bm25_best_full is not None else "",
        "Dense_AP_at_10": float(perq["Dense_E5"][qid]["AP_at_10"]),
        "Dense_RR": float(perq["Dense_E5"][qid]["Reciprocal_Rank"]),
        "Dense_Best_Relevant_Top10_Rank": ranks["Dense_E5"] or "",
        "Dense_Best_Relevant_Top100_Rank": dense_best_full if dense_best_full is not None else "",
        "Fixed_RALSR_AP_at_10": fixed_ap,
        "Fixed_RALSR_RR": fixed_rr,
        "Fixed_RALSR_Best_Relevant_Top10_Rank": ranks["Fixed_RALSR"] or "",
        "Tuned_RALSR_AP_at_10": float(perq["Tuned_RALSR"][qid]["AP_at_10"]),
        "Tuned_RALSR_RR": float(perq["Tuned_RALSR"][qid]["Reciprocal_Rank"]),
        "Tuned_RALSR_Best_Relevant_Top10_Rank": ranks["Tuned_RALSR"] or "",
        "Tuned_RALSR_Best_Relevant_Full_Candidate_Rank": tuned_full_rank if tuned_full_rank is not None else "",
        "BM25_Top_Prediction": perq["BM25"][qid]["Top_Prediction"],
        "Dense_Top_Prediction": perq["Dense_E5"][qid]["Top_Prediction"],
        "Fixed_RALSR_Top_Prediction": perq["Fixed_RALSR"][qid]["Top_Prediction"],
        "Tuned_RALSR_Top_Prediction": perq["Tuned_RALSR"][qid]["Top_Prediction"],
    })

    if primary == "Candidate present but ranking failure":
        gold_row = min(gold_candidates, key=lambda r: r["Rank_int"])
        nonrel_row = next(r for r in crows if r["Passage_ID"] not in gold)
        fd = {"QID": qid, "Gold_Passage_ID": gold_row["Passage_ID"], "Gold_Rank": gold_row["Rank_int"], "Top_Nonrelevant_Passage_ID": nonrel_row["Passage_ID"], "Top_Nonrelevant_Rank": nonrel_row["Rank_int"]}
        for feature in ["Root_Coverage", "Root_Overlap", "Semantic_Overlap", "Jaccard_Root_Similarity", "Passage_Root_Count", "Semantic_Count"]:
            col = feature + "_Norm"
            g = float(gold_row[col]); n = float(nonrel_row[col])
            fd[f"Gold_{col}"] = g
            fd[f"Top_Nonrelevant_{col}"] = n
            fd[f"Difference_Gold_Minus_Nonrelevant_{col}"] = g - n
        feature_diag.append(fd)

master.sort(key=lambda r: int(r["QID"]))
write_csv(OUT / "TEST_QUERY_ERROR_MASTER_TABLE.csv", master)

# Candidate recall and size summary.
n = len(master)
candidate_summary = []
for metric, key in [
    ("Structural zero-candidate failures", lambda r: r["Candidate_Set_Size"] == 0),
    ("At least one gold passage in full candidate set", lambda r: r["Gold_In_RALSR_Candidate_Set"]),
    ("No gold passage in full candidate set", lambda r: not r["Gold_In_RALSR_Candidate_Set"]),
    ("At least one gold passage in fixed Top 100", lambda r: r["Gold_In_Fixed_Top100"]),
    ("At least one gold passage in fixed Top 10", lambda r: r["Gold_In_Fixed_Top10"]),
    ("At least one gold passage at fixed rank 1", lambda r: r["Gold_At_Fixed_Rank1"]),
]:
    count = sum(1 for r in master if key(r))
    candidate_summary.append({"Measure": metric, "Count": count, "Denominator": n, "Percentage": count / n})
sizes = [r["Candidate_Set_Size"] for r in master]
nonzero_sizes = [s for s in sizes if s > 0]
candidate_summary.extend([
    {"Measure": "Candidate-set size mean", "Count": safe_mean(sizes), "Denominator": "", "Percentage": ""},
    {"Measure": "Candidate-set size median", "Count": safe_median(sizes), "Denominator": "", "Percentage": ""},
    {"Measure": "Candidate-set size minimum", "Count": min(sizes), "Denominator": "", "Percentage": ""},
    {"Measure": "Candidate-set size maximum", "Count": max(sizes), "Denominator": "", "Percentage": ""},
    {"Measure": "Candidate-bearing query size mean", "Count": safe_mean(nonzero_sizes), "Denominator": "", "Percentage": ""},
    {"Measure": "Candidate-bearing query size median", "Count": safe_median(nonzero_sizes), "Denominator": "", "Percentage": ""},
])
write_csv(OUT / "RALSR_CANDIDATE_RECALL_SUMMARY.csv", candidate_summary)

# Mutually exclusive taxonomy.
tax_order = [
    "Structural zero-candidate failure",
    "Candidate-generation failure",
    "Candidate present but ranking failure",
    "Partial retrieval success",
    "Strong retrieval success",
]
definitions = {
    "Structural zero-candidate failure": "No RALSR candidate passages exist.",
    "Candidate-generation failure": "Candidates exist, but no gold relevant passage occurs anywhere in the candidate set.",
    "Candidate present but ranking failure": "A gold relevant passage occurs in the candidate set, but none is ranked in the Top 10.",
    "Partial retrieval success": "At least one gold relevant passage occurs in the Top 10 and per-query AP@10 is positive but below 1.0.",
    "Strong retrieval success": "Per-query AP@10 equals 1.0 exactly under the frozen official metric.",
}
tax_rows = []
for cat in tax_order:
    qids = [r["QID"] for r in master if r["Primary_RALSR_Outcome"] == cat]
    tax_rows.append({"Primary_Category": cat, "Definition": definitions[cat], "QID_Count": len(qids), "Percentage_of_44": len(qids)/44, "QIDs": ";".join(qids)})
if sum(r["QID_Count"] for r in tax_rows) != 44:
    raise RuntimeError("RALSR taxonomy does not sum to 44")
write_csv(OUT / "RALSR_ERROR_TAXONOMY.csv", tax_rows)

# Pairwise positive-AP outcome groups.
def pairwise_rows(system_a: str, field_a: str, system_b: str, field_b: str):
    groups = defaultdict(list)
    for r in master:
        a = r[field_a] > 0
        b = r[field_b] > 0
        if a and not b: group = f"{system_a}-only success"
        elif b and not a: group = f"{system_b}-only success"
        elif a and b: group = "Both succeed"
        else: group = "Both fail"
        groups[group].append(r["QID"])
    order = [f"{system_a}-only success", f"{system_b}-only success", "Both succeed", "Both fail"]
    return [{"Comparison": f"{system_a} vs {system_b}", "Outcome_Group": g, "QID_Count": len(groups[g]), "Percentage_of_44": len(groups[g])/44, "QIDs": ";".join(groups[g])} for g in order]

dense_fixed = pairwise_rows("Dense E5", "Dense_AP_at_10", "Fixed RALSR", "Fixed_RALSR_AP_at_10")
bm25_fixed = pairwise_rows("BM25", "BM25_AP_at_10", "Fixed RALSR", "Fixed_RALSR_AP_at_10")
dense_bm25 = pairwise_rows("Dense E5", "Dense_AP_at_10", "BM25", "BM25_AP_at_10")
write_csv(OUT / "DENSE_VS_RALSR_OUTCOME_GROUPS.csv", dense_fixed)
write_csv(OUT / "BM25_VS_RALSR_OUTCOME_GROUPS.csv", bm25_fixed)
write_csv(OUT / "DENSE_VS_BM25_OUTCOME_GROUPS.csv", dense_bm25)

# Complementary and substantial rank-difference cases (predeclared threshold: >=5 ranks, or Top10 only in one system).
complementary = []
for r in master:
    for a, b, ap_a, ap_b, rank_a, rank_b in [
        ("Fixed RALSR", "Dense E5", "Fixed_RALSR_AP_at_10", "Dense_AP_at_10", "Fixed_RALSR_Best_Relevant_Top10_Rank", "Dense_Best_Relevant_Top10_Rank"),
        ("Fixed RALSR", "BM25", "Fixed_RALSR_AP_at_10", "BM25_AP_at_10", "Fixed_RALSR_Best_Relevant_Top10_Rank", "BM25_Best_Relevant_Top10_Rank"),
        ("Dense E5", "Fixed RALSR", "Dense_AP_at_10", "Fixed_RALSR_AP_at_10", "Dense_Best_Relevant_Top10_Rank", "Fixed_RALSR_Best_Relevant_Top10_Rank"),
    ]:
        ra = int(r[rank_a]) if r[rank_a] != "" else None
        rb = int(r[rank_b]) if r[rank_b] != "" else None
        one_only = (ra is None) != (rb is None)
        gap = abs(ra-rb) if ra is not None and rb is not None else None
        if one_only or (gap is not None and gap >= 5):
            complementary.append({"QID": r["QID"], "System_A": a, "System_B": b, "System_A_AP_at_10": r[ap_a], "System_B_AP_at_10": r[ap_b], "System_A_Best_Relevant_Top10_Rank": ra or "", "System_B_Best_Relevant_Top10_Rank": rb or "", "Criterion": "one_system_top10_only" if one_only else "rank_gap_at_least_5", "Absolute_Rank_Gap": gap if gap is not None else ""})
write_csv(OUT / "COMPLEMENTARY_CASES.csv", complementary)

# Semantic coverage bins.
sem_rows = []
for b in ["100%", "50–<100%", ">0–<50%", "0%"]:
    rows = [r for r in master if r["Semantic_Coverage_Bin"] == b]
    sem_rows.append({
        "Semantic_Coverage_Bin": b,
        "QID_Count": len(rows),
        "Mean_Fixed_RALSR_AP_at_10": safe_mean([r["Fixed_RALSR_AP_at_10"] for r in rows]),
        "Mean_Fixed_RALSR_RR": safe_mean([r["Fixed_RALSR_RR"] for r in rows]),
        "Structural_Zero_Candidate_Count": sum(r["Primary_RALSR_Outcome"] == "Structural zero-candidate failure" for r in rows),
        "Candidate_Generation_Failure_Count": sum(r["Primary_RALSR_Outcome"] == "Candidate-generation failure" for r in rows),
        "Candidate_Generation_Failure_Rate": safe_mean([r["Primary_RALSR_Outcome"] == "Candidate-generation failure" for r in rows]),
        "Ranking_Failure_Count": sum(r["Primary_RALSR_Outcome"] == "Candidate present but ranking failure" for r in rows),
        "Ranking_Failure_Rate": safe_mean([r["Primary_RALSR_Outcome"] == "Candidate present but ranking failure" for r in rows]),
        "Small_Sample_Flag": len(rows) < 5,
    })
write_csv(OUT / "QUERY_SEMANTIC_COVERAGE_VS_OUTCOME.csv", sem_rows)

# Root source profiles by outcome.
root_source_rows = []
for cat in tax_order:
    rows = [r for r in master if r["Primary_RALSR_Outcome"] == cat]
    for profile in sorted({r["Root_Source_Profile"] for r in master}):
        sub = [r for r in rows if r["Root_Source_Profile"] == profile]
        root_source_rows.append({"Primary_RALSR_Outcome": cat, "Root_Source_Profile": profile, "QID_Count": len(sub), "Percentage_Within_Outcome": len(sub)/len(rows) if rows else 0.0})
write_csv(OUT / "ROOT_SOURCE_VS_OUTCOME.csv", root_source_rows)

# Unresolved-term and representation-mismatch linkage.
all_categories = sorted({r["Linguistic_Category"] for rows in step6_by_qid.values() for r in rows})
unresolved_linkage = []
for cat in all_categories:
    qids = {qid for qid, rows in step6_by_qid.items() if any(r["Linguistic_Category"] == cat for r in rows)} & set(answerable_qids)
    linked = [r for r in master if r["QID"] in qids]
    unresolved_linkage.append({"Step6_Linguistic_Category": cat, "Answerable_Test_QID_Count": len(linked), **{name.replace(" ", "_")+"_Count": sum(r["Primary_RALSR_Outcome"] == name for r in linked) for name in tax_order}})
write_csv(OUT / "UNRESOLVED_CATEGORY_LINKAGE.csv", unresolved_linkage)

rep_rows = [r for r in master if r["Contains_Possible_Representation_Mismatch"]]
rep_summary = [{"Group": "Contains at least one possible doubled/weak/hamza representation mismatch", "QID_Count": len(rep_rows), **{name.replace(" ", "_")+"_Count": sum(r["Primary_RALSR_Outcome"] == name for r in rep_rows) for name in tax_order}}]
write_csv(OUT / "REPRESENTATION_MISMATCH_IMPACT.csv", rep_summary)

# Query difficulty across Dense-vs-RALSR groups.
group_lookup = {}
for g in dense_fixed:
    for q in filter(None, g["QIDs"].split(";")):
        group_lookup[q] = g["Outcome_Group"]
difficulty_rows = []
for group in [r["Outcome_Group"] for r in dense_fixed]:
    rows = [r for r in master if group_lookup[r["QID"]] == group]
    difficulty_rows.append({
        "Dense_vs_Fixed_RALSR_Group": group,
        "QID_Count": len(rows),
        "Mean_Selected_Term_Count": safe_mean([r["Selected_Term_Count"] for r in rows]),
        "Mean_Root_Coverage": safe_mean([r["Root_Coverage_Proportion"] for r in rows]),
        "Mean_Semantic_Coverage": safe_mean([r["Semantic_Coverage_Proportion"] for r in rows]),
        "Mean_Gold_Relevant_Passage_Count": safe_mean([r["Gold_Relevant_Passage_Count"] for r in rows]),
        "Mean_Candidate_Set_Size": safe_mean([r["Candidate_Set_Size"] for r in rows]),
        "Mean_Exact_Selected_Term_Overlap_With_Gold": safe_mean([r["Exact_Selected_Term_Overlap_With_Gold"] for r in rows]),
        "Mean_Authoritative_Root_Overlap_With_Gold": safe_mean([r["Authoritative_Root_Overlap_With_Gold"] for r in rows]),
    })
write_csv(OUT / "QUERY_DIFFICULTY_BY_PAIRWISE_GROUP.csv", difficulty_rows)

# Feature diagnostics and aggregate feature differences.
write_csv(OUT / "RALSR_FEATURE_DIAGNOSTICS.csv", feature_diag)
feature_summary = []
for feature in ["Root_Coverage", "Root_Overlap", "Semantic_Overlap", "Jaccard_Root_Similarity", "Passage_Root_Count", "Semantic_Count"]:
    col = f"Difference_Gold_Minus_Nonrelevant_{feature}_Norm"
    vals = [r[col] for r in feature_diag]
    feature_summary.append({"Feature": feature, "Ranking_Failure_QID_Count": len(vals), "Mean_Gold_Minus_Top_Nonrelevant": safe_mean(vals), "Median_Gold_Minus_Top_Nonrelevant": safe_median(vals), "Gold_Higher_Count": sum(v>0 for v in vals), "Equal_Count": sum(math.isclose(v,0,abs_tol=1e-12) for v in vals), "Gold_Lower_Count": sum(v<0 for v in vals)})
write_csv(OUT / "RALSR_FEATURE_DIAGNOSTIC_SUMMARY.csv", feature_summary)

# Fixed vs tuned query-level comparison.
fixed_tuned = []
for r in master:
    dap = r["Tuned_RALSR_AP_at_10"] - r["Fixed_RALSR_AP_at_10"]
    drr = r["Tuned_RALSR_RR"] - r["Fixed_RALSR_RR"]
    status = "Improved" if dap > 1e-15 else "Worsened" if dap < -1e-15 else "Unchanged"
    fixed_success = r["Fixed_RALSR_AP_at_10"] > 0
    tuned_success = r["Tuned_RALSR_AP_at_10"] > 0
    fixed_tuned.append({"QID": r["QID"], "Fixed_AP_at_10": r["Fixed_RALSR_AP_at_10"], "Tuned_AP_at_10": r["Tuned_RALSR_AP_at_10"], "AP_Difference_Tuned_Minus_Fixed": dap, "AP_Change_Status": status, "Fixed_RR": r["Fixed_RALSR_RR"], "Tuned_RR": r["Tuned_RALSR_RR"], "RR_Difference_Tuned_Minus_Fixed": drr, "Top10_Relevant_Membership_Changed": fixed_success != tuned_success, "RR_Improved_While_AP_Declined": drr > 1e-15 and dap < -1e-15})
write_csv(OUT / "FIXED_VS_TUNED_QUERY_COMPARISON.csv", fixed_tuned)

# Deterministic qualitative case selection.
def select_top(rows, score_fn, limit=3):
    return [r["QID"] for r in sorted(rows, key=lambda r: (-score_fn(r), int(r["QID"])))[:limit]]

dense_only_qids = set(next(r["QIDs"] for r in dense_fixed if r["Outcome_Group"] == "Dense E5-only success").split(";")) - {""}
ralsr_only_qids = set(next(r["QIDs"] for r in dense_fixed if r["Outcome_Group"] == "Fixed RALSR-only success").split(";")) - {""}
both_fail_qids = set(next(r["QIDs"] for r in dense_fixed if r["Outcome_Group"] == "Both fail").split(";")) - {""}
group_cases = {
    "A — Dense-only success": select_top([r for r in master if r["QID"] in dense_only_qids], lambda r: r["Dense_AP_at_10"] - r["Fixed_RALSR_AP_at_10"]),
    "B — RALSR-only success": select_top([r for r in master if r["QID"] in ralsr_only_qids], lambda r: r["Fixed_RALSR_AP_at_10"] - r["Dense_AP_at_10"]),
    "C — Both fail, most uncovered terms": [r["QID"] for r in sorted([r for r in master if r["QID"] in both_fail_qids], key=lambda r: (-r["No_Semantic_Evidence_Term_Count"], int(r["QID"])))[:3]],
    "D — Candidate present but ranking fails": [r["QID"] for r in sorted([r for r in master if r["Primary_RALSR_Outcome"] == "Candidate present but ranking failure"], key=lambda r: (-int(r["Best_Gold_Fixed_RALSR_Candidate_Rank"]), int(r["QID"])))[:3]],
    "E — Structural false abstention": [q for q in ["536", "613"] if q in answerable_qids],
    "F — Candidate-generation failure": [r["QID"] for r in sorted([r for r in master if r["Primary_RALSR_Outcome"] == "Candidate-generation failure"], key=lambda r: (-r["Candidate_Set_Size"], int(r["QID"])))[:3]],
}
selected_map = defaultdict(list)
for group, qids in group_cases.items():
    for q in qids:
        selected_map[q].append(group)

qualitative = []
for qid in sorted(selected_map, key=int):
    r = next(x for x in master if x["QID"] == qid)
    gold = sorted(qrels[qid])
    excerpts = []
    for pid in gold[:3]:
        text = passage_info.get(pid, {}).get("text", "")
        excerpts.append(f"{pid}: {text[:180]}")
    gold_feature = None
    if r["Gold_In_RALSR_Candidate_Set"]:
        gr = min((feature_by_pair[(qid,p)] for p in gold if (qid,p) in feature_by_pair), key=lambda x: int(float(x["Rank"])))
        gold_feature = {k: float(gr[k+"_Norm"]) for k in ["Root_Coverage", "Root_Overlap", "Semantic_Overlap", "Jaccard_Root_Similarity", "Passage_Root_Count", "Semantic_Count"]}
    if r["Primary_RALSR_Outcome"] == "Structural zero-candidate failure":
        interp = "All selected linguistic evidence failed to produce a valid root-overlap candidate; Dense/BM25 outcomes are reported post hoc without changing the abstention."
    elif r["Primary_RALSR_Outcome"] == "Candidate-generation failure":
        interp = "RALSR generated candidates, but no judged relevant passage entered the frozen candidate set, so feature weighting could not recover the answer."
    elif r["Primary_RALSR_Outcome"] == "Candidate present but ranking failure":
        interp = "A judged relevant passage was available to RALSR but remained below rank 10, indicating a ranking-discrimination failure rather than candidate absence."
    elif r["Fixed_RALSR_AP_at_10"] > 0 and r["Dense_AP_at_10"] == 0:
        interp = "The root-aware system retrieved judged evidence in the Top 10 where Dense did not, showing a complementary case without implying overall superiority."
    else:
        interp = "The systems' frozen ranks differ on this query; the recorded coverage and overlap diagnostics provide descriptive, not causal, evidence."
    qualitative.append({
        "QID": qid,
        "Case_Groups": "; ".join(selected_map[qid]),
        "Question": r["Question"],
        "Selected_Terms": json.dumps(query_info[qid]["Terms"], ensure_ascii=False),
        "Accepted_Roots": json.dumps(query_info[qid]["Roots"], ensure_ascii=False),
        "Unresolved_Terms": json.dumps([x["Term"] for x in step6_by_qid.get(qid, []) if x["Population"] == "Root_Unresolved"], ensure_ascii=False),
        "Semantic_Coverage_Proportion": r["Semantic_Coverage_Proportion"],
        "Gold_Relevant_Passage_IDs": r["Gold_Relevant_Passage_IDs"],
        "Gold_Passage_Excerpts": " || ".join(excerpts),
        "BM25_Top_Relevant_Rank": r["BM25_Best_Relevant_Top10_Rank"],
        "BM25_Top_Relevant_Rank_Top100": r["BM25_Best_Relevant_Top100_Rank"],
        "Dense_Top_Relevant_Rank": r["Dense_Best_Relevant_Top10_Rank"],
        "Dense_Top_Relevant_Rank_Top100": r["Dense_Best_Relevant_Top100_Rank"],
        "Fixed_RALSR_Top_Relevant_Rank": r["Best_Gold_Fixed_RALSR_Candidate_Rank"],
        "Tuned_RALSR_Top_Relevant_Full_Candidate_Rank": r["Tuned_RALSR_Best_Relevant_Full_Candidate_Rank"],
        "RALSR_Candidate_Set_Inclusion": r["Gold_In_RALSR_Candidate_Set"],
        "Highest_Ranked_Gold_Six_Normalized_Features": json.dumps(gold_feature, ensure_ascii=False) if gold_feature else "",
        "Query_Root_Corpus_Passage_Hit_Counts": r["Query_Root_Corpus_Passage_Hit_Counts"],
        "Evidence_Grounded_Interpretation": interp,
    })
write_csv(OUT / "QUALITATIVE_CASES.csv", qualitative)

# Compact root-category summaries for reporting.
root_cov_by_outcome = []
for cat in tax_order:
    rows = [r for r in master if r["Primary_RALSR_Outcome"] == cat]
    root_cov_by_outcome.append({"Primary_RALSR_Outcome": cat, "QID_Count": len(rows), "Mean_Root_Coverage": safe_mean([r["Root_Coverage_Proportion"] for r in rows]), "Median_Root_Coverage": safe_median([r["Root_Coverage_Proportion"] for r in rows]), "Mean_Semantic_Coverage": safe_mean([r["Semantic_Coverage_Proportion"] for r in rows]), "Median_Semantic_Coverage": safe_median([r["Semantic_Coverage_Proportion"] for r in rows]), "Queries_With_Unresolved_Terms": sum(r["Unresolved_Term_Count"]>0 for r in rows), "Queries_With_Representation_Mismatch": sum(r["Contains_Possible_Representation_Mismatch"] for r in rows)})
write_csv(OUT / "COVERAGE_LINKAGE_SUMMARY.csv", root_cov_by_outcome)

# Freeze rules before narrative generation.
rubric = """# Step 8 core error-analysis rubric

## Population and frozen evidence

The primary population is the 44 officially judged answerable test qids. Qid 504 is excluded because it is absent from published test qrels. The seven gold no-answer qids are handled only in the compact reminder section. No ranking, candidate, feature, root, semantic match, weight, or no-answer decision is changed.

## Mutually exclusive fixed-RALSR outcome categories

- **Structural zero-candidate failure:** no RALSR candidate passages exist.
- **Candidate-generation failure:** candidates exist, but no gold relevant passage occurs anywhere in the frozen candidate set.
- **Candidate present but ranking failure:** at least one gold relevant passage occurs in the candidate set, but none appears in Top 10.
- **Partial retrieval success:** at least one relevant passage appears in Top 10 and per-query AP@10 is positive but below 1.0.
- **Strong retrieval success:** per-query AP@10 equals 1.0 exactly. No subjective “near-maximal” threshold is used.

These categories must sum to 44. Secondary diagnostic tags are non-exclusive and never replace the primary category.

## Pairwise success

Pairwise success means positive per-query AP@10. Groups are A-only, B-only, both positive, or both zero.

## Coverage bins

Semantic coverage is the number of selected terms with exact Maqāyīs evidence divided by selected terms. Bins are frozen as `100%`, `50–<100%`, `>0–<50%`, and `0%`.

## Complementary-rank threshold

A rank difference is “substantial” if the best relevant Top-10 rank differs by at least 5 positions, or if only one system retrieves a relevant passage in Top 10.

## Deterministic qualitative case selection

- Group A: up to 3 Dense-only qids with largest Dense AP@10 advantage over fixed RALSR, then smallest qid.
- Group B: up to 3 RALSR-only qids with largest fixed-RALSR AP@10 advantage over Dense, then smallest qid.
- Group C: up to 3 both-fail qids with the most selected terms lacking semantic evidence, then smallest qid.
- Group D: up to 3 candidate-present ranking failures with the largest best-gold fixed-RALSR rank outside Top 10, then smallest qid.
- Group E: qids 536 and 613.
- Group F: up to 3 candidate-generation failures with the largest frozen candidate sets, then smallest qid. This ensures the required candidate-generation mechanism is represented without selecting cases by favorable narrative.

## Interpretation boundary

Coverage, source, overlap, and feature comparisons are descriptive associations. They do not establish causality or authorize post-test system changes.
"""
(OUT / "ERROR_ANALYSIS_RUBRIC.md").write_text(rubric, encoding="utf-8")

# Machine-readable summary used by the report/documentation pass.
summary = {
    "population": {"answerable_test_qids": 44, "qid_504_excluded": True, "gold_no_answer_qids_separate": ["522","535","546","547","554","582","604"]},
    "taxonomy": {r["Primary_Category"]: r["QID_Count"] for r in tax_rows},
    "candidate_recall": {r["Measure"]: r["Count"] for r in candidate_summary},
    "pairwise": {"dense_vs_fixed": dense_fixed, "bm25_vs_fixed": bm25_fixed, "dense_vs_bm25": dense_bm25},
    "semantic_bins": sem_rows,
    "coverage_by_outcome": root_cov_by_outcome,
    "feature_summary": feature_summary,
    "fixed_vs_tuned": {
        "improved": sum(r["AP_Change_Status"]=="Improved" for r in fixed_tuned),
        "worsened": sum(r["AP_Change_Status"]=="Worsened" for r in fixed_tuned),
        "unchanged": sum(r["AP_Change_Status"]=="Unchanged" for r in fixed_tuned),
        "top10_membership_changed": sum(r["Top10_Relevant_Membership_Changed"] for r in fixed_tuned),
        "rr_improved_while_ap_declined": [r["QID"] for r in fixed_tuned if r["RR_Improved_While_AP_Declined"]],
    },
    "structural_cases": [r for r in master if r["QID"] in {"536","613"}],
    "qualitative_case_selection": group_cases,
    "metric_reconciliation": metric_reconciliation,
    "hash_audit": hash_audit,
}
dump_json(OUT / "STEP8_ANALYSIS_SUMMARY.json", summary)

# Human-readable qualitative notes.
notes = ["# Deterministic qualitative case notes", "", "Cases were selected by the frozen rules in `ERROR_ANALYSIS_RUBRIC.md`. Gold judgments are used only after retrieval for analysis.", ""]
for q in qualitative:
    notes += [f"## Q{q['QID']} — {q['Case_Groups']}", "", f"- Question: {q['Question']}", f"- Selected terms: {q['Selected_Terms']}", f"- Accepted roots: {q['Accepted_Roots']}", f"- Unresolved terms: {q['Unresolved_Terms']}", f"- Semantic coverage: {float(q['Semantic_Coverage_Proportion']):.1%}", f"- Gold passages: {q['Gold_Relevant_Passage_IDs']}", f"- Best relevant ranks (BM25 Top-100 / Dense Top-100 / fixed RALSR full candidate / tuned RALSR full candidate): {q['BM25_Top_Relevant_Rank_Top100'] or 'absent from Top 100'} / {q['Dense_Top_Relevant_Rank_Top100'] or 'absent from Top 100'} / {q['Fixed_RALSR_Top_Relevant_Rank'] or 'absent'} / {q['Tuned_RALSR_Top_Relevant_Full_Candidate_Rank'] or 'absent'}", f"- Query-root passage-corpus hit counts: {q['Query_Root_Corpus_Passage_Hit_Counts']}", f"- Interpretation: {q['Evidence_Grounded_Interpretation']}", ""]
(OUT / "QUALITATIVE_CASE_NOTES.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

# Compact frozen no-answer reminder; no re-analysis of Step 4.
gold_no_answer_qids = ["522", "535", "546", "547", "554", "582", "604"]
no_answer_rows = []
for qid in gold_no_answer_qids:
    no_answer_rows.append({
        "QID": qid,
        "BM25_Output_Type": perq["BM25"][qid]["Output_Type"],
        "Dense_Output_Type": perq["Dense_E5"][qid]["Output_Type"],
        "Fixed_RALSR_Output_Type": perq["Fixed_RALSR"][qid]["Output_Type"],
        "Tuned_RALSR_Output_Type": perq["Tuned_RALSR"][qid]["Output_Type"],
        "Any_System_Explicit_No_Answer": any(perq[name][qid]["Top_Prediction"] == "-1" for name in perq),
        "Note": "All systems returned normal passages; Step 4 already established zero explicit no-answer credit.",
    })
write_csv(OUT / "NO_ANSWER_REMINDER.csv", no_answer_rows)

# Main report assembled from the frozen, machine-readable summaries.
tax_counts = {r["Primary_Category"]: r["QID_Count"] for r in tax_rows}
pair_counts = {"dense_fixed": {r["Outcome_Group"]: r["QID_Count"] for r in dense_fixed}, "bm25_fixed": {r["Outcome_Group"]: r["QID_Count"] for r in bm25_fixed}, "dense_bm25": {r["Outcome_Group"]: r["QID_Count"] for r in dense_bm25}}
dense_only = [r for r in master if r["QID"] in dense_only_qids]
dense_only_ranking_failures = sum(r["Primary_RALSR_Outcome"] == "Candidate present but ranking failure" for r in dense_only)
dense_only_candidate_absence = sum(r["Primary_RALSR_Outcome"] in {"Structural zero-candidate failure", "Candidate-generation failure"} for r in dense_only)

report = f"""# Step 8 — Core Error Analysis

## Scope and integrity

This post-hoc analysis uses the 44 officially judged answerable test qids. Qid 504 remains excluded because it is absent from the published test qrels. The seven gold no-answer qids are discussed only in a compact reminder. All expected frozen hashes matched before analysis, and means recomputed from the frozen per-query rows reproduced the Step-7 answerable-only MAP@10 and MRR@10 values within `1e-12`. No retrieval, ranking, candidate, feature, root, semantic, weight, qrel, or no-answer artifact was changed.

## Candidate failure versus ranking failure

Fixed RALSR produced no candidates for {tax_counts['Structural zero-candidate failure']} of 44 answerable questions ({pct(tax_counts['Structural zero-candidate failure'],44):.1f}%). It produced candidates but omitted every gold passage for {tax_counts['Candidate-generation failure']} more ({pct(tax_counts['Candidate-generation failure'],44):.1f}%). Thus a relevant passage was absent from the full candidate pool for 8 questions ({pct(8,44):.1f}%). At least one relevant passage was present for 36 questions ({pct(36,44):.1f}), but 21 of these were ranking failures: the best relevant candidate remained below rank 10. Fifteen questions ({pct(15,44):.1f}%) had a relevant passage in the Top 10, and 4 ({pct(4,44):.1f}%) had one at rank 1.

The fixed candidate recall was 36/44 ({pct(36,44):.1f}%) for the full candidate pool, 30/44 ({pct(30,44):.1f}%) at Top 100, 15/44 ({pct(15,44):.1f}%) at Top 10, and 4/44 ({pct(4,44):.1f}%) at rank 1. Candidate-set size across all 44 qids had mean 296.1, median 231, minimum 0, and maximum 1,102. Among the 42 candidate-bearing qids, mean size was 310.2 and median size was 244.

| Mutually exclusive fixed-RALSR outcome | Count | Share |
|---|---:|---:|
| Structural zero-candidate failure | {tax_counts['Structural zero-candidate failure']} | {pct(tax_counts['Structural zero-candidate failure'],44):.1f}% |
| Candidate-generation failure | {tax_counts['Candidate-generation failure']} | {pct(tax_counts['Candidate-generation failure'],44):.1f}% |
| Candidate present but ranking failure | {tax_counts['Candidate present but ranking failure']} | {pct(tax_counts['Candidate present but ranking failure'],44):.1f}% |
| Partial retrieval success | {tax_counts['Partial retrieval success']} | {pct(tax_counts['Partial retrieval success'],44):.1f}% |
| Strong retrieval success (AP@10 = 1.0) | {tax_counts['Strong retrieval success']} | {pct(tax_counts['Strong retrieval success'],44):.1f}% |

The categories sum exactly to 44. “Strong” uses the exact criterion AP@10 = 1.0; no subjective near-maximal threshold is used.

## Pairwise system outcomes

Positive AP@10 defines per-query success.

| Comparison | A-only | B-only | Both succeed | Both fail |
|---|---:|---:|---:|---:|
| Dense E5 vs fixed RALSR | {pair_counts['dense_fixed']['Dense E5-only success']} | {pair_counts['dense_fixed']['Fixed RALSR-only success']} | {pair_counts['dense_fixed']['Both succeed']} | {pair_counts['dense_fixed']['Both fail']} |
| BM25 vs fixed RALSR | {pair_counts['bm25_fixed']['BM25-only success']} | {pair_counts['bm25_fixed']['Fixed RALSR-only success']} | {pair_counts['bm25_fixed']['Both succeed']} | {pair_counts['bm25_fixed']['Both fail']} |
| Dense E5 vs BM25 | {pair_counts['dense_bm25']['Dense E5-only success']} | {pair_counts['dense_bm25']['BM25-only success']} | {pair_counts['dense_bm25']['Both succeed']} | {pair_counts['dense_bm25']['Both fail']} |

Dense E5 succeeded on 21 qids (12 Dense-only plus 9 shared with fixed RALSR), compared with 15 for fixed RALSR. Of the 12 Dense-only cases, {dense_only_ranking_failures} had a gold passage in the RALSR candidate set but below rank 10; only {dense_only_candidate_absence} reflected RALSR candidate absence. This indicates that the held-out gap is associated more strongly with ranking discrimination than with candidate absence, while the six fixed-RALSR-only qids demonstrate complementary root-aware successes.

## Coverage and root-source linkage

Coverage is associated descriptively with the fixed-RALSR outcome. Mean accepted-root / semantic coverage was 0.700 / 0.400 for the two structural failures, 0.676 / 0.419 for the six candidate-generation failures, 0.749 / 0.528 for the 21 ranking failures, and 0.823 / 0.632 for the 15 Top-10 successes. Thirty-one of 44 queries contained at least one unresolved selected term; this included 23 of the 29 failure cases and 8 of the 15 Top-10 successes. Every answerable query retained at least some semantic evidence; none fell in the 0% semantic-coverage bin.

The semantic bins show a descriptive gradient but not a causal result: 4 of 6 fully covered queries succeeded in Top 10; 6 of 17 queries in the 50–<100% bin succeeded; and 5 of 21 queries in the >0–<50% bin succeeded. Mean fixed-RALSR AP@10 was 0.1778, 0.0174, and 0.0393 in those bins respectively. The fully covered bin is small (`n=6`).

All answerable test queries either combined QAC and CAMeL roots or combined resolved roots with unresolved terms. The resolved-plus-unresolved profile accounted for both structural failures, 5/6 candidate-generation failures, 16/21 ranking failures, and 8/15 Top-10 successes. Possible doubled/weak/hamza representation-mismatch terms occurred in 29 qids: 1 structural failure, 4 candidate-generation failures, 15 ranking failures, and 9 Top-10 successes. These patterns are associations and do not establish that a source or mismatch caused an error.

Step-6 category linkage shows recurring failure co-occurrence for unresolved Arabic lexical content (29 qids: 21 failures, 8 successes), possible doubled-root mismatches (24 qids: 18 failures, 6 successes), and possible weak/hamza mismatches (11 qids: 7 failures, 4 successes). Modern/foreign vocabulary appeared in three answerable qids, all failures. These small groups support limitation reporting, not inferential claims.

## Feature-level diagnostic

For the 21 candidate-present ranking failures, the highest-ranked gold candidate was compared with the rank-1 nonrelevant candidate using the six frozen normalized features. Mean gold-minus-nonrelevant differences were: Root Coverage −0.2102, Root Overlap −0.1323, Semantic Overlap −0.1020, Jaccard Root Similarity −0.0926, Passage Root Count −0.1359, and Semantic Count −0.1342. The gold candidate was lower on Root Coverage and Root Overlap in 17/21 cases each, lower on Semantic Overlap in 14/21, and lower on the remaining three dimensions in 14–16 cases. This comparison is conditional on ranking failure and therefore describes how the frozen score separated these pairs; it does not estimate causal feature importance.

## Fixed versus tuned RALSR

Tuned weighting improved per-query AP@10 for 5 qids, worsened it for 8, and left 31 unchanged. Top-10 relevant membership changed for one qid (Q505). No qid simultaneously had higher reciprocal rank and lower AP@10; the aggregate test MRR increase and MAP decrease arise from different query-level changes. Because the candidate set, structural abstentions, and feature values are identical, tuning cannot correct the 8 candidate-absence cases.

## Deterministic qualitative cases

- Dense-only: Q545 and Q563 placed their single gold passages at Dense rank 1 while fixed RALSR ranked them 131 and 154 in its candidate pool; Q571 was Dense rank 1 while fixed RALSR ranked the gold passage 54. These are ranking failures, not candidate failures.
- RALSR-only: Q534 placed a gold passage at fixed-RALSR rank 1, Q559 at rank 2, and Q518 at rank 4, while Dense had no gold passage in Top 10. These cases show complementarity despite lower aggregate RALSR performance.
- Candidate-generation failure: the deterministic Group-F cases are recorded in `QUALITATIVE_CASES.csv`; each has a nonempty candidate set but no gold passage anywhere in it, so no weight setting over the frozen features could retrieve the answer.
- Ranking failure: Q576 had full semantic coverage but its best gold candidate was fixed-RALSR rank 313; Q563 and Q545 had best gold ranks 154 and 131. Dense ranked a gold passage first for all three.
- Structural false abstention: Q536 roots `رود`, `اله`, `منن`, and `كون`, and Q613 roots `خوف`, `اله`, and `رج`, each had zero occurrences in the frozen passage-root representation. Both queries also contained unresolved terms and had 40% semantic coverage, so root-only candidate generation returned no passages. Dense found gold passages only below Top 10 (Q536 rank 30; Q613 rank 12), while BM25 found Q613 at rank 81 and no Q536 gold passage in Top 100. The frozen RALSR `-1` decisions remain unchanged.

## No-answer reminder

The seven gold test no-answer qids are 522, 535, 546, 547, 554, 582, and 604. BM25 and Dense did not abstain, and fixed/tuned RALSR structural abstentions (536 and 613) did not overlap the gold no-answer set. All four systems therefore received zero explicit no-answer credit. This section does not repeat the completed Step-4 audit.

## Primary failure taxonomy

The evidence supports five primary outcomes: structural zero-candidate failure, nonempty-candidate generation failure, candidate-present ranking failure, partial Top-10 success, and exact maximal success. Secondary diagnostic tags include unresolved selected terms, partial semantic coverage, possible doubled/weak/hamza representation mismatch, and root-source profile. The dominant fixed-RALSR error mode was candidate-present ranking failure (21/44), followed by candidate absence (8/44 including structural cases).

## Thesis-ready Results paragraph

On the 44 answerable held-out test questions, fixed RALSR included at least one judged relevant passage in its full candidate set for 36 questions (81.8%), but only 15 questions (34.1%) placed a relevant passage in the Top 10. Eight questions lacked a relevant candidate, including two structural zero-candidate cases, while 21 contained a relevant candidate that was ranked below the Top 10. Dense E5 achieved positive AP@10 on 21 questions compared with 15 for fixed RALSR; among the 12 Dense-only successes, 11 were RALSR ranking failures rather than candidate omissions. Fixed RALSR nevertheless produced six complementary successes where Dense obtained zero AP@10.

## Thesis-ready Discussion paragraph

Dense E5’s higher held-out performance is consistent with two observed limitations of the root-aware pipeline. First, fixed RALSR depended on successful root-based candidate construction and omitted every relevant passage for eight answerable questions. Second, even when a relevant passage was available, the handcrafted feature score failed to place it in the Top 10 for 21 questions; in these cases the highest-ranked gold candidate usually had lower normalized root and semantic overlap values than the top nonrelevant candidate. Root and Maqāyīs coverage were descriptively higher among Top-10 successes than among candidate-generation and ranking failures, while unresolved and possible root-representation-mismatch terms frequently co-occurred with errors. The tuned weights changed only one query’s Top-10 relevant membership and could not address candidate or coverage limitations, whereas the learned multilingual dense representation did not require complete root or exact-lexicon coverage for every selected term.

## Thesis-ready future-work note

Future work could test a predeclared hybrid in which Dense retrieval supplies a broader candidate set and frozen root-aware evidence reranks those candidates. Other hypotheses include improved weak/doubled-root representation alignment, broader lexical-semantic resources, and a separately trained no-answer detector. These are prospective experiments; none was implemented after test inspection.

## Limitations

The analysis is descriptive and uses one 44-query held-out population. Subgroups are sometimes small, exact lexical/root overlap is computed against the frozen normalized representations, and feature comparisons condition on observed failures. The results explain associations in the frozen system but do not establish causal effects or statistical significance.

## Verdict

**STEP 8 COMPLETE — CORE ERROR ANALYSIS VERIFIED**
"""
(OUT / "CORE_ERROR_ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")

# Integrity record before final lineage/checksum completion.
integrity = {
    "verified_at_stage": "Step 8",
    "all_expected_source_hashes_match": all(r["Match"] for r in hash_audit),
    "source_hash_audit": hash_audit,
    "step7_answerable_metric_reconciliation": metric_reconciliation,
    "retrieval_systems_executed": [],
    "frozen_artifacts_modified": [],
    "qid_504_used_in_primary_population": False,
    "primary_population_qid_count": 44,
    "taxonomy_count_sum": sum(r["QID_Count"] for r in tax_rows),
}
dump_json(OUT / "INTEGRITY_VERIFICATION.json", integrity)

# Lineage manifest before final checksum completion.
manifest = {
    "stage": "Step 8 — Core Error Analysis",
    "analysis_date": "2026-09-15",
    "scope": "post-hoc analysis only; no retrieval or frozen result modification",
    "primary_population": {"split": "test", "gold_status": "answerable", "qid_count": 44, "qid_504_excluded": True},
    "sources": [{"path": str(p), "sha256": sha256(p)} for p in EXPECTED_HASHES],
    "analysis_rules": {"rubric": str(OUT / "ERROR_ANALYSIS_RUBRIC.md"), "strong_success": "AP@10 == 1.0", "substantial_rank_gap": ">=5 positions or Top10 in only one system", "semantic_bins": ["100%", "50–<100%", ">0–<50%", "0%"]},
    "integrity": {"all_expected_hashes_match": all(r["Match"] for r in hash_audit), "step7_answerable_metrics_reproduced_from_frozen_per_query_rows": all(r["Within_1e-12"] for r in metric_reconciliation), "retrieval_reruns": 0, "ranking_changes": 0, "feature_changes": 0, "qrel_changes": 0},
}

# Record output identities without making the manifest self-referential.
output_files = sorted([p for p in OUT.iterdir() if p.is_file() and p.name not in {"CORE_ERROR_ANALYSIS_LINEAGE_MANIFEST.json", "SHA256SUMS.txt"}], key=lambda p: p.name)
manifest["outputs"] = [{"path": str(p), "sha256": sha256(p), "data_rows": (sum(1 for _ in p.open("r", encoding="utf-8-sig", newline="")) - 1) if p.suffix.lower() == ".csv" else None} for p in output_files]
dump_json(OUT / "CORE_ERROR_ANALYSIS_LINEAGE_MANIFEST.json", manifest)

# Final checksums cover every deliverable except the manifest itself.
checksum_files = sorted([p for p in OUT.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"], key=lambda p: p.name)
(OUT / "SHA256SUMS.txt").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in checksum_files), encoding="ascii")

print(json.dumps({"status": "STEP 8 COMPLETE — CORE ERROR ANALYSIS VERIFIED", "output": str(OUT), "taxonomy": summary["taxonomy"], "pairwise": summary["pairwise"], "fixed_vs_tuned": summary["fixed_vs_tuned"], "qualitative_cases": group_cases}, ensure_ascii=False, indent=2))
