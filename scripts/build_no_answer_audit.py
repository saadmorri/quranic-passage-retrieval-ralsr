from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch: {path}\nexpected={expected}\nactual={actual}")


def read_question_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row[0].strip() for row in csv.reader(handle, delimiter="\t") if row and row[0].strip()]


def read_qrels(path: Path) -> tuple[dict[str, list[tuple[str, int]]], list[list[str]]]:
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if not row:
                continue
            if len(row) != 4:
                raise RuntimeError(f"Malformed qrel row in {path}: {row}")
            qid, q0, docid, relevance = [value.strip() for value in row]
            grouped[qid].append((docid, int(relevance)))
            rows.append([qid, q0, docid, relevance])
    return dict(grouped), rows


def read_run_types(path: Path, expected_qids: set[str]) -> tuple[dict[str, str], set[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) != 6:
                raise RuntimeError(f"Malformed run row {path}:{line_number}")
            grouped[row[0].strip()].append(row[2].strip())
    if set(grouped) != expected_qids:
        raise RuntimeError(f"Run qid coverage mismatch: {path}")
    result = {}
    structural = set()
    for qid, docids in grouped.items():
        if docids == ["-1"]:
            result[qid] = "Structural_No_Answer"
            structural.add(qid)
        elif "-1" in docids:
            raise RuntimeError(f"Mixed -1 and normal predictions: {path}, qid {qid}")
        else:
            result[qid] = "Normal_Passages"
    return result, structural


def write_list(path: Path, qids: list[str]) -> None:
    path.write_text("".join(f"{qid}\n" for qid in qids), encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def filter_qrels_bytes(source: Path, destination: Path, excluded_qids: set[str]) -> None:
    retained = []
    for line in source.read_bytes().splitlines(keepends=True):
        first = line.split(b"\t", 1)[0].decode("utf-8-sig").strip()
        if first and first not in excluded_qids:
            retained.append(line)
        elif not first:
            retained.append(line)
    destination.write_bytes(b"".join(retained))


def main() -> None:
    codex = Path(r"${THESIS_CODEX_ROOT}")
    workspace = codex / "Master Thesis - Quranic Passage Retrieval - Writing Workspace"
    correction = codex / "Master Thesis - Quranic Passage Retrieval - Correction Control"
    output = correction / "No_Answer_Audit"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "Scripts").mkdir(exist_ok=True)

    quranqa = workspace / "02_Methodology_Evidence" / "Data and Resources" / "QuranQA"
    step3 = correction / "Core_Retrieval_Frozen_Runs" / "Official_Top10"
    tuned = correction / "Tuned_RALSR_Ablation"
    fixed_source = workspace / "02_Methodology_Evidence" / "RALSR" / "Corrected_Outputs" / "RALSR_Split_Safe_Normalization_No_Answer" / "RALSR_Top100_Split_Safe_With_No_Answer.csv"
    bm25_source = workspace / "02_Methodology_Evidence" / "Baselines" / "Corrected_Outputs" / "Base_BM25_Corrected_Query" / "BM25_Retrieval_Top100_Corrected.csv"
    dense_source = workspace / "02_Methodology_Evidence" / "Baselines" / "Dense_Retrieval_Top100.csv"
    official_dir = correction / "Official_QuranQA_TaskA_Evaluation"
    scorer = official_dir / "01_Official_Organizer_Files" / "QQA23_TaskA_eval.py"
    independent = official_dir / "02_Evaluation_Pipeline" / "quranqa_taskA_independent_eval.py"

    expected_hashes = {
        bm25_source: "B24A5946FF821136352616B65B6C9AA1C1D66279114F8780F95D8425E1803235",
        dense_source: "FD603ACEA95DA89FE31E7F1E9D3AB3A778BCE36D8C566C5C02E727A6582A2E6F",
        fixed_source: "7B70FBEACBF4E1AB11888CCB1798E9C48349C2977795F871601866A1D04D4D6C",
        tuned / "TUNED_RALSR_WEIGHTS.json": "06C2447FABEA838DDBC09153B126EBCD4B23D1872DCCE35673966D1AA26D9F6B",
        tuned / "thesis_TunedTR.tsv": "D0BE7B7F08FE0C32B2F3FF321F2E182A8451E4FF8A2F67F9E00ECCA54278C0F4",
        tuned / "thesis_TunedDV.tsv": "9C5CEDCB8BDBB17AE5BB710472B6D432789DD32079FDA70ABFCEBADD5ADE5207",
        scorer: "39F98576783FBB41222360F2319BC011A90400075122334728BC68BBA55DBE1A",
        independent: "DABB1B45B0905B589E8A8AF07967B7CE1184CED8D88ADA75BEE9BE4F80239253",
    }
    question_hashes = {
        "train": "3B707F9A857A668B71A2DADEA9AA6545F038E656F6092E6D78EC0C4C99A3A0A9",
        "dev": "BAAE0A5DFE7ADC3E300DC79F504527897E84C2E3DACB19789638ECB29E97229B",
        "test": "33C82BD0918F2742360FCCB7598E655241FB5CFEEEECB7F2B009EC59B16E1155",
    }
    qrel_hashes = {
        "train": "48E64E24A715BD77B824D9A4C17863C39614FE198A0A44FED23E76A6BEA0CFE6",
        "dev": "6F74218F1259AA144C795F19D0E4FB99095F927511EE708FA7EE4125F37A3590",
        "test": "E23E4CF0628EB2FF39562852A5632DE0D948C8F643B5EB8E08B1D8B69CBA0332",
    }
    run_hashes = {
        "BM25": {
            "train": "3064A67AA640568A20C78FB3989E95909F95D7019FED658EA26558DCFD9A50C1",
            "dev": "8074118B7C496A74FC984606C1118E1978EED31358F5A5EF93B60C50EF9900E0",
            "test": "4DAE625B5877A4C1D92DBDE6E57DB3E226C1C85AA1AA529C0B76D5DA403684E2",
        },
        "Dense": {
            "train": "BD640EC2B831019AAD6794BE9C20778223F738AAFB47749B788E443409169020",
            "dev": "B04E6595EA4CA1257E6F6E314FCBD86EF8CC79906A9630F60BF87C3CF773ECFE",
            "test": "46585DEE581BC70B9321989FE719DBB8AC3F1CD47D0B187221A631D29B22798D",
        },
        "Fixed_RALSR": {
            "train": "8914022F74E143B5B3FB0187F3B444D551CEFBF7B8926AEEF16D5E7E69A83124",
            "dev": "7BFC6E2A33B080D81AE4F80C7BFD710D378E0A761147BEEF0F5D30D4C8C4A883",
            "test": "6455AFF49056D7DC92901B70880E9CB0D587AA6DA6C261F5CBD00F31218CC019",
        },
    }
    run_files = {
        "BM25": {split: step3 / f"thesis_BM25{'tr' if split == 'train' else 'dv' if split == 'dev' else 'te'}.tsv" for split in ("train", "dev", "test")},
        "Dense": {split: step3 / f"thesis_E5{'tr' if split == 'train' else 'dv' if split == 'dev' else 'te'}.tsv" for split in ("train", "dev", "test")},
        "Fixed_RALSR": {split: step3 / f"thesis_RALSR{'tr' if split == 'train' else 'dv' if split == 'dev' else 'te'}.tsv" for split in ("train", "dev", "test")},
    }

    for path, expected in expected_hashes.items():
        verify(path, expected)
    for split in ("train", "dev", "test"):
        verify(quranqa / f"QQA23_TaskA_ayatec_v1.2_{split}.tsv", question_hashes[split])
        verify(quranqa / f"QQA23_TaskA_ayatec_v1.2_qrels_{split}.gold", qrel_hashes[split])
        for system in run_files:
            verify(run_files[system][split], run_hashes[system][split])

    expected_structural = {
        "train": {"102", "108", "110", "137", "141", "143", "212", "235", "252", "258"},
        "dev": {"234"},
        "test": {"536", "613"},
    }
    tuned_files = {"train": tuned / "thesis_TunedTR.tsv", "dev": tuned / "thesis_TunedDV.tsv"}
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    audit = {"splits": {}, "integrity": {}, "organizer_semantics": {}}
    source_manifest = {"questions": {}, "qrels": {}, "runs": {}, "scorer": {"path": str(scorer), "sha256": sha256(scorer)}}

    for split in ("train", "dev", "test"):
        question_path = quranqa / f"QQA23_TaskA_ayatec_v1.2_{split}.tsv"
        qrels_path = quranqa / f"QQA23_TaskA_ayatec_v1.2_qrels_{split}.gold"
        qids = read_question_ids(question_path)
        qid_set = set(qids)
        qrels, _ = read_qrels(qrels_path)
        qrel_qids = set(qrels)
        gold_no = {qid for qid, docs in qrels.items() if any(docid == "-1" for docid, _ in docs)}
        if any(rel != 1 for qid in gold_no for docid, rel in qrels[qid] if docid == "-1"):
            raise RuntimeError("Official -1 qrel did not have relevance 1")
        answerable = qrel_qids - gold_no
        absent = qid_set - qrel_qids
        unexpected_qrels = qrel_qids - qid_set
        if unexpected_qrels:
            raise RuntimeError(f"Qrels contain unexpected qids for {split}: {unexpected_qrels}")

        output_types = {}
        structural_sets = {}
        for system in ("BM25", "Dense", "Fixed_RALSR"):
            types, structural = read_run_types(run_files[system][split], qid_set)
            output_types[system] = types
            structural_sets[system] = structural
            if system in ("BM25", "Dense") and structural:
                raise RuntimeError(f"Unexpected {system} -1 predictions in {split}: {structural}")
        if structural_sets["Fixed_RALSR"] != expected_structural[split]:
            raise RuntimeError(f"Fixed RALSR structural set mismatch for {split}")

        if split in tuned_files:
            tuned_types, tuned_structural = read_run_types(tuned_files[split], qid_set)
            if tuned_structural != expected_structural[split]:
                raise RuntimeError(f"Tuned RALSR structural set mismatch for {split}")
            tuned_evidence = "Frozen_tuned_official_run"
        else:
            tuned_types = {qid: output_types["Fixed_RALSR"][qid] for qid in qids}
            tuned_structural = set(expected_structural[split])
            tuned_evidence = "Frozen_shared_candidate_set; tuned_test_run_not_generated"

        correct = expected_structural[split] & gold_no
        false = expected_structural[split] & answerable
        unjudged_structural = expected_structural[split] & absent
        missed = gold_no - expected_structural[split]

        for qid in qids:
            if qid in absent:
                gold_status = "Qrel_Absent"
                relevant_count = ""
            elif qid in gold_no:
                gold_status = "No_Answer"
                relevant_count = 0
            else:
                gold_status = "Answerable"
                relevant_count = sum(1 for docid, rel in qrels[qid] if docid != "-1" and rel > 0)

            structural = qid in expected_structural[split]
            if structural and gold_status == "No_Answer":
                assessment = "Correct"
            elif structural and gold_status == "Answerable":
                assessment = "False"
            elif structural and gold_status == "Qrel_Absent":
                assessment = "Unjudged"
            elif not structural and gold_status == "No_Answer":
                assessment = "Missed"
            else:
                assessment = "Not_Applicable"

            notes = []
            if qid == "504":
                notes.append("Official test question absent from published test qrels; not classified")
            if qid == "141":
                notes.append("Officially answerable; false structural abstention; no gold passage inserted")
            if gold_status == "No_Answer":
                notes.append("BM25 and Dense return normal passages")
            if split == "test":
                notes.append("Tuned output type derives from the unchanged frozen candidate set; no tuned test run generated")

            all_rows.append({
                "Split": split,
                "QID": qid,
                "Gold_Status": gold_status,
                "Gold_Relevant_Passage_Count": relevant_count,
                "BM25_Output_Type": output_types["BM25"][qid],
                "Dense_Output_Type": output_types["Dense"][qid],
                "Fixed_RALSR_Output_Type": output_types["Fixed_RALSR"][qid],
                "Tuned_RALSR_Output_Type": tuned_types[qid],
                "Tuned_RALSR_Evidence": tuned_evidence,
                "Structural_No_Answer": structural,
                "Structural_Abstention_Assessment": assessment,
                "Notes": "; ".join(notes),
            })

        summary_rows.append({
            "Split": split,
            "Official_Question_Count": len(qids),
            "Judged_QID_Count": len(qrel_qids),
            "Gold_No_Answer_Count": len(gold_no),
            "Gold_Answerable_Count": len(answerable),
            "Qrel_Absent_Count": len(absent),
            "Gold_No_Answer_Percentage_of_Judged": round(100 * len(gold_no) / len(qrel_qids), 1),
            "RALSR_Structural_Abstention_Count": len(expected_structural[split]),
            "RALSR_Structural_Abstention_Percentage_of_Official_Questions": round(100 * len(expected_structural[split]) / len(qids), 1),
            "Correct_Structural_Abstention_Count": len(correct),
            "False_Structural_Abstention_Count": len(false),
            "Unjudged_Structural_Abstention_Count": len(unjudged_structural),
            "Missed_Gold_No_Answer_Count": len(missed),
            "Gold_No_Answer_QIDs": " ".join(sorted(gold_no, key=int)),
            "Structural_No_Answer_QIDs": " ".join(sorted(expected_structural[split], key=int)),
            "Correct_Structural_Abstention_QIDs": " ".join(sorted(correct, key=int)),
            "False_Structural_Abstention_QIDs": " ".join(sorted(false, key=int)),
            "Unjudged_Structural_Abstention_QIDs": " ".join(sorted(unjudged_structural, key=int)),
            "Missed_Gold_No_Answer_QIDs": " ".join(sorted(missed, key=int)),
            "Qrel_Absent_QIDs": " ".join(sorted(absent, key=int)),
        })

        write_list(output / f"GOLD_NO_ANSWER_QIDS_{split}.txt", sorted(gold_no, key=int))
        write_list(output / f"ANSWERABLE_QIDS_{split}.txt", sorted(answerable, key=int))
        write_list(output / f"UNJUDGED_OR_QREL_ABSENT_QIDS_{split}.txt", sorted(absent, key=int))
        derivative = output / f"qrels_{split}_answerable_only.gold"
        filter_qrels_bytes(qrels_path, derivative, gold_no)
        derivative_qrels, _ = read_qrels(derivative)
        if set(derivative_qrels) != answerable or any("-1" == docid for docs in derivative_qrels.values() for docid, _ in docs):
            raise RuntimeError(f"Answerable-only qrel derivative failed for {split}")

        source_manifest["questions"][split] = {"path": str(question_path), "sha256": sha256(question_path), "qids": len(qids)}
        source_manifest["qrels"][split] = {
            "official_path": str(qrels_path),
            "official_sha256": sha256(qrels_path),
            "official_git_blob_id": {"train": "119837f85dd1a2ee08f846b3792b21b787ef3a57", "dev": "8c0758a2b75394e98f8a0f568b2d77690a9a96d3", "test": "9c2c403d563a8ccae4d65b3e06382b2fc4f2d7a9"}[split],
            "answerable_only_path": str(derivative),
            "answerable_only_sha256": sha256(derivative),
        }
        source_manifest["runs"][split] = {
            system: {"path": str(run_files[system][split]), "sha256": sha256(run_files[system][split])}
            for system in run_files
        }
        if split in tuned_files:
            source_manifest["runs"][split]["Tuned_RALSR"] = {"path": str(tuned_files[split]), "sha256": sha256(tuned_files[split])}
        else:
            source_manifest["runs"][split]["Tuned_RALSR"] = {
                "path": None,
                "status": "not_generated_in_tuned_ablation",
                "structural_decisions_source": "unchanged fixed Notebook-9B candidate set",
            }

        audit["splits"][split] = {
            "official_questions": len(qids),
            "judged_qids": len(qrel_qids),
            "gold_no_answer_qids": sorted(gold_no, key=int),
            "gold_answerable_qids": len(answerable),
            "qrel_absent_qids": sorted(absent, key=int),
            "structural_no_answer_qids": sorted(expected_structural[split], key=int),
            "correct_structural_abstentions": sorted(correct, key=int),
            "false_structural_abstentions": sorted(false, key=int),
            "unjudged_structural_abstentions": sorted(unjudged_structural, key=int),
            "missed_gold_no_answer": sorted(missed, key=int),
        }

    write_csv(output / "NO_ANSWER_MASTER_TABLE.csv", list(all_rows[0]), all_rows)
    write_csv(output / "NO_ANSWER_SPLIT_SUMMARY.csv", list(summary_rows[0]), summary_rows)

    if len(all_rows) != 251:
        raise RuntimeError(f"Master audit table row count changed: {len(all_rows)}")
    q504 = next(row for row in all_rows if row["QID"] == "504")
    q141 = next(row for row in all_rows if row["QID"] == "141")
    if q504["Gold_Status"] != "Qrel_Absent" or q504["Fixed_RALSR_Output_Type"] != "Normal_Passages":
        raise RuntimeError("Qid 504 audit rule failed")
    if q141["Gold_Status"] != "Answerable" or q141["Structural_Abstention_Assessment"] != "False":
        raise RuntimeError("Qid 141 audit rule failed")

    audit["organizer_semantics"] = {
        "gold_no_answer_detection": "qid has qrel docid -1; all preserved official -1 rows have relevance 1",
        "full_credit": "run has exactly one prediction for qid and its docid is -1",
        "zero_credit": "normal passage list, mixed -1/passages, or other structure on a gold no-answer qid",
        "metrics": ["map_cut_10", "recip_rank"],
        "population": "qids represented in supplied qrels; qid 504 is not judged by published test qrels",
        "missing_run_qid_note": "official source does not enforce complete qid coverage; controlled thesis protocol requires complete coverage with the strict validator",
    }
    audit["evaluation_populations"] = {
        "official_judged": "all and only qids represented in each preserved official qrels file",
        "answerable_only": "officially judged qids excluding qids whose qrels contain docid -1",
        "qid_504": "excluded from both test scoring populations because it is absent from published test qrels; retained in prediction files",
    }
    audit["system_design"] = {
        "BM25": {"abstention_mechanism": False, "minus1_predictions": 0},
        "Dense": {"abstention_mechanism": False, "minus1_predictions": 0},
        "Fixed_RALSR": {"abstention": "zero candidates only", "threshold_or_classifier": False},
        "Tuned_RALSR": {"abstention": "same frozen candidate set as fixed RALSR", "threshold_or_classifier": False},
    }
    audit["final_effectiveness_metrics_computed"] = False

    (output / "NO_ANSWER_AUDIT_DATA.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_manifest.update({
        "stage": "Step 4 - No-answer audit and evaluation-protocol freeze",
        "created_local": datetime.now().astimezone().isoformat(),
        "official_sources_modified": False,
        "retrieval_outputs_modified": False,
        "retrieval_rerun": False,
        "final_effectiveness_scoring": False,
        "tuned_weights": {
            "path": str(tuned / "TUNED_RALSR_WEIGHTS.json"),
            "sha256": sha256(tuned / "TUNED_RALSR_WEIGHTS.json"),
        },
    })
    (output / "NO_ANSWER_LINEAGE_MANIFEST.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    integrity = {
        "status": "PASS",
        "official_qrels_unchanged": {split: sha256(quranqa / f"QQA23_TaskA_ayatec_v1.2_qrels_{split}.gold") == qrel_hashes[split] for split in qrel_hashes},
        "source_runs_unchanged": {
            "BM25": sha256(bm25_source) == expected_hashes[bm25_source],
            "Dense": sha256(dense_source) == expected_hashes[dense_source],
            "Fixed_RALSR": sha256(fixed_source) == expected_hashes[fixed_source],
        },
        "step3_runs_unchanged": {system: {split: sha256(run_files[system][split]) == run_hashes[system][split] for split in run_files[system]} for system in run_files},
        "tuned_train_dev_unchanged": {
            "train": sha256(tuned_files["train"]) == expected_hashes[tuned_files["train"]],
            "dev": sha256(tuned_files["dev"]) == expected_hashes[tuned_files["dev"]],
        },
        "tuned_weights_unchanged": sha256(tuned / "TUNED_RALSR_WEIGHTS.json") == expected_hashes[tuned / "TUNED_RALSR_WEIGHTS.json"],
        "master_rows": len(all_rows),
        "retrieval_rerun": False,
        "candidate_regeneration": False,
        "final_metric_table_created": False,
    }
    if not all(integrity["official_qrels_unchanged"].values()) or not all(integrity["source_runs_unchanged"].values()):
        raise RuntimeError("Final integrity verification failed")
    (output / "INTEGRITY_VERIFICATION.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")

    shutil.copy2(Path(__file__).resolve(), output / "Scripts" / Path(__file__).name)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
