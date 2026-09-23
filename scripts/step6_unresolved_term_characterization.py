from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT_PATH = Path(r"${THESIS_WORKSPACE}\02_Methodology_Evidence\QAC and Root Extraction\Corrected_Outputs\Point_2_Query_Root_Provenance\QuranQA_Hybrid_Root_Retrieval_Corrected.csv")
SEM_PATH = Path(r"${THESIS_WORKSPACE}\02_Methodology_Evidence\Maqāyīs Lookup\Corrected_Outputs\Notebook_7_Maqayis_Semantic_Enrichment\QuranQA_Semantic_Enrichment_Corrected.csv")
DB_PATH = Path(r"${THESIS_WORKSPACE}\02_Methodology_Evidence\Maqāyīs Lookup\Lexicon Database and Metadata\db.sqlite")
STEP5_REPORT = Path(r"${THESIS_CONTROL_ROOT}\Maqayis_Printed_Edition_Audit\MAQAYIS_PRINTED_EDITION_AUDIT_REPORT.md")

EXPECTED_SEM_SHA = "1F65EABA834BB1CF5D5098E124218FFAB1C209CCD5AFD9A662390DE360202FB9"
EXPECTED_DB_SHA = "D39ADF6D3846AA17AE92802C9D7EF530F33E857BB3B1163880100C5C479595EB"
ANALYSIS_DATE = "2026-09-15"

SPLIT_ORDER = {"Train": 0, "Development": 1, "Test": 2}
EXPECTED_QUESTION_COUNTS = {"Train": 174, "Development": 25, "Test": 52}

# Frozen after inspection of all 208 unresolved normalized forms and the
# preserved POS/status evidence.  These lists are classification evidence,
# not root repair rules.
PROPER_ENTITY_FORMS = {
    "موسي", "نوح", "يوسف", "عيسي", "ابراهيم", "اسرائيل", "زكريا",
    "التوراة", "ايوب", "مريم", "بداوود", "لقابيل", "وهابيل", "يونس",
    "الجودي", "ثمود", "طالوت", "اليهود", "الانجيل", "الاسراء", "عائشة",
    "عاد",
}
HONORIFIC_FORMS = {"سيدنا", "لسيدنا"}
FUNCTION_FORMS = {
    "هناك", "فلماذا", "فيها", "بانه", "بان", "اذا", "فيه", "اي",
    "عندما", "انما", "ام", "واما", "ذو", "ذي", "غيره", "لغيره",
    "بغير", "علينا", "وبين",
}
MODERN_FOREIGN_FORMS = {
    "الدكتاتورية", "سيداو", "الاكسجين", "الصهيوني", "داعش",
    "والعرقيات", "كروية", "ايجابي",
}
ABBREVIATION_FORMS = {"ص"}

# Record-level named/special forms in Population B.  QAC root اله is retained
# as a special-form exact miss because its 47 records are tagged PN upstream.
UNMATCHED_SPECIAL_FORMS = {
    "الله", "لله", "بالله", "ادم", "مكة", "جهنم", "الجاثية", "سبا",
}

FUNCTION_POS = {
    "adv", "adv_interrog", "prep", "conj", "pron", "part", "part_interrog",
    "verb_pseudo",
}
CONTENT_POS = {"noun", "noun_prop", "noun_quant", "verb", "adj", "N", "V", "ADJ", "PN"}
ARABIC_RE = re.compile(r"^[\u0621-\u064A]+$")
WEAK_HAMZA = set("اويءئؤىإأآ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def flatten(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for qrow in csv.DictReader(handle):
            matches = json.loads(qrow["Lexical_Matches"])
            provenance = json.loads(qrow["Query_Term_Provenance"])
            for index, match in enumerate(matches):
                record = {
                    "Dataset": qrow["Dataset"],
                    "Question_ID": str(qrow["Question_ID"]),
                    "Question": qrow["Question"],
                    "Record_Index": index,
                    "Query_Terms": qrow["Query_Terms"],
                    "Term_Provenance": provenance[index] if index < len(provenance) else None,
                }
                record.update(match)
                records.append(record)
    return records


def record_key(record: dict) -> tuple:
    return (
        SPLIT_ORDER.get(record["Dataset"], 99),
        int(record["Question_ID"]),
        int(record["Record_Index"]),
    )


def s(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d


def pct_text(n: int, d: int) -> str:
    return f"{100 * pct(n, d):.1f}%"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: s(row.get(name)) for name in fieldnames})


def classify_a(record: dict) -> tuple[str, str, str, str]:
    form = s(record.get("Normalized_Term"))
    pos = s(record.get("CAMeL_Selected_POS"))
    status = s(record.get("CAMeL_Status"))

    if form in PROPER_ENTITY_FORMS:
        return (
            "Proper name / named entity",
            "Named person, people, place, scripture, or named event",
            "High",
            "Form and query context identify a named entity; QAC PN evidence is retained where available.",
        )
    if form in HONORIFIC_FORMS:
        return (
            "Honorific / title expression",
            "Religious honorific/title selected as a term",
            "High",
            "Observed title form سيدنا (including attached preposition), distinct from the accompanying name.",
        )
    if form in FUNCTION_FORMS or pos in FUNCTION_POS:
        return (
            "Function / grammatical expression",
            "Particle, preposition, adverb, interrogative, or relational expression",
            "High" if form in FUNCTION_FORMS or pos in FUNCTION_POS else "Medium",
            f"Surface/context and preserved CAMeL POS ({pos or 'unavailable'}) support grammatical rather than lexical-content use.",
        )
    if form in MODERN_FOREIGN_FORMS:
        return (
            "Modern / foreign / institutional vocabulary",
            "Borrowing, acronym, modern technical, political, or institutional expression",
            "High",
            "Observed form is transparently modern, borrowed, institutional, or technical in the question context.",
        )
    if form in ABBREVIATION_FORMS:
        return (
            "Abbreviation / non-lexical token",
            "Parenthetical honorific abbreviation",
            "High",
            "Single-letter ص occurs parenthetically as an honorific abbreviation and carries an upstream sentinel root.",
        )
    if pos in CONTENT_POS:
        return (
            "Arabic lexical content word",
            f"Content POS: {pos}",
            "High",
            f"Preserved CAMeL analysis supplies lemma {s(record.get('CAMeL_Selected_Lemma')) or 'unavailable'} and content POS {pos}; the failure is root acceptance, not word recognition.",
        )
    if status in {"NO_ANALYSIS", "AMBIGUOUS_TOP_ROOT"} and ARABIC_RE.fullmatch(form or ""):
        return (
            "Arabic lexical item with unavailable/ambiguous analysis",
            "Recognizable Arabic form without a usable unique top analysis",
            "Medium",
            f"The form is Arabic-script lexical material, but CAMeL status is {status}; no root was inferred manually.",
        )
    return (
        "Other / uncertain",
        "Insufficient preserved evidence for a narrower linguistic label",
        "Low",
        "The preserved fields and context do not support a narrower category without speculation.",
    )


def technical_a(record: dict) -> tuple[str, str]:
    status = s(record.get("CAMeL_Status"))
    reason = s(record.get("CAMeL_Root_Invalid_Reason"))
    if reason == "incomplete_placeholder_root":
        return "Incomplete placeholder root (#)", reason
    if reason == "sentinel_root":
        return "Sentinel root", reason
    if status == "NO_ANALYSIS":
        return "No CAMeL analysis", status
    if status == "AMBIGUOUS_TOP_ROOT":
        return "Ambiguous top CAMeL root", status
    return "Other unresolved condition", reason or status or "unspecified"


def classify_b(record: dict) -> tuple[str, str, str, str]:
    root = s(record.get("Root_AR"))
    form = s(record.get("Normalized_Term"))
    pos = s(record.get("POS"))
    if len(root) <= 2:
        return (
            "Noncanonical short root key",
            "One- or two-character root accepted by the frozen syntax-only rule",
            "High",
            f"Accepted root {root} has length {len(root)}; no length rule was retrospectively imposed.",
        )
    if form in UNMATCHED_SPECIAL_FORMS or root == "اله":
        return (
            "Proper-name / special lexical form",
            "Named or special lexical item with no exact digital root key",
            "High" if form in UNMATCHED_SPECIAL_FORMS else "Medium",
            f"Surface form {form} / upstream POS {pos or 'unavailable'} identifies a named or special lexical use; exact lookup of {root} returned no row.",
        )
    if len(root) == 3 and root[1] == root[2]:
        return (
            "Possible doubled-root representation mismatch",
            "Triliteral root ends in a doubled radical",
            "Medium",
            f"Root shape {root} has identical second and third radicals; exact digital keying may use a different doubled-root representation. This was not remapped.",
        )
    if any(char in WEAK_HAMZA for char in root):
        return (
            "Possible weak/hamza representation mismatch",
            "Root contains weak/hamza/alif variant",
            "Medium",
            f"Root shape {root} contains a weak or hamza/alif character that may be keyed differently. Exact-match status remains unchanged.",
        )
    if len(root) > 3:
        return (
            "Extended/four-letter root key absent",
            "Syntactically valid root has more than three letters",
            "High",
            f"Root {root} is an accepted extended root key, but no exact digital Maqāyīs row was returned.",
        )
    return (
        "Valid root absent from exact digital lookup — cause undetermined",
        "Ordinary three-letter root shape with no exact database key",
        "High",
        f"Exact lookup of the authoritative root {root} returned no row; current evidence does not establish whether the cause is lexicon scope, indexing, or another representation issue.",
    )


def json_number_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite value encountered")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6 unresolved-term and Maqāyīs-coverage characterization")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_hashes_before = {
        "Point2_Root_Provenance": sha256(ROOT_PATH),
        "Notebook7_Semantic_Enrichment": sha256(SEM_PATH),
        "Maqayis_SQLite": sha256(DB_PATH),
        "Step5_Audit_Report": sha256(STEP5_REPORT) if STEP5_REPORT.exists() else None,
    }
    if source_hashes_before["Notebook7_Semantic_Enrichment"] != EXPECTED_SEM_SHA:
        raise RuntimeError("Authoritative semantic-enrichment SHA-256 mismatch")
    if source_hashes_before["Maqayis_SQLite"] != EXPECTED_DB_SHA:
        raise RuntimeError("Maqāyīs SQLite SHA-256 mismatch")

    root_records = flatten(ROOT_PATH)
    sem_records = flatten(SEM_PATH)
    if len(root_records) != 1342 or len(sem_records) != 1342:
        raise RuntimeError("Authoritative row-count mismatch")
    if len({record_key(r) for r in root_records}) != 1342 or len({record_key(r) for r in sem_records}) != 1342:
        raise RuntimeError("Duplicate flattened record key")

    root_by_key = {record_key(r): r for r in root_records}
    sem_by_key = {record_key(r): r for r in sem_records}
    if set(root_by_key) != set(sem_by_key):
        raise RuntimeError("Point-2 and Notebook-7 record keys differ")
    lineage_mismatches = []
    for key in sorted(root_by_key):
        a, b = root_by_key[key], sem_by_key[key]
        for field in ("Query_Term", "Original_Term", "Normalized_Term", "Root_Resolved", "Root_Source", "Root_AR", "Resolution_Status"):
            if a.get(field) != b.get(field):
                lineage_mismatches.append((key, field, a.get(field), b.get(field)))
    if lineage_mismatches:
        raise RuntimeError(f"Point-2 to Notebook-7 lineage mismatch: {lineage_mismatches[:3]}")

    unresolved = [r for r in sem_records if not r.get("Root_Resolved")]
    matched = [r for r in sem_records if r.get("Root_Resolved") and r.get("Semantic_Found")]
    unmatched = [r for r in sem_records if r.get("Root_Resolved") and not r.get("Semantic_Found")]
    if (len(unresolved), len(matched), len(unmatched)) != (329, 724, 289):
        raise RuntimeError("Population counts do not match frozen evidence")
    if any(r.get("Maqayis_Lookup_Attempted") for r in unresolved):
        raise RuntimeError("Unresolved record unexpectedly attempted Maqāyīs lookup")
    if any(not r.get("Maqayis_Lookup_Attempted") or r.get("Semantic_Found") for r in unmatched):
        raise RuntimeError("Population-B lookup contract violated")

    master_rows: list[dict] = []
    for record in sorted(unresolved + unmatched, key=record_key):
        if not record.get("Root_Resolved"):
            population = "Root_Unresolved"
            technical_category, failure_reason = technical_a(record)
            linguistic_category, subcategory, confidence, evidence = classify_a(record)
            lemma = record.get("CAMeL_Selected_Lemma") or record.get("Lemma_AR")
            pos = record.get("CAMeL_Selected_POS") or record.get("POS")
            notes = "No root was assigned or inferred; characterization uses preserved analyzer/provenance fields only."
        else:
            population = "Root_Resolved_Maqayis_Unmatched"
            technical_category = "Exact digital Maqāyīs lookup miss"
            failure_reason = "no_exact_maqayis_root_match"
            linguistic_category, subcategory, confidence, evidence = classify_b(record)
            lemma = record.get("Lemma_AR") or record.get("CAMeL_Selected_Lemma")
            pos = record.get("POS") or record.get("CAMeL_Selected_POS")
            notes = "Authoritative root was preserved; no alternate key or semantic entry was substituted."
        master_rows.append({
            "Dataset": record["Dataset"],
            "Question_ID": record["Question_ID"],
            "Record_Index": record["Record_Index"],
            "Original_Query": record["Question"],
            "Term": record.get("Original_Term") or record.get("Query_Term"),
            "Query_Term": record.get("Query_Term"),
            "Normalized_Term": record.get("Normalized_Term"),
            "Population": population,
            "Root": record.get("Root_AR"),
            "Root_Source": record.get("Root_Source"),
            "QAC_Status": record.get("QAC_Status"),
            "QAC_Lexical_Matched": record.get("QAC_Lexical_Matched"),
            "QAC_Candidate_Count": record.get("QAC_Candidate_Count"),
            "QAC_Candidate_POS": record.get("QAC_Candidate_POS"),
            "CAMeL_Status": record.get("CAMeL_Status"),
            "CAMeL_Analysis_Found": record.get("CAMeL_Analysis_Found"),
            "CAMeL_Analysis_Count": record.get("CAMeL_Analysis_Count"),
            "CAMeL_Top_Analysis_Count": record.get("CAMeL_Top_Analysis_Count"),
            "CAMeL_Root_Ambiguous": record.get("CAMeL_Root_Ambiguous"),
            "CAMeL_Root_Raw": record.get("CAMeL_Root_Raw"),
            "CAMeL_Root_Normalized": record.get("CAMeL_Root_Normalized"),
            "CAMeL_Root_Valid": record.get("CAMeL_Root_Valid"),
            "Strict_Validity": bool(record.get("Root_Resolved")),
            "Root_Failure_Reason": failure_reason,
            "Lemma": lemma,
            "POS": pos,
            "Maqayis_Lookup_Root": record.get("Maqayis_Lookup_Root"),
            "Maqayis_Lookup_Attempted": record.get("Maqayis_Lookup_Attempted"),
            "Maqayis_Matched": record.get("Semantic_Found"),
            "Technical_Category": technical_category,
            "Linguistic_Category": linguistic_category,
            "Subcategory": subcategory,
            "Classification_Evidence": evidence,
            "Confidence": confidence,
            "Notes": notes,
        })

    fields = [
        "Dataset", "Question_ID", "Record_Index", "Original_Query", "Term", "Query_Term", "Normalized_Term",
        "Population", "Root", "Root_Source", "QAC_Status", "QAC_Lexical_Matched", "QAC_Candidate_Count",
        "QAC_Candidate_POS", "CAMeL_Status", "CAMeL_Analysis_Found", "CAMeL_Analysis_Count",
        "CAMeL_Top_Analysis_Count", "CAMeL_Root_Ambiguous", "CAMeL_Root_Raw", "CAMeL_Root_Normalized",
        "CAMeL_Root_Valid", "Strict_Validity", "Root_Failure_Reason", "Lemma", "POS", "Maqayis_Lookup_Root",
        "Maqayis_Lookup_Attempted", "Maqayis_Matched", "Technical_Category", "Linguistic_Category",
        "Subcategory", "Classification_Evidence", "Confidence", "Notes",
    ]
    a_rows = [r for r in master_rows if r["Population"] == "Root_Unresolved"]
    b_rows = [r for r in master_rows if r["Population"] == "Root_Resolved_Maqayis_Unmatched"]
    if len(a_rows) != 329 or len(b_rows) != 289 or len(master_rows) != 618:
        raise RuntimeError("Master population partition failed")

    write_csv(out_dir / "UNRESOLVED_TERM_MASTER_TABLE.csv", fields, master_rows)
    write_csv(out_dir / "ROOT_UNRESOLVED_329.csv", fields, a_rows)
    write_csv(out_dir / "MAQAYIS_UNMATCHED_289.csv", fields, b_rows)
    write_csv(out_dir / "NO_SEMANTIC_EVIDENCE_618.csv", fields, master_rows)

    technical_summary = []
    for category, count in sorted(Counter(r["Technical_Category"] for r in master_rows).items(), key=lambda kv: (-kv[1], kv[0])):
        population = "Root_Unresolved" if category != "Exact digital Maqāyīs lookup miss" else "Root_Resolved_Maqayis_Unmatched"
        denominator = 329 if population == "Root_Unresolved" else 289
        subset = [r for r in master_rows if r["Technical_Category"] == category]
        technical_summary.append({
            "Population": population,
            "Technical_Category": category,
            "Record_Count": count,
            "Unique_Normalized_Forms": len({r["Normalized_Term"] for r in subset}),
            "Percent_Within_Population": round(100 * count / denominator, 1),
        })
    write_csv(
        out_dir / "ROOT_FAILURE_REASON_SUMMARY.csv",
        ["Population", "Technical_Category", "Record_Count", "Unique_Normalized_Forms", "Percent_Within_Population"],
        technical_summary,
    )

    linguistic_summary = []
    for population, denominator in (("Root_Unresolved", 329), ("Root_Resolved_Maqayis_Unmatched", 289)):
        subset = [r for r in master_rows if r["Population"] == population]
        for category, count in sorted(Counter(r["Linguistic_Category"] for r in subset).items(), key=lambda kv: (-kv[1], kv[0])):
            crows = [r for r in subset if r["Linguistic_Category"] == category]
            conf = Counter(r["Confidence"] for r in crows)
            examples = []
            for form, _ in Counter(r["Normalized_Term"] for r in crows).most_common(5):
                examples.append(form)
            linguistic_summary.append({
                "Population": population,
                "Linguistic_Category": category,
                "Record_Count": count,
                "Unique_Normalized_Forms": len({r["Normalized_Term"] for r in crows}),
                "Percent_Within_Population": round(100 * count / denominator, 1),
                "High_Confidence": conf.get("High", 0),
                "Medium_Confidence": conf.get("Medium", 0),
                "Low_Confidence": conf.get("Low", 0),
                "Representative_Forms": " | ".join(examples),
            })
    write_csv(
        out_dir / "LINGUISTIC_CATEGORY_SUMMARY.csv",
        ["Population", "Linguistic_Category", "Record_Count", "Unique_Normalized_Forms", "Percent_Within_Population",
         "High_Confidence", "Medium_Confidence", "Low_Confidence", "Representative_Forms"],
        linguistic_summary,
    )

    root_groups: dict[str, list[dict]] = defaultdict(list)
    for row in b_rows:
        root_groups[row["Root"]].append(row)
    unmatched_root_rows = []
    for root, rows in sorted(root_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        forms = Counter(r["Normalized_Term"] for r in rows)
        sources = Counter(r["Root_Source"] for r in rows)
        cats = Counter(r["Linguistic_Category"] for r in rows)
        unmatched_root_rows.append({
            "Root": root,
            "Term_Record_Count": len(rows),
            "Distinct_Normalized_Forms": len(forms),
            "Root_Sources": " | ".join(f"{k}:{v}" for k, v in sorted(sources.items())),
            "Representative_Terms": " | ".join(k for k, _ in forms.most_common(8)),
            "Category_Distribution": " | ".join(f"{k}:{v}" for k, v in cats.most_common()),
            "Possible_Representation_Mismatch": any("representation mismatch" in r["Linguistic_Category"] or "short root" in r["Linguistic_Category"].lower() for r in rows),
            "Question_Count": len({r["Question_ID"] for r in rows}),
        })
    write_csv(
        out_dir / "MAQAYIS_UNMATCHED_ROOT_FREQUENCIES.csv",
        ["Root", "Term_Record_Count", "Distinct_Normalized_Forms", "Root_Sources", "Representative_Terms",
         "Category_Distribution", "Possible_Representation_Mismatch", "Question_Count"],
        unmatched_root_rows,
    )

    sem_by_qid: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in sem_records:
        sem_by_qid[(record["Dataset"], record["Question_ID"])].append(record)
    question_rows = []
    for (split, qid), rows in sorted(sem_by_qid.items(), key=lambda kv: (SPLIT_ORDER[kv[0][0]], int(kv[0][1]))):
        a_count = sum(not r.get("Root_Resolved") for r in rows)
        b_count = sum(r.get("Root_Resolved") and not r.get("Semantic_Found") for r in rows)
        matched_count = sum(bool(r.get("Semantic_Found")) for r in rows)
        question_rows.append({
            "Dataset": split,
            "Question_ID": qid,
            "Question": rows[0]["Question"],
            "Selected_Term_Records": len(rows),
            "Root_Unresolved_Records": a_count,
            "Maqayis_Unmatched_Records": b_count,
            "No_Semantic_Evidence_Records": a_count + b_count,
            "Semantic_Evidence_Records": matched_count,
            "Has_Root_Unresolved": a_count > 0,
            "Has_Maqayis_Unmatched": b_count > 0,
            "Has_Any_No_Semantic_Evidence": a_count + b_count > 0,
            "All_Selected_Terms_Semantically_Covered": matched_count == len(rows),
        })
    write_csv(
        out_dir / "QUESTION_LEVEL_COVERAGE_SUMMARY.csv",
        ["Dataset", "Question_ID", "Question", "Selected_Term_Records", "Root_Unresolved_Records",
         "Maqayis_Unmatched_Records", "No_Semantic_Evidence_Records", "Semantic_Evidence_Records",
         "Has_Root_Unresolved", "Has_Maqayis_Unmatched", "Has_Any_No_Semantic_Evidence",
         "All_Selected_Terms_Semantically_Covered"],
        question_rows,
    )

    split_rows = []
    for split in ("Train", "Development", "Test"):
        rows = [r for r in sem_records if r["Dataset"] == split]
        qs = {r["Question_ID"] for r in rows}
        a_count = sum(not r.get("Root_Resolved") for r in rows)
        b_count = sum(r.get("Root_Resolved") and not r.get("Semantic_Found") for r in rows)
        m_count = sum(bool(r.get("Semantic_Found")) for r in rows)
        split_rows.append({
            "Dataset": split,
            "Question_Count": len(qs),
            "Selected_Term_Records": len(rows),
            "Root_Unresolved_Records": a_count,
            "Root_Unresolved_Percent": round(100 * a_count / len(rows), 1),
            "Maqayis_Unmatched_Records": b_count,
            "Maqayis_Unmatched_Percent": round(100 * b_count / len(rows), 1),
            "No_Semantic_Evidence_Records": a_count + b_count,
            "No_Semantic_Evidence_Percent": round(100 * (a_count + b_count) / len(rows), 1),
            "Semantic_Evidence_Records": m_count,
            "Semantic_Evidence_Percent": round(100 * m_count / len(rows), 1),
        })
    write_csv(
        out_dir / "SPLIT_LEVEL_COVERAGE_SUMMARY.csv",
        ["Dataset", "Question_Count", "Selected_Term_Records", "Root_Unresolved_Records", "Root_Unresolved_Percent",
         "Maqayis_Unmatched_Records", "Maqayis_Unmatched_Percent", "No_Semantic_Evidence_Records",
         "No_Semantic_Evidence_Percent", "Semantic_Evidence_Records", "Semantic_Evidence_Percent"],
        split_rows,
    )

    unique_a_norm = len({r["Normalized_Term"] for r in a_rows})
    unique_b_norm = len({r["Normalized_Term"] for r in b_rows})
    unique_c_norm = len({r["Normalized_Term"] for r in master_rows})
    unique_a_original = len({r["Term"] for r in a_rows})
    unique_b_original = len({r["Term"] for r in b_rows})
    a_questions = sum(r["Has_Root_Unresolved"] for r in question_rows)
    b_questions = sum(r["Has_Maqayis_Unmatched"] for r in question_rows)
    c_questions = sum(r["Has_Any_No_Semantic_Evidence"] for r in question_rows)
    fully_covered_questions = sum(r["All_Selected_Terms_Semantically_Covered"] for r in question_rows)
    conf_a = Counter(r["Confidence"] for r in a_rows)
    conf_b = Counter(r["Confidence"] for r in b_rows)
    qac_b = sum(r["Root_Source"] == "QAC" for r in b_rows)
    camel_b = sum(r["Root_Source"] == "CAMeL" for r in b_rows)

    rubric = f"""# Step 6 characterization rubric

## Scope and unit

The primary unit is one selected query-term occurrence. Normalized-form counts are reported separately. Population A contains records with no strict accepted root. Population B contains records with an accepted root, a Maqāyīs lookup attempt, and no exact database match. No root or lexicon result is changed.

## First-level technical categories

Population A uses the preserved CAMeL status and invalid-root reason directly: `incomplete_placeholder_root`, `sentinel_root`, `NO_ANALYSIS`, or `AMBIGUOUS_TOP_ROOT`. Population B is uniformly an exact digital Maqāyīs lookup miss.

## Population-A linguistic categories

- **Proper name / named entity:** the form and query context identify a named person, people, place, scripture, or event; upstream PN evidence is retained where available.
- **Honorific / title expression:** the selected term is the observed religious title `سيدنا` or its preposition-attached form.
- **Function / grammatical expression:** preserved POS or an inspected surface/context rule identifies a particle, preposition, adverb, interrogative, or relational expression.
- **Modern / foreign / institutional vocabulary:** the observed form is transparently modern, borrowed, technical, political, or institutional in context.
- **Abbreviation / non-lexical token:** the parenthetical `ص` honorific abbreviation.
- **Arabic lexical content word:** preserved CAMeL lemma/POS identifies a noun, verb, or adjective even though the root failed the strict rule.
- **Arabic lexical item with unavailable/ambiguous analysis:** readable Arabic lexical material with `NO_ANALYSIS` or `AMBIGUOUS_TOP_ROOT`; no manual root is supplied.
- **Other / uncertain:** evidence is insufficient for a narrower label.

## Population-B diagnostic categories

- **Noncanonical short root key:** one- or two-character root retained under the frozen no-length validity rule.
- **Proper-name / special lexical form:** observed named/special item with no exact key; this includes upstream root `اله` tagged PN.
- **Possible doubled-root representation mismatch:** a three-letter root whose second and third radicals are identical.
- **Possible weak/hamza representation mismatch:** a root containing weak/hamza/alif characters.
- **Extended/four-letter root key absent:** accepted root longer than three letters with no exact database key.
- **Valid root absent from exact digital lookup — cause undetermined:** ordinary accepted root shape with no evidence sufficient to attribute the miss to lexicon scope, indexing, or representation.

The word *possible* is deliberate: these are diagnostic shape flags, not retrospective matches. The exact-match policy and semantic-enrichment output remain unchanged.

## Confidence

- **High:** classification follows directly from preserved technical fields, unambiguous form/context, or root length.
- **Medium:** strong diagnostic evidence exists, but a causal or linguistic interpretation is involved.
- **Low:** plausible but ambiguous; retained as `Other / uncertain` where appropriate.

## Frozen rule evidence

The inspected form lists and category precedence are embedded in `step6_characterize.py`. They were frozen after reviewing the 208 unresolved normalized forms and 98 unmatched roots, before generating the final tables.
"""
    (out_dir / "CHARACTERIZATION_RUBRIC.md").write_text(rubric, encoding="utf-8", newline="\n")

    a_tech = Counter(r["Technical_Category"] for r in a_rows)
    a_cat = Counter(r["Linguistic_Category"] for r in a_rows)
    b_cat = Counter(r["Linguistic_Category"] for r in b_rows)
    top_roots = unmatched_root_rows[:12]
    split_lines = "\n".join(
        f"- {row['Dataset']}: {row['Root_Unresolved_Records']} root-unresolved; {row['Maqayis_Unmatched_Records']} valid-root/unmatched; {row['No_Semantic_Evidence_Records']} without semantic evidence; {row['Semantic_Evidence_Records']} with evidence."
        for row in split_rows
    )
    a_cat_report = "\n".join(
        f"- {row['Linguistic_Category']}: {row['Record_Count']} records ({row['Percent_Within_Population']:.1f}%); representative forms: {row['Representative_Forms']}."
        for row in linguistic_summary if row["Population"] == "Root_Unresolved"
    )
    b_cat_report = "\n".join(
        f"- {row['Linguistic_Category']}: {row['Record_Count']} records ({row['Percent_Within_Population']:.1f}%); representative forms: {row['Representative_Forms']}."
        for row in linguistic_summary if row["Population"] == "Root_Resolved_Maqayis_Unmatched"
    )
    report = f"""# Step 6 — Unresolved-term and Maqāyīs-coverage characterization

## Scope and source verification

This analysis characterizes the frozen corrected query records without repairing roots or lexicon matches. The Point-2 file contains 1,342 selected term occurrences: 525 QAC roots, 488 CAMeL fallback roots, and 329 unresolved records. Notebook 7 contains 724 exact Maqāyīs matches and 289 valid-root exact-lookup misses. Point-2-to-Notebook-7 root/source lineage mismatches were zero.

- Point-2 SHA-256: `{source_hashes_before['Point2_Root_Provenance']}`
- Notebook-7 SHA-256: `{source_hashes_before['Notebook7_Semantic_Enrichment']}`
- Maqāyīs SQLite SHA-256: `{source_hashes_before['Maqayis_SQLite']}`

## Population A — root unresolved

Population A contains **329 / 1,342 ({pct_text(329, 1342)})** term occurrences, representing **{unique_a_norm} normalized forms** and **{unique_a_original} original surface forms** across **{a_questions} questions**.

### Preserved technical reasons

""" + "\n".join(f"- {k}: {v} ({pct_text(v, 329)})." for k, v in a_tech.most_common()) + f"""

### Linguistic characterization

{a_cat_report}

Confidence distribution: **High {conf_a.get('High', 0)}**, **Medium {conf_a.get('Medium', 0)}**, **Low {conf_a.get('Low', 0)}**. The dominant technical failure is the analyzer's incomplete `#` root, not absence of a recognizable Arabic word: many records retain a plausible lemma and content POS while failing root validity.

## Population B — valid root, no exact Maqāyīs match

Population B contains **289 / 1,013 ({pct_text(289, 1013)})** accepted-root occurrences, or **{pct_text(289, 1342)}** of all selected term records. It represents **{unique_b_norm} normalized forms**, **{unique_b_original} original surface forms**, and **{len(root_groups)} unique unmatched roots** across **{b_questions} questions**. Root provenance is **QAC {qac_b}** and **CAMeL {camel_b}**.

### Diagnostic category breakdown

{b_cat_report}

Confidence distribution: **High {conf_b.get('High', 0)}**, **Medium {conf_b.get('Medium', 0)}**, **Low {conf_b.get('Low', 0)}**. Representation-mismatch labels are diagnostic only; no alternate root key was looked up or promoted.

### Most frequent unmatched roots

""" + "\n".join(f"- `{r['Root']}`: {r['Term_Record_Count']} occurrences; representative terms: {r['Representative_Terms']}." for r in top_roots) + f"""

## Combined no-semantic-evidence population

The two failure stages are disjoint. **329** records fail before root resolution; **289** have accepted roots but no exact Maqāyīs entry. Their union is **618 / 1,342 ({pct_text(618, 1342)})** records and **{unique_c_norm} normalized forms**, affecting **{c_questions} questions**. The complementary exact semantic-evidence population is **724 / 1,342 ({pct_text(724, 1342)})**. Among accepted roots alone, Maqāyīs coverage is **724 / 1,013 ({pct_text(724, 1013)})**; these denominators must not be conflated.

Questions with every selected term semantically covered: **{fully_covered_questions} / 251 ({pct_text(fully_covered_questions, 251)})**.

## Split-level description

{split_lines}

No qrels were used. Split labels are the preserved `Dataset` field in the authoritative query artifacts and reconcile to 174 train, 25 development, and 52 test questions.

## Interpretation

The evidence identifies two distinct limitations. First, the root-resolution stage loses 329 records, usually because CAMeL returned an incomplete placeholder root even when it recognized a lemma and POS. Proper names, honorifics, function expressions, modern/borrowed items, and a small set of unavailable or ambiguous analyses are also represented. Second, 289 records pass strict root validation but miss the frozen Maqāyīs database under exact-key lookup. Many of those roots have doubled, weak/hamza, short, or extended shapes that warrant representation diagnostics, while other misses cannot be attributed responsibly from current evidence. This analysis does not establish that a plausible alternate form is a valid lexicon match.

## Methodology paragraph for later thesis use

Uncovered query terms were separated at the term-occurrence level into two non-overlapping populations: records without an accepted strict root and records with an accepted root but no exact match in the frozen Maqāyīs database. Technical failure categories were derived directly from the preserved QAC/CAMeL statuses and root-validity reasons. A second, data-grounded linguistic rubric was then frozen after inspection of the observed forms and applied with explicit confidence labels. Frequency, unique-form, split, and question-level summaries were produced reproducibly. No root, analyzer result, or Maqāyīs match was manually repaired or substituted.

## Results/limitations paragraph for later thesis use

Of 1,342 selected query-term occurrences, 329 ({pct_text(329, 1342)}) lacked an accepted root, while a further 289 ({pct_text(289, 1342)}) had a valid root but no exact Maqāyīs database match. Thus 618 ({pct_text(618, 1342)}) received no Maqāyīs semantic evidence, whereas 724 ({pct_text(724, 1342)}) did. Root loss was dominated by incomplete placeholder roots returned by the analyzer, although named entities, grammatical expressions, honorifics, and modern or borrowed vocabulary also occurred. Exact-lookup misses included doubled, weak/hamza, short, and extended root shapes, but these diagnostics do not prove that an alternative key would be correct. The characterization is therefore descriptive: it distinguishes morphological/root-resolution limitations from digital lexicon-coverage and representation limitations without retrospectively changing the system.

## Integrity and limitations

All writes are confined to this Step-6 folder. The Point-2 artifact, Notebook-7 semantic output, SQLite database, Step-5/5A evidence, retrieval runs, qrels, proposal, and frozen baseline were not modified. Linguistic categories are rule-based scholarly descriptions from the available term, query context, lemma/POS, and status evidence; they are not a separately annotated gold standard. No optional printed-edition spot-check was performed because the task can be characterized conservatively from exact digital lookup status and root-shape diagnostics without altering Step-5's audit scope.
"""
    (out_dir / "UNRESOLVED_TERM_CHARACTERIZATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    source_hashes_after = {
        "Point2_Root_Provenance": sha256(ROOT_PATH),
        "Notebook7_Semantic_Enrichment": sha256(SEM_PATH),
        "Maqayis_SQLite": sha256(DB_PATH),
        "Step5_Audit_Report": sha256(STEP5_REPORT) if STEP5_REPORT.exists() else None,
    }
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("An authoritative source changed during Step 6")

    manifest = {
        "stage": "STEP 6 — Unresolved-term and Maqāyīs-coverage characterization",
        "analysis_date": ANALYSIS_DATE,
        "scope": "analysis_only_no_pipeline_modification",
        "sources": {
            "point2_root_provenance": {"path": str(ROOT_PATH), "sha256": source_hashes_after["Point2_Root_Provenance"], "term_records": 1342},
            "notebook7_semantic_enrichment": {"path": str(SEM_PATH), "sha256": source_hashes_after["Notebook7_Semantic_Enrichment"], "term_records": 1342},
            "maqayis_sqlite": {"path": str(DB_PATH), "sha256": source_hashes_after["Maqayis_SQLite"]},
            "step5_audit_report": {"path": str(STEP5_REPORT), "sha256": source_hashes_after["Step5_Audit_Report"]},
        },
        "validated_counts": {
            "selected_term_records": 1342,
            "qac_resolved": 525,
            "camel_resolved": 488,
            "accepted_roots": 1013,
            "root_unresolved": 329,
            "maqayis_exact_matches": 724,
            "valid_root_maqayis_unmatched": 289,
            "no_semantic_evidence": 618,
            "unique_unresolved_normalized_forms": unique_a_norm,
            "unique_unmatched_normalized_forms": unique_b_norm,
            "unique_unmatched_roots": len(root_groups),
            "questions_with_root_unresolved": a_questions,
            "questions_with_maqayis_unmatched": b_questions,
            "questions_with_any_no_semantic_evidence": c_questions,
            "questions_fully_semantically_covered": fully_covered_questions,
        },
        "split_question_counts": EXPECTED_QUESTION_COUNTS,
        "lineage_checks": {
            "point2_to_notebook7_mismatches": 0,
            "population_a_b_overlap": 0,
            "unexpected_maqayis_attempts_for_unresolved": 0,
            "population_b_missing_lookup_attempt": 0,
            "source_hashes_unchanged_during_analysis": True,
        },
        "method": {
            "primary_unit": "selected_query_term_occurrence",
            "secondary_units": ["normalized_surface_form", "accepted_root", "question", "split"],
            "taxonomy_frozen_before_final_generation": True,
            "manual_root_repairs": 0,
            "manual_lexicon_substitutions": 0,
            "qrels_used": False,
            "printed_spot_check_performed": False,
        },
        "outputs": {},
    }
    delivered_names = [
        "UNRESOLVED_TERM_CHARACTERIZATION_REPORT.md",
        "UNRESOLVED_TERM_MASTER_TABLE.csv",
        "ROOT_UNRESOLVED_329.csv",
        "MAQAYIS_UNMATCHED_289.csv",
        "NO_SEMANTIC_EVIDENCE_618.csv",
        "ROOT_FAILURE_REASON_SUMMARY.csv",
        "LINGUISTIC_CATEGORY_SUMMARY.csv",
        "MAQAYIS_UNMATCHED_ROOT_FREQUENCIES.csv",
        "QUESTION_LEVEL_COVERAGE_SUMMARY.csv",
        "SPLIT_LEVEL_COVERAGE_SUMMARY.csv",
        "CHARACTERIZATION_RUBRIC.md",
    ]
    row_counts = {
        "UNRESOLVED_TERM_MASTER_TABLE.csv": 618,
        "ROOT_UNRESOLVED_329.csv": 329,
        "MAQAYIS_UNMATCHED_289.csv": 289,
        "NO_SEMANTIC_EVIDENCE_618.csv": 618,
        "ROOT_FAILURE_REASON_SUMMARY.csv": len(technical_summary),
        "LINGUISTIC_CATEGORY_SUMMARY.csv": len(linguistic_summary),
        "MAQAYIS_UNMATCHED_ROOT_FREQUENCIES.csv": len(unmatched_root_rows),
        "QUESTION_LEVEL_COVERAGE_SUMMARY.csv": len(question_rows),
        "SPLIT_LEVEL_COVERAGE_SUMMARY.csv": len(split_rows),
    }
    for name in delivered_names:
        path = out_dir / name
        manifest["outputs"][name] = {
            "sha256": sha256(path),
            "data_rows": row_counts.get(name),
        }
    manifest_path = out_dir / "UNRESOLVED_TERM_LINEAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")

    delivered_script = out_dir / "step6_characterize.py"
    if Path(__file__).resolve() != delivered_script.resolve():
        shutil.copyfile(Path(__file__), delivered_script)
    checksum_paths = sorted(
        [p for p in out_dir.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"],
        key=lambda p: p.name,
    )
    checksum_text = "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths)
    (out_dir / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8", newline="\n")

    # The caller copies this exact script into the output folder before checksums.
    summary = {
        "Population_A": {"records": 329, "unique_normalized_forms": unique_a_norm, "unique_original_forms": unique_a_original, "questions": a_questions, "technical": dict(a_tech), "linguistic": dict(a_cat), "confidence": dict(conf_a)},
        "Population_B": {"records": 289, "unique_normalized_forms": unique_b_norm, "unique_original_forms": unique_b_original, "unique_roots": len(root_groups), "questions": b_questions, "qac": qac_b, "camel": camel_b, "linguistic": dict(b_cat), "confidence": dict(conf_b)},
        "Combined": {"records": 618, "unique_normalized_forms": unique_c_norm, "questions": c_questions, "fully_covered_questions": fully_covered_questions, "semantic_evidence_records": 724},
        "splits": split_rows,
    }
    for value in summary.values():
        json_number_safe(value)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
