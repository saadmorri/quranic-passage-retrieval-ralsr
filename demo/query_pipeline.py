from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd


RAW_STOPWORDS = {
    "أن", "أو", "أين", "إلى", "إن", "التي", "الذي", "الذين", "اللاتي",
    "ب", "بماذا", "بين", "تلك", "ثم", "ذلك", "على", "عن", "في", "ك",
    "كان", "كانت", "كم", "كيف", "ل", "لا", "لم", "لماذا", "لن", "ما",
    "ماذا", "متى", "مع", "من", "هذا", "هذه", "هل", "هم", "هن", "هو",
    "هي", "و",
}
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_LETTERS_ONLY = re.compile(r"^[\u0621-\u063A\u0641-\u064A\u066E-\u06D3\u06FA-\u06FC]+$")
NON_ROOT_SENTINELS = frozenset(
    {"NTWS", "NOAN", "DIGIT", "PUNC", "FOREIGN", "NA", "NAN", "NONE", "NULL", "UNK", "UNKNOWN"}
)


def remove_diacritics(text: str) -> str:
    return ARABIC_DIACRITICS.sub("", text if isinstance(text, str) else "")


def normalize_arabic(text: str) -> str:
    text = remove_diacritics(text if isinstance(text, str) else "").replace("ـ", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    return text.replace("ى", "ي")


def remove_punctuation(text: str) -> str:
    punctuation = string.punctuation + "،؛؟«»…ـ"
    text = text.translate(str.maketrans("", "", punctuation))
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


STOPWORDS = {normalize_arabic(word) for word in RAW_STOPWORDS}


def select_query_terms(question: str) -> List[str]:
    normalized = remove_punctuation(normalize_arabic(question))
    selected, seen = [], set()
    for token in normalized.split():
        if token and token not in STOPWORDS and token not in seen:
            selected.append(token)
            seen.add(token)
    return selected


def validate_camel_root(root: Any) -> Dict[str, Any]:
    if root is None:
        return {"Raw": None, "Normalized": None, "Valid": False, "Invalid_Reason": "missing_root"}
    raw = str(root).strip()
    if not raw or raw.upper() in {"NAN", "NONE", "NULL"}:
        return {"Raw": raw or None, "Normalized": None, "Valid": False, "Invalid_Reason": "missing_root"}
    normalized = ARABIC_DIACRITICS.sub("", raw.replace(".", "")).strip()
    sentinel_key = re.sub(r"[\s._-]+", "", raw.upper())
    if sentinel_key in NON_ROOT_SENTINELS:
        return {"Raw": raw, "Normalized": normalized or None, "Valid": False, "Invalid_Reason": "sentinel_root"}
    if "#" in raw:
        return {"Raw": raw, "Normalized": normalized or None, "Valid": False, "Invalid_Reason": "incomplete_placeholder_root"}
    if not normalized or ARABIC_LETTERS_ONLY.fullmatch(normalized) is None:
        return {"Raw": raw, "Normalized": normalized or None, "Valid": False, "Invalid_Reason": "non_arabic_root"}
    return {"Raw": raw, "Normalized": normalized, "Valid": True, "Invalid_Reason": None}


def normalize_for_camel(word: str) -> str:
    word = normalize_arabic(str(word or "").strip())
    for prefix in ("وال", "بال", "كال", "فال", "لل"):
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            word = word[len(prefix):]
            break
    if word in {"اليه", "اليها", "اليهم", "اليهن", "اليكما", "اليكم", "اليكن"}:
        return word
    if word.startswith("ال") and len(word[2:]) >= 3:
        word = word[2:]
    return word


def score_analysis(word: str, analysis: dict) -> int:
    score = 100 if validate_camel_root(analysis.get("root"))["Valid"] else 0
    lemma = analysis.get("lex")
    if lemma and normalize_for_camel(remove_diacritics(str(lemma))) == normalize_for_camel(word):
        score += 40
    score += {"noun": 30, "noun_prop": 25, "verb": 20, "adj": 15}.get(analysis.get("pos"), 0)
    return score


def select_camel_analysis(word: str, analyses: List[dict]) -> Dict[str, Any]:
    if not analyses:
        return {"resolved": False, "status": "NO_ANALYSIS", "analysis": None, "root": None}
    scored = [(score_analysis(word, analysis), analysis) for analysis in analyses]
    top_score = max(score for score, _ in scored)
    top = [analysis for score, analysis in scored if score == top_score]
    roots = sorted({v["Normalized"] for v in (validate_camel_root(a.get("root")) for a in top) if v["Valid"]})
    if len(roots) > 1:
        return {"resolved": False, "status": "AMBIGUOUS_TOP_ROOT", "analysis": None, "root": None}
    def key(a):
        valid = validate_camel_root(a.get("root"))
        return (valid["Normalized"] or "", str(a.get("lex") or ""), str(a.get("stem") or ""), str(a.get("pos") or ""), str(a.get("bw") or ""))
    selected = sorted(top, key=key)[0]
    validation = validate_camel_root(selected.get("root"))
    if len(roots) == 1:
        return {"resolved": True, "status": "RESOLVED", "analysis": selected, "root": roots[0]}
    status = "ANALYSIS_NO_ROOT" if validation["Invalid_Reason"] == "missing_root" else "ANALYSIS_INVALID_ROOT"
    return {"resolved": False, "status": status, "analysis": selected, "root": None}


class QueryProcessor:
    def __init__(self, qac_lookup: pd.DataFrame, maqayis: Dict[str, str]):
        self.maqayis = maqayis
        qac = qac_lookup.copy()
        for col in ("Word_Normalized", "Lemma_Normalized"):
            qac[col] = qac[col].fillna("").astype(str).str.strip()
        self.word_groups = {key: group for key, group in qac.groupby("Word_Normalized", sort=False) if key}
        self.lemma_groups = {key: group for key, group in qac.groupby("Lemma_Normalized", sort=False) if key}
        self._camel_analyzer = None

    def _qac_group(self, term: str):
        if term in self.word_groups:
            return self.word_groups[term], "Word"
        if term.startswith("ال") and len(term) > 2:
            stripped = term[2:]
            if stripped in self.word_groups:
                return self.word_groups[stripped], "Word (without ال)"
            if stripped in self.lemma_groups:
                return self.lemma_groups[stripped], "Lemma (without ال)"
        if term in self.lemma_groups:
            return self.lemma_groups[term], "Lemma"
        return None, "-"

    def _qac(self, term: str) -> Dict[str, Any]:
        group, lookup_type = self._qac_group(term)
        if group is None or group.empty:
            return {"resolved": False, "status": "NO_MATCH", "root": None, "source": None, "lookup_type": "-"}
        roots = sorted({str(value).strip() for value in group["Root_AR"].dropna() if str(value).strip()})
        if len(roots) == 1:
            return {"resolved": True, "status": "RESOLVED", "root": roots[0], "source": "QAC", "lookup_type": lookup_type}
        status = "LEXICAL_MATCH_NO_ROOT" if len(roots) == 0 else "AMBIGUOUS_ROOT"
        return {"resolved": False, "status": status, "root": None, "source": None, "lookup_type": lookup_type}

    def _camel(self, term: str) -> Dict[str, Any]:
        if self._camel_analyzer is None:
            from camel_tools.morphology.analyzer import Analyzer
            from camel_tools.morphology.database import MorphologyDB
            self._camel_analyzer = Analyzer(MorphologyDB.builtin_db())
        normalized = normalize_for_camel(term)
        selection = select_camel_analysis(term, self._camel_analyzer.analyze(normalized))
        return {
            "resolved": bool(selection["resolved"]),
            "status": selection["status"],
            "root": selection["root"],
            "source": "CAMeL" if selection["resolved"] else None,
            "lookup_type": "CAMeL fallback",
            "normalized": normalized,
        }

    def process(self, question: str) -> Dict[str, Any]:
        terms = select_query_terms(question)
        records = []
        for term in terms:
            result = self._qac(term)
            if not result["resolved"]:
                result = self._camel(term)
            root = result.get("root")
            meaning = self.maqayis.get(root) if root else None
            records.append({
                "term": term,
                "root": root,
                "source": result.get("source"),
                "status": result.get("status"),
                "maqayis_found": bool(meaning),
                "semantic_meaning": meaning,
            })
        roots = list(dict.fromkeys(record["root"] for record in records if record["root"]))
        semantics = list(dict.fromkeys(record["semantic_meaning"] for record in records if record["semantic_meaning"]))
        return {
            "question": question,
            "selected_terms": terms,
            "term_records": records,
            "accepted_roots": roots,
            "semantics": semantics,
            "unresolved_terms": [record["term"] for record in records if not record["root"]],
        }
