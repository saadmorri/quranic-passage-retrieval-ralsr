from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd

from config import DemoConfig, RUN_KEYS


RUN_COLUMNS = ["QID", "Q0", "Passage_ID", "Rank", "Score", "Tag"]


def parse_json_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    return parsed


def load_questions(config: DemoConfig) -> pd.DataFrame:
    frames = []
    for split, path in (
        ("train", config.question_train),
        ("dev", config.question_dev),
        ("test", config.question_test),
    ):
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["QID", "Question"],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        frame["QID"] = frame["QID"].str.strip().astype(int)
        frame["Split"] = split
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 251 or result["QID"].nunique() != 251:
        raise ValueError("Official QuranQA question set must contain 251 unique qids")
    return result


def load_qpc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["Passage_ID", "Passage_Text"],
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if len(frame) != 1266 or frame["Passage_ID"].nunique() != 1266:
        raise ValueError("Official QPC must contain 1,266 unique passages")
    return frame


def load_run(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=RUN_COLUMNS,
        dtype={"QID": int, "Q0": str, "Passage_ID": str, "Rank": int, "Score": float, "Tag": str},
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    return frame


def load_maqayis(path: Path) -> Dict[str, str]:
    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT word, meanings FROM maqayeesul_luga ORDER BY id"
        ).fetchall()
    lookup: Dict[str, str] = {}
    for word, meaning in rows:
        root = "" if word is None else str(word).strip()
        if root and root not in lookup:
            lookup[root] = "" if meaning is None else str(meaning).strip()
    return lookup


@dataclass
class ResourceStore:
    config: DemoConfig
    questions: pd.DataFrame
    qpc: pd.DataFrame
    passage_representation: pd.DataFrame
    processed_qpc: pd.DataFrame
    qac_lookup: pd.DataFrame
    normalization: pd.DataFrame
    fixed_ralsr_full: pd.DataFrame
    crossencoder_full: pd.DataFrame
    maqayis: Dict[str, str]
    runs: Dict[str, Dict[str, pd.DataFrame]]

    @classmethod
    def load(cls, config: DemoConfig) -> "ResourceStore":
        questions = load_questions(config)
        qpc = load_qpc(config.qpc_tsv)
        passage_representation = pd.read_csv(config.passage_representation_csv, encoding="utf-8-sig")
        processed_qpc = pd.read_csv(config.processed_qpc_csv, encoding="utf-8-sig")
        qac_lookup = pd.read_csv(config.qac_word_lookup_csv, encoding="utf-8-sig")
        normalization = pd.read_csv(config.normalization_csv, encoding="utf-8-sig")
        fixed = pd.read_csv(config.fixed_ralsr_full_csv, encoding="utf-8-sig")
        ce = pd.read_csv(config.crossencoder_full_csv, encoding="utf-8-sig")
        if len(passage_representation) != 1266 or len(processed_qpc) != 1266:
            raise ValueError("Validated QPC representations must contain 1,266 rows")
        runs = {
            system: {split: load_run(path) for split, path in split_map.items()}
            for system, split_map in config.runs.items()
        }
        expected = set(RUN_KEYS.values())
        if set(runs) != expected:
            raise ValueError(f"Frozen run keys must be {sorted(expected)}")
        return cls(
            config=config,
            questions=questions,
            qpc=qpc,
            passage_representation=passage_representation,
            processed_qpc=processed_qpc,
            qac_lookup=qac_lookup,
            normalization=normalization,
            fixed_ralsr_full=fixed,
            crossencoder_full=ce,
            maqayis=load_maqayis(config.maqayis_db),
            runs=runs,
        )
