from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


SYSTEMS = ("BM25", "Dense E5", "Fixed RALSR", "RALSR + CrossEncoder")
RUN_KEYS = {
    "BM25": "bm25",
    "Dense E5": "dense",
    "Fixed RALSR": "ralsr",
    "RALSR + CrossEncoder": "ralsr_ce",
}


@dataclass(frozen=True)
class DemoConfig:
    step_root: Path
    question_train: Path
    question_dev: Path
    question_test: Path
    qpc_tsv: Path
    processed_qpc_csv: Path
    passage_representation_csv: Path
    qac_word_lookup_csv: Path
    maqayis_db: Path
    normalization_csv: Path
    fixed_ralsr_full_csv: Path
    crossencoder_full_csv: Path
    e5_model_dir: Path
    crossencoder_model_dir: Path
    camel_tools_data: Path
    runtime_dir: Path
    runs: Dict[str, Dict[str, Path]]

    @property
    def dense_embedding_cache(self) -> Path:
        return self.runtime_dir / "e5_qpc_passage_embeddings.npy"

    @property
    def dense_metadata_cache(self) -> Path:
        return self.runtime_dir / "e5_qpc_passage_metadata.csv"

    def validate(self) -> None:
        paths = {
            "question_train": self.question_train,
            "question_dev": self.question_dev,
            "question_test": self.question_test,
            "qpc_tsv": self.qpc_tsv,
            "processed_qpc_csv": self.processed_qpc_csv,
            "passage_representation_csv": self.passage_representation_csv,
            "qac_word_lookup_csv": self.qac_word_lookup_csv,
            "maqayis_db": self.maqayis_db,
            "normalization_csv": self.normalization_csv,
            "fixed_ralsr_full_csv": self.fixed_ralsr_full_csv,
            "crossencoder_full_csv": self.crossencoder_full_csv,
            "e5_model_dir": self.e5_model_dir,
            "crossencoder_model_dir": self.crossencoder_model_dir,
            "camel_tools_data": self.camel_tools_data,
        }
        for system, split_map in self.runs.items():
            for split, path in split_map.items():
                paths[f"run:{system}:{split}"] = path
        missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing Step 21 resources:\n" + "\n".join(missing))
        self.runtime_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> DemoConfig:
    step_root = Path(__file__).resolve().parents[1]
    config_path = Path(
        os.environ.get("STEP21_LOCAL_CONFIG", step_root / "local_config.json")
    ).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Local resource configuration not found: {config_path}. "
            "Copy local_config.example.json and supply authorized local paths."
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    def p(key: str) -> Path:
        return Path(raw[key]).expanduser().resolve()

    runs = {
        system: {split: Path(value).expanduser().resolve() for split, value in split_map.items()}
        for system, split_map in raw["runs"].items()
    }
    cfg = DemoConfig(
        step_root=step_root,
        question_train=p("question_train"),
        question_dev=p("question_dev"),
        question_test=p("question_test"),
        qpc_tsv=p("qpc_tsv"),
        processed_qpc_csv=p("processed_qpc_csv"),
        passage_representation_csv=p("passage_representation_csv"),
        qac_word_lookup_csv=p("qac_word_lookup_csv"),
        maqayis_db=p("maqayis_db"),
        normalization_csv=p("normalization_csv"),
        fixed_ralsr_full_csv=p("fixed_ralsr_full_csv"),
        crossencoder_full_csv=p("crossencoder_full_csv"),
        e5_model_dir=p("e5_model_dir"),
        crossencoder_model_dir=p("crossencoder_model_dir"),
        camel_tools_data=p("camel_tools_data"),
        runtime_dir=p("runtime_dir"),
        runs=runs,
    )
    cfg.validate()
    return cfg
