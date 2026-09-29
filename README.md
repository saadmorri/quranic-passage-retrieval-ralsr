# Evaluating Root-Aware Lexical-Semantic Retrieval for Qur'anic Passages Using the Qur'anic Arabic Corpus and Maqāyīs al-Lughah

**Author:** Saad Moh'd Saad Abu Morri  
**Supervisor:** Dr. Mukhammad Muso Abduzhabbarov  
**Institution:** Ala-Too International University  
**Department:** Department of Computer Science  
**Programme:** Data Science Master’s Programme

This repository is the curated source-code and reproducibility package for a Master's thesis on Qur'anic passage retrieval. It evaluates sparse, dense, root-aware lexical-semantic, compact semantic representation, and zero-shot reranking conditions on QuranQA 2023 Task A.

## Research objective

The work examines whether explicit Arabic root evidence from the Qur'anic Arabic Corpus (QAC), CAMeL Tools, and exact-root semantic evidence from *Maqāyīs al-Lughah* can improve passage retrieval. The primary proposed linguistic system is fixed-weight Root-Aware Lexical-Semantic Retrieval (RALSR). A training-selected tuned RALSR is retained only as a diagnostic ablation.

## Held-out test results

These are the frozen **official judged** QuranQA Task A test results (51 judged qids; qid 504 is present in predictions but absent from published test qrels).

| System | MAP@10 | MRR@10 |
|---|---:|---:|
| BM25 | 0.0734 | 0.1742 |
| Dense E5 | **0.0995** | **0.2895** |
| Fixed RALSR | 0.0429 | 0.1352 |
| Tuned RALSR | 0.0408 | 0.1359 |
| Dense E5 + CSR | 0.0719 | 0.1875 |
| RALSR + CrossEncoder | 0.0681 | 0.1818 |

**Dense E5 achieved the strongest overall held-out result.** Full-precision official and answerable-only values are in [`results/final_summary/final_test_system_comparison.csv`](results/final_summary/final_test_system_comparison.csv).

## Primary fixed RALSR configuration

| Feature | Weight |
|---|---:|
| Root Coverage | 0.30 |
| Root Overlap | 0.25 |
| Semantic Overlap | 0.20 |
| Jaccard Root Similarity | 0.15 |
| Passage Root Count | 0.05 |
| Semantic Count | 0.05 |

The fixed weights were specified a priori. The tuned condition is a separate training-grid ablation; no post-test tuning occurred. RALSR uses train-only Min–Max normalization and emits `Passage_ID = -1` only for structural zero-candidate cases.

## Repository guide

- [`src/`](src/) — code-cell snapshots for the linguistic and retrieval pipeline.
- [`scripts/`](scripts/) — final experiment, conversion, evaluation, and analysis runners.
- [`notebooks/`](notebooks/) — curated authoritative/corrected methodology and evidence notebooks.
- [`configs/`](configs/) — fixed configurations and public path template.
- [`results/final_summary/`](results/final_summary/) — compact frozen verification results.
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — pipeline overview.
- [`docs/THESIS_MAPPING.md`](docs/THESIS_MAPPING.md) — thesis-section-to-code map.
- [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md) — external resource acquisition and exclusions.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — environment and execution order.

## Interactive demonstration

The [`demo/`](demo/) application provides a defense-oriented Arabic Qur’anic passage-retrieval interface with two clearly separated modes:

- **QuranQA Benchmark Demonstration** reads the same frozen rankings evaluated in the thesis.
- **Free Arabic Question** runs BM25, Dense E5, fixed RALSR, or zero-shot CrossEncoder reranking over the fixed-RALSR candidates.

The full application is designed for local use because its required QuranQA/QPC, QAC, Maqāyīs, passage-representation, model, and frozen-run resources are deliberately not redistributed in this repository. Copy [`local_config.example.json`](local_config.example.json) to the untracked `local_config.json`, supply authorized local paths, install [`demo/requirements.txt`](demo/requirements.txt), and launch [`run_demo.bat`](run_demo.bat) on Windows. See [`docs/INTERACTIVE_DEMO.md`](docs/INTERACTIVE_DEMO.md) for configuration and scientific-boundary details.

> Benchmark results shown in the thesis are frozen experimental results. Free-query retrieval is an interactive demonstration and is not included in the thesis evaluation metrics.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `configs/paths.example.env` to a local, untracked `.env`-style configuration and set paths to externally acquired resources. The public package intentionally excludes QuranQA/QPC files, the QAC distribution, the Maqāyīs database and printed PDF, model weights, embeddings, and large intermediate results.

## Principal stage order

1. Query preprocessing and term selection.
2. QAC-first root lookup with strict CAMeL fallback validation.
3. Exact-root Maqāyīs semantic enrichment.
4. Passage-side linguistic representation.
5. BM25, Dense E5, and fixed RALSR retrieval.
6. Tuned-RALSR ablation and Dense E5 + CSR extension.
7. Zero-shot CrossEncoder reranking of frozen RALSR candidates.
8. Official-format validation, evaluation, and error analysis.

The authoritative frozen results are supplied for verification; expensive model inference is not run automatically. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## External models

- Dense: `intfloat/multilingual-e5-base`, frozen revision `d128750597153bb5987e10b1c3493a34e5a4502a`.
- CrossEncoder: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, frozen revision `1427fd652930e4ba29e8149678df786c240d8825`.

Model weights are not stored in this repository.

## Citation

Use [`CITATION.cff`](CITATION.cff). No DOI has been assigned.

## Licensing and third-party materials

No broad open-source license is granted by this repository. See [`LICENSE-NOTICE.md`](LICENSE-NOTICE.md). Third-party data, lexicons, model files, and official evaluator assets retain their own terms and are not bundled unless explicitly identified.
