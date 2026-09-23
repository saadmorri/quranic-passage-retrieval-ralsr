# Reproducibility

## Recorded environment

The corrected/final experiments were run on Windows 10 x64 with Python 3.12.x. Package versions are pinned in `requirements.txt`. The original Dense notebook did not print package versions; the preserved `nlp_thesis` environment and later frozen runs established the dependency versions used for reproducibility. CPU was recorded for Dense E5 and CrossEncoder inference.

## Configure external paths

1. Acquire resources listed in `DATA_AND_MODELS.md`.
2. Copy `configs/paths.example.env` to a local untracked configuration.
3. Set `THESIS_WORKSPACE`, `THESIS_CONTROL_ROOT`, model paths, and external-data paths.

Public notebook/script copies use placeholders instead of the author's machine-specific paths. This is the only portability transformation; no scoring, root, normalization, candidate, or ranking rule was changed.

## Recommended execution order

1. `notebooks/preprocessing/04_query_preprocessing.ipynb`
2. `notebooks/linguistic_pipeline/05_qac_root_lookup.ipynb`
3. `notebooks/linguistic_pipeline/06_camel_fallback_strict.ipynb`
4. `notebooks/linguistic_pipeline/07_maqayis_enrichment.ipynb`
5. `notebooks/linguistic_pipeline/08_qpc_preparation.ipynb`
6. `notebooks/linguistic_pipeline/09a_passage_linguistic_representation.ipynb`
7. `notebooks/retrieval/09b_ralsr_candidate_generation.ipynb`
8. `notebooks/retrieval/09c_ralsr_split_safe_scoring.ipynb`
9. `notebooks/retrieval/10a_bm25.ipynb` and `10b_dense_e5.ipynb`
10. `notebooks/retrieval/09d_csr_construction.ipynb` and `10d_dense_csr.ipynb`
11. Final scripts in `scripts/` for tuning, official evaluation, error analysis, CSR evaluation, and CrossEncoder reranking.

## Expensive steps

Dense embedding generation and CrossEncoder candidate scoring download/use external pretrained models and are computationally expensive. The thesis CrossEncoder run scored 20,148 frozen RALSR candidate pairs. The public package therefore supplies compact frozen result summaries instead of model weights, embeddings, or checkpoints.

## Frozen results

`results/final_summary/final_test_system_comparison.csv` is the authoritative compact public verification table. It is copied byte-for-byte from the final CrossEncoder evidence table. The repository does not automatically recompute or overwrite it. Evaluation runners expect the official QuranQA resources and frozen internal runs to be supplied locally.

## Evaluation

Official-format runs have six tab-separated fields: `qid Q0 Passage_ID rank score tag`. The public evaluation helpers convert/validate this format and independently calculate MAP@10/MRR@10. The organizers' original checker/scorer are not copied here; obtain them from the official QuranQA repository and verify their identities as documented in the notebooks.

## Determinism and safeguards

- deterministic secondary ordering uses numeric `Passage_ID` ascending where specified;
- train-only feature normalization is frozen for dev/test;
- qrels do not enter retrieval construction;
- test results were exposed only after all core/tuned configurations were frozen;
- large authoritative scientific artifacts remain in the private archival workspace and are referenced by hashes rather than republished.
