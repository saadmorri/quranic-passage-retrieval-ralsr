# Thesis-to-code map
| Thesis section | Method | Authoritative/public notebook | Script or source snapshot |
|---|---|---|---|
| 3.4 | Query preprocessing and term selection | notebooks/preprocessing/04_query_preprocessing.ipynb | src/preprocessing/query_preprocessing.py |
| 3.5 | QAC lookup | notebooks/linguistic_pipeline/05_qac_root_lookup.ipynb | src/morphology/qac_root_lookup.py |
| 3.6–3.7 | CAMeL fallback and strict root validity | notebooks/linguistic_pipeline/06_camel_fallback_strict.ipynb | src/morphology/camel_fallback_strict.py |
| 3.8 | Maqāyīs exact-root semantic enrichment | notebooks/linguistic_pipeline/07_maqayis_enrichment.ipynb | src/semantics/maqayis_enrichment.py |
| 3.9 | Passage-side processing | notebooks/linguistic_pipeline/08_qpc_preparation.ipynb; notebooks/linguistic_pipeline/09a_passage_linguistic_representation.ipynb | src/morphology/qpc_preparation.py; src/morphology/passage_linguistic_representation.py |
| 3.10 | BM25 | notebooks/retrieval/10a_bm25.ipynb | src/retrieval/bm25_baseline.py |
| 3.11 | Dense E5 | notebooks/retrieval/10b_dense_e5.ipynb | src/retrieval/dense_e5_baseline.py |
| 3.12–3.16 | RALSR candidates, six features, normalization, fixed scoring, no-answer | notebooks/retrieval/09b_ralsr_candidate_generation.ipynb; notebooks/retrieval/09c_ralsr_split_safe_scoring.ipynb | src/retrieval/ralsr_candidate_generation.py; src/retrieval/ralsr_fixed_scoring.py |
| 3.17 | Tuned RALSR ablation | notebooks/evidence/tuned_ralsr_ablation.ipynb | scripts/tuned_ralsr_ablation.py |
| 3.18 | CSR construction and Dense E5 + CSR | notebooks/retrieval/09d_csr_construction.ipynb; notebooks/retrieval/10d_dense_csr.ipynb | src/semantics/csr_construction.py; src/retrieval/dense_csr_retrieval.py; scripts/step9_dense_csr_evaluation.py |
| 3.19 | CrossEncoder reranking | notebooks/evidence/10_crossencoder_final_evaluation.ipynb | scripts/step10_crossencoder_final_evaluation.py |
| 3.20 | Official evaluation and no-answer protocol | notebooks/evidence/07_evaluation_protocol_and_core_results.ipynb | scripts/evaluation/ |
| 4.2 | Coverage and unresolved-term analysis | notebooks/evidence/06_unresolved_term_characterization.ipynb | scripts/step6_unresolved_term_characterization.py |
| 4.3–4.8 | Results and error analysis | notebooks/evidence/08_core_error_analysis.ipynb; notebooks/evidence/09_dense_csr_final_evaluation.ipynb; notebooks/evidence/10_crossencoder_final_evaluation.ipynb | scripts/step8_core_error_analysis.py; results/final_summary/ |

Original authoritative filenames and source hashes are recorded in `notebooks/NOTEBOOK_PROVENANCE.csv` and the local Step-18 included-file register.
