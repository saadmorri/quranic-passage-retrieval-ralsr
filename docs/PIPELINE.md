# Pipeline

```text
QuranQA questions
  -> Arabic normalization and selected query terms
  -> QAC root lookup
  -> CAMeL fallback for unresolved terms
  -> strict accepted-root validation
  -> exact-root Maqāyīs semantic lookup
  -> passage-side CAMeL representation
  -> BM25 / Dense E5 / fixed RALSR
  -> tuned-RALSR diagnostic / Dense E5 + CSR / RALSR + CrossEncoder
  -> official six-column run conversion and validation
  -> MAP@10 and MRR@10 evaluation
  -> no-answer and query-level error analysis
```

## Linguistic lineage

Query terms are resolved QAC-first. CAMeL Tools is a strict fallback, and a syntactically invalid or sentinel root is never accepted. Maqāyīs enrichment is an exact lookup against the accepted root; no root substitution is performed. Passage representations are generated separately from the QPC using the frozen passage-side CAMeL procedure.

## RALSR

Root-aware candidate generation precedes six-feature scoring. Min–Max parameters are calculated from training candidate pairs only, then applied unchanged to development and test. Fixed RALSR is the primary linguistic condition. A zero-candidate query receives one structural `-1` prediction; candidate-bearing queries never abstain.

## Extensions

- **Tuned RALSR:** changes only the six weights, selected on train and checked once on dev.
- **Dense E5 + CSR:** concatenates original text, authoritative roots, and selected Maqāyīs-derived keywords before E5 encoding.
- **RALSR + CrossEncoder:** zero-shot reranking of the unchanged frozen fixed-RALSR candidate set; no score fusion and no candidate recovery.

Qrels are used only during evaluation (and train-only tuned-weight selection), never during prediction construction for the frozen retrieval systems.
