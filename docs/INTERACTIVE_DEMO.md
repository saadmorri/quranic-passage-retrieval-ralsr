# Interactive Arabic Retrieval Demonstration

## Purpose

This Streamlit application is a demonstration interface for the systems already implemented and evaluated in the thesis. It retrieves ranked Qur'anic passages; it is not a chatbot and does not generate religious answers or explanations.

## Modes

### QuranQA Benchmark Demonstration

The benchmark mode reads the frozen six-column train/dev/test runs for BM25, Dense E5, fixed RALSR, and RALSR + CrossEncoder. It does not regenerate benchmark predictions. QID 504 is retained in the complete test display and identified as unjudged in the published test qrels.

### Free Arabic Question

This mode accepts Arabic questions only and performs live application inference. It shows Top 5 or Top 10 passages, but it does not calculate MAP, MRR, AP, accuracy, or official relevance because a new question has no official qrels.

## Local resources

Copy `local_config.example.json` to `local_config.json` and supply paths to your authorized local copies of:

- the official QuranQA train/dev/test question files and QPC;
- the corrected processed QPC;
- the frozen QAC lookup, Maqāyīs database, passage-side representation, and train-only normalization table;
- the frozen official runs for the four displayed systems;
- the local `intfloat/multilingual-e5-base` snapshot at revision `d128750597153bb5987e10b1c3493a34e5a4502a`;
- the local `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` snapshot at revision `1427fd652930e4ba29e8149678df786c240d8825`;
- a CAMeL Tools data directory.

These resources, model weights, and runtime E5 embeddings are intentionally excluded from Git.

## Starting the app

The validated defense machine uses two existing Python environments: one for the thesis NLP dependencies and one for Streamlit. `run_demo.bat` invokes `run_demo.ps1`, starts a localhost-only NLP backend, checks its health, and opens the Streamlit UI at `http://127.0.0.1:8501`.

If your environments use different Python paths, set `STEP21_NLP_PYTHON` and `STEP21_STREAMLIT_PYTHON` before launching. The focused package list is in `demo/requirements.txt`.

## Scientific invariants

- BM25 remains word-based BM25Okapi with the thesis parameters.
- Dense E5 keeps the frozen model, input prefixes, normalized embeddings, and dot-product ranking.
- RALSR candidates require at least one shared accepted root; semantic evidence cannot add candidates.
- The strict accepted-root rule, exact Maqāyīs lookup, six features, train-derived normalization, and fixed weights are unchanged.
- The CrossEncoder reads `[question, passage]` pairs, uses raw single-logit scores, applies no score fusion, and reranks fixed-RALSR candidates only.
- A zero-candidate RALSR query is a structural `-1` condition, not a learned no-answer prediction.

## Cloud status

No full public Streamlit deployment is provided. The app depends on resources that the thesis repository deliberately excludes because redistribution permission was not established. The complete local version is the authoritative defense demonstration.

