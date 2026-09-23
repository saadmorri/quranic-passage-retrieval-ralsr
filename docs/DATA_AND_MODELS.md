# Data and models

External resources are intentionally not copied into this public package. This keeps the repository reviewable and avoids republishing material with layered or uncertain rights.

| Resource | Included? | Source / acquisition | Expected local configuration |
|---|---|---|---|
| QuranQA 2023 Task A questions, QPC, qrels | No | Official organizer repository: <https://gitlab.com/bigirqu/quran-qa-2023> | `QURANQA_TASK_A_DIR` |
| Qur'an Passage Collection (QPC), 1,266 passages | No | Included upstream with Task A; follow all upstream and Qur'anic-text notices | under Task-A data |
| qid 504 | N/A | Present in official test questions, absent from published test qrels; retained in predictions and excluded from judged metrics | no invented qrel |
| Qur'anic Arabic Corpus morphology 0.4 | No | <https://corpus.quran.com/download/>; observe QAC and Tanzil notices | `QAC_MORPHOLOGY_FILE` |
| CAMeL Tools | Python dependency | Install `camel-tools==1.6.0`; morphology DB `calima-msa-r13` / `morphology-db-msa-r13` 0.4.0 | CAMeL data location |
| Arabic Lexicons / Maqāyīs database | No | `wizsk/arabic_lexicons`, release v3.1.0: <https://github.com/wizsk/arabic_lexicons/releases/tag/v3.1.0> | `MAQAYIS_SQLITE`; table `maqayeesul_luga` |
| Hārūn printed Maqāyīs PDF | No | Copyrighted scholarly comparator; obtain independently through a lawful library/source | not required for retrieval execution |
| Dense model | No weights | <https://huggingface.co/intfloat/multilingual-e5-base>, revision `d128750597153bb5987e10b1c3493a34e5a4502a` | `DENSE_E5_MODEL_DIR` |
| CrossEncoder | No weights | <https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1>, revision `1427fd652930e4ba29e8149678df786c240d8825` | `CROSSENCODER_MODEL_DIR` |

## Recorded checksums

- Dense model weights used in the thesis: `A18A44FAD1D0B46DED15928144138CFF1135D5CC8233BDD90BE5F18822DE09A7`.
- Maqāyīs SQLite used in the thesis: `D39ADF6D3846AA17AE92802C9D7EF530F33E857BB3B1163880100C5C479595EB`.
- Official QPC used in the thesis: `0A86C33C465AB6CF9321924D2C03B23ED72F8360134AE92BA4BD4A90C93BE08C`.

These checksums identify the thesis inputs; the corresponding large/external files are not redistributed here.
