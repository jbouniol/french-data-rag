# Data card: Code du travail corpus (Alter-Ego)

## Source
- Hugging Face dataset [`AgentPublic/legi`](https://huggingface.co/datasets/AgentPublic/legi) (DINUM), derived from the DILA LEGI database.
- Folder used: `data/legi-latest/legi_code_du_travail/` (1 parquet file, 262 MB, 48,993 chunks, 47,777 articles across all versions).
- **Licence**: Licence Ouverte Etalab 2.0 (free reuse, with attribution).
- **Extraction date**: downloaded on 2026-10-13 from the `legi-latest` snapshot. The most recent version in force in the corpus dates from 2026-08-16.
- The data is never versioned in git (`data/` is ignored). Rebuild: `uv run python -m src.ingest.build`.

## Content
About 500 articles of the Code du travail in force, centred on **apprenticeship** and neighbouring topics for an apprentice.

| File | Content |
|---|---|
| `data/processed/articles.parquet` | **540 `VIGUEUR` articles**, 1 row = 1 article (the retrieval corpus) |
| `data/processed/article_versions.parquet` | 1,066 versions (including the 540 in force) of the same articles: `MODIFIE`, `ABROGE`, `TRANSFERE`. Outside the corpus, to answer "at a given date" |
| `data/processed/dinum_baseline_chunks.parquet` | 542 DINUM chunks + BGE-M3 embeddings of the 540 articles. Comparison baseline only |

Columns of `articles.parquet` and `article_versions.parquet`: `doc_id` (LEGIARTI…), `num` (e.g. `L6222-1`), `text`, `hierarchy`, `etat`, `date_debut`, `date_fin`, `url`, `nota`, `is_in_corpus`.

## Selection rules
Defined in [`src/ingest/selection.yaml`](../src/ingest/selection.yaml). Only `etat == VIGUEUR` articles are kept (`VIGUEUR_DIFF`, in force at a future date, are excluded). A rule adds the articles not already taken by the previous rules.

| Rule | Criterion | Articles added | Why |
|---|---|---|---|
| `apprenticeship` | L/R/D 62xx (partie 6, livre II) | 301 | Core of the topic: contract, training, aid, termination |
| `professionalization_contract` | L/R/D 6325 | 62 | The other work-study contract (the whole livre III is continuing training: 774 articles, too broad) |
| `young_workers` | `subtitles` "jeunes travailleurs" | 37 | Working time, rest, forbidden work for under-18s |
| `paid_leave` | L/R/D 3141, without leave funds | 44 | Frequent question; leave funds = construction, off topic |
| `trial_period` | `subtitles` "Période d'essai" | 13 | Including CDD, temp work, VRP |
| `cdi_termination` | L/R/D 1231 and 1232 | 35 | Resignation, dismissal for personal reasons |
| `cdd_termination` | L/R/D 1243 | 17 | Termination of a CDD |
| `mutual_termination` | L1237-11 to L1237-19 | 31 | Termination by mutual agreement |
| **Total** | | **540** | |

The detailed observations are in [`data_log.md`](data_log.md).

## Statistics
Tokens counted with the BGE-M3 tokenizer (`BAAI/bge-m3`, via `tokenizers`), on the article text without header. Script: `uv run python -m src.ingest.stats`.

| | Value |
|---|---|
| Articles | 540 (269 législatifs, 271 réglementaires) |
| Tokens per article | mean 138, median 96, p90 282, p99 805, max 1,006 |
| Total | ≈ 74,700 tokens |
| Articles > 512 tokens | 14 |
| Articles with `nota` | 97 |

Breakdown per book:

| Partie | Livre | Articles | Mean tokens |
|---|---|---|---|
| 6 législative | II: L'apprentissage | 114 | 155 |
| 6 réglementaire | II: L'apprentissage | 187 | 150 |
| 6 législative | III: La formation professionnelle (chapter contrats de professionnalisation) | 26 | 137 |
| 6 réglementaire | III: La formation professionnelle continue (same chapter) | 36 | 156 |
| 1 législative | II: Le contrat de travail | 76 | 122 |
| 1 réglementaire | II: Le contrat de travail | 19 | 121 |
| 3 législative | I: Durée du travail, repos et congés | 52 | 111 |
| 3 réglementaire | I: Durée du travail, repos et congés | 25 | 90 |
| 4 réglementaire | I: Dispositions générales | 4 | 71 |
| 7 législative | III: Voyageurs, représentants, placiers… | 1 | 25 |

## Cleaning and transformations
- **Article text**: the DINUM chunks are glued back by `chunk_index` (concatenation of the `text` column, without the header added in `chunk_text`). 98.5% of articles have a single chunk.
- **`hierarchy`**: obtained by splitting `subtitles` on ` - ` and normalising whitespace. We keep code (législative/réglementaire), partie, livre, titre, chapitre. Sections and sub-sections are dropped. If a level appears twice (renamed), the last label is kept.
- **`url`**: `https://www.legifrance.gouv.fr/codes/article_lc/{doc_id}`.
- **Dates**: `date_debut` / `date_fin` as dates. The version in force has `date_fin = 2999-01-01`.
- **`MODIFIE_MORT_NE` versions** (14 for the selected articles) removed from `article_versions.parquet`: modified before coming into force, `date_fin` earlier than `date_debut`, never applicable.
- **DINUM embeddings**: decoded (JSON → list of floats). They only serve as a baseline; the final index uses our own chunking and embedding choices.

## Known limits
- **SMIC amount missing**: it is set by a decree outside the Code du travail (only the calculation method is in it). The system must answer that it cannot quote the amount.
- **One `doc_id` per version**: each version of an article has its own LEGIARTI identifier. The stable key across versions is `num`.
- **Gaps in the history**: for 17 articles there is a period with no version (article repealed then recreated, e.g. D6222-33 between 2019 and 2020). At a date in that period, no text is applicable.
- **No collective agreements or case law**: only the Code du travail rules are covered.
- **Thematic scope**: no dismissal for serious misconduct or economic dismissal, no part-time work, no overtime (except for young workers). The golden set must stay in scope.
- **Inconsistent livre III title**: "La formation professionnelle" (législatif) versus "La formation professionnelle continue" (réglementaire) for the same chapter. Not harmonised.
- **Raw text**: cross-references between articles are not resolved, any tables are flattened into text.
- **Freshness**: the corpus reflects `legi-latest` at the extraction date; later changes are not picked up.
