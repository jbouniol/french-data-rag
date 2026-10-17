# Data log (workstream A)

Format: **we observe** → **we do**. One line per observation, numbered by step (1.x, 2.x…). Updated as we go.

Status: **Done** (applied in the code), **Decided** (validated, not coded yet), **To do**.

---

## Step 1: exploring the raw data

Script: `src/ingest/explore.py`

| # | We observe | We do | Status |
|---|---|---|---|
| 1.1 | The Code du travail fits in a single parquet file (262 MB, 48,993 chunks). The "2.2 M rows" figure covers all the codes. | Download only `data/legi-latest/legi_code_du_travail/`. | Done |
| 1.2 | The dataset has dated snapshots (`legi-20250905`, …, `legi-latest`). | Use `legi-latest`. Record the extraction date in the data card. | Done |
| 1.3 | 47,777 articles (`doc_id`), but only 11,479 are `VIGUEUR`. The rest: MODIFIE (20,056), ABROGE (13,922), etc. | Corpus = `VIGUEUR` only. The other versions go to `article_versions.parquet`. | Done |
| 1.4 | 92 `VIGUEUR_DIFF` articles (in force at a future date). | Exclude them: strict `etat == VIGUEUR`. | Done |
| 1.5 | Each version of an article has its own `doc_id` (LEGIARTI…). E.g. L6222-1 has 5 `doc_id`s. | Keep `num` (e.g. L6222-1) as the stable key across versions. Note it in the data card. | Done |
| 1.6 | The version in force has `end_date = 2999-01-01`. Versions follow each other without a gap, except 17 cases (see 3.8). | Keep as is for `get_article_at_date`. | Done |
| 1.7 | 98.5% of articles have a single chunk. 710 have several (max 32). | Rebuild full articles in `articles.parquet` (1 article = 1 retrieval unit, see 3.5). Keep the DINUM chunks in `dinum_baseline_chunks.parquet`. | Done |
| 1.8 | `chunk_index` starts at 1. `text` = clean fragment of the chunk. `chunk_text` = header + fragment. | Rebuild the article by concatenating `text` by `chunk_index` (with `\n`). Use `chunk_text` for the DINUM baseline. | Done |
| 1.9 | `subtitles` is irregular: repeated spaces, duplicated levels after renaming, "Chapitre unique" without a title. | Split on `\s+-\s+`, normalise spaces, keep the last label when a level is duplicated. | Done |
| 1.10 | An old "Partie réglementaire ancienne" exists (D118, D322-…), almost entirely not in force. | The `VIGUEUR` filter drops it. Select by `number` prefix, not by `subtitles` alone. | Done |
| 1.11 | Partie 6 livre II (L/R/D 62xx): 1,171 versions, **301 `VIGUEUR`**. | Add other rules to reach ~500 (step 2). | Done |

---

## Step 2: corpus selection (~500 articles)

Files: `src/ingest/selection.yaml` (rules) and `src/ingest/select.py` (application + per-rule log).

### Findings that drove the rules

| # | We observe | We do | Status |
|---|---|---|---|
| 2.1 | **Livre III of partie 6 is not the contrat de professionnalisation**: it is all continuing vocational training (774 `VIGUEUR` articles: CPF, funding, controls…). | Take only the "Contrats de professionnalisation" chapter (L/R/D 6325, 62 articles). | Done |
| 2.2 | "Jeunes travailleurs": the numbers are scattered (L3161 to L3164, R3162 to R3165, R4153). | Select by `subtitles` ("jeunes travailleurs") rather than by number. | Done |
| 2.3 | Paid leave (L/R/D 3141): 75 articles, 31 of them about the leave funds (caisses de congés), specific to construction. | Exclude the `subtitles` "Caisses de congés". | Done |
| 2.4 | Trial period: 13 articles via `subtitles`, 5 of them outside the standard contract (CDD L1242, temp work L1251, VRP L7313). | Keep them: they also answer questions about the trial period. | Done |
| 2.5 | Contract termination: L/R/D 123x is 354 articles, too many for the topic. | Restrict to 3 chunks (the `*_termination` rules) instead of taking everything. | Done |

### Result per rule

| # | Rule | Criterion | Added | Cumulative |
|---|---|---|---|---|
| 2.6 | `apprenticeship` | L/R/D 62xx | 301 | 301 |
| 2.7 | `professionalization_contract` | L/R/D 6325 | 62 | 363 |
| 2.8 | `young_workers` | subtitles "jeunes travailleurs" | 37 | 400 |
| 2.9 | `paid_leave` | L/R/D 3141, without leave funds | 44 | 444 |
| 2.10 | `trial_period` | subtitles "Période d'essai" | 13 | 457 |
| 2.11 | `cdi_termination` | L/R/D 1231 and 1232 (resignation, dismissal for personal reasons) | 35 | 492 |
| 2.12 | `cdd_termination` | L/R/D 1243 | 17 | 509 |
| 2.13 | `mutual_termination` | L1237-11 to L1237-19 | 31 | 540 |

| # | We observe | We do | Status |
|---|---|---|---|
| 2.14 | Total: **540 articles**, 540 distinct numbers (no duplicate). | Within the 400-600 range of the test. Adjust in `selection.yaml` if the golden set reveals gaps. | Done |

---

## Step 3: produced files and data card

Files: `src/ingest/build.py` (build), `src/ingest/stats.py` (stats), `tests/test_corpus.py`, `docs/data_card.md`.

| # | We observe | We do | Status |
|---|---|---|---|
| 3.1 | Rebuilding by `chunk_index`: 540 articles with no empty text, 542 baseline chunks (2 articles have 2 chunks). | `articles.parquet` = concatenated text. `dinum_baseline_chunks.parquet` keeps the original chunks. | Done |
| 3.2 | Parsed `subtitles`: 536 articles have 4 levels (partie, livre, titre, chapitre), 4 have no chapitre. | `hierarchy` = "code > partie > livre > titre > chapitre". Sections and sub-sections dropped. | Done |
| 3.3 | **14 `MODIFIE_MORT_NE` versions** have `date_fin` < `date_debut` (modified before coming into force). Found by the dates test. | Drop them from `article_versions.parquet` (1,066 versions in the end). | Done |
| 3.4 | 97 articles have a `nota` (application details). | `nota` column added to `articles.parquet`. | Done |
| 3.5 | BGE-M3 tokens: mean 138, p99 805, max 1,006, only 14 articles > 512. | 1 article = 1 retrieval unit is viable. No re-chunking needed. | Done |
| 3.6 | Livre III of partie 6: "La formation professionnelle" (législatif) and "…continue" (réglementaire) for the same chapter. | Not harmonised. Noted in the data card (limits). | Done |
| 3.7 | Downloading the BGE-M3 tokenizer stalls with Xet (as in 1.1). | Set `HF_HUB_DISABLE_XET=1` by default in `stats.py`. | Done |
| 3.8 | Final check: 17 articles have a gap between two versions (an `ABROGE` version, then the article is recreated later). E.g. D6222-33 repealed in 2019, back in force in 2020. | Expected: on those dates the article did not exist. `get_article_at_date` must return "no applicable article", not the neighbouring version. Noted in the data card. | Done |
| 3.9 | In DINUM's `chunk_text`, 133/542 chunks (24.5%) have no hierarchy at all, chunk 2 of `L6231-2` loses its hierarchy, and `nota` is never included. | Add our own `chunk_text` column (`Code du travail - Article {num}\n{hierarchy}\n{text}\nNota : {nota}`) to `articles.parquet`. | Done |
