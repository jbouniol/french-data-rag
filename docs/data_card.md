# Data card: corpus Code du travail (Alter-Ego)

## Source
- Dataset Hugging Face [`AgentPublic/legi`](https://huggingface.co/datasets/AgentPublic/legi) (DINUM), issu de la base LEGI de la DILA.
- Dossier utilisé: `data/legi-latest/legi_code_du_travail/` (1 fichier parquet, 262 Mo, 48 993 chunks, 47 777 articles toutes versions confondues).
- **Licence**: Licence Ouverte Etalab 2.0 (réutilisation libre avec mention de la source).
- **Date d'extraction**: téléchargé le 13/10/2026 depuis le snapshot `legi-latest`. La version en vigueur la plus récente du corpus date du 16/08/2026.
- Les données ne sont jamais versionnées dans git (`data/` est ignoré). Reconstruction: `uv run python -m src.ingest.build`.

## Contenu
Environ 500 articles du Code du travail en vigueur, centrés sur l'**apprentissage** et les sujets voisins pour un alternant.

| Fichier | Contenu |
|---|---|
| `data/processed/articles.parquet` | **540 articles** `VIGUEUR`, 1 ligne = 1 article (le corpus de retrieval) |
| `data/processed/article_versions.parquet` | 1 066 versions (dont les 540 en vigueur) des mêmes articles: `MODIFIE`, `ABROGE`, `TRANSFERE`. Hors corpus, pour répondre « à une date donnée » |
| `data/processed/dinum_baseline_chunks.parquet` | 542 chunks DINUM + embeddings BGE-M3 des 540 articles. Baseline de comparaison uniquement |

Colonnes de `articles.parquet` et `article_versions.parquet`: `doc_id` (LEGIARTI…), `num` (ex. `L6222-1`), `text`, `hierarchy`, `etat`, `date_debut`, `date_fin`, `url`, `nota`, `is_in_corpus`.

## Règles de sélection
Définies dans [`src/ingest/selection.yaml`](../src/ingest/selection.yaml). Seuls les articles `etat == VIGUEUR` sont retenus (les `VIGUEUR_DIFF`, en vigueur à une date future, sont exclus). Une règle ajoute les articles non déjà pris par les règles précédentes.

| Règle | Critère | Articles ajoutés | Pourquoi |
|---|---|---|---|
| `apprentissage` | L/R/D 62xx (partie 6, livre II) | 301 | Cœur du sujet: contrat, formation, aides, rupture |
| `contrat_professionnalisation` | L/R/D 6325 | 62 | L'autre contrat en alternance (le livre III entier est de la formation continue: 774 articles, trop large) |
| `jeunes_travailleurs` | `subtitles` « jeunes travailleurs » | 37 | Durée du travail, repos, travaux interdits des moins de 18 ans |
| `conges_payes` | L/R/D 3141, hors caisses de congés | 44 | Question fréquente; caisses de congés = BTP, hors sujet |
| `periode_essai` | `subtitles` « Période d'essai » | 13 | Dont CDD, intérim, VRP |
| `rupture_cdi` | L/R/D 1231 et 1232 | 35 | Démission, licenciement pour motif personnel |
| `rupture_cdd` | L/R/D 1243 | 17 | Rupture du CDD |
| `rupture_conventionnelle` | L1237-11 à L1237-19 | 31 | Rupture d'un commun accord |
| **Total** | | **540** | |

Le détail des observations est dans [`data_log.md`](data_log.md).

## Statistiques
Tokens comptés avec le tokenizer BGE-M3 (`BAAI/bge-m3`, via `tokenizers`), sur le texte de l'article sans en-tête. Script: `uv run python -m src.ingest.stats`.

| | Valeur |
|---|---|
| Articles | 540 (269 législatifs, 271 réglementaires) |
| Tokens par article | moyenne 138, médiane 96, p90 282, p99 805, max 1 006 |
| Total | ≈ 74 700 tokens |
| Articles > 512 tokens | 14 |
| Articles avec `nota` | 97 |

Répartition par livre:

| Partie | Livre | Articles | Tokens moyens |
|---|---|---|---|
| 6 législative | II: L'apprentissage | 114 | 155 |
| 6 réglementaire | II: L'apprentissage | 187 | 150 |
| 6 législative | III: La formation professionnelle (chap. contrats de professionnalisation) | 26 | 137 |
| 6 réglementaire | III: La formation professionnelle continue (idem) | 36 | 156 |
| 1 législative | II: Le contrat de travail | 76 | 122 |
| 1 réglementaire | II: Le contrat de travail | 19 | 121 |
| 3 législative | I: Durée du travail, repos et congés | 52 | 111 |
| 3 réglementaire | I: Durée du travail, repos et congés | 25 | 90 |
| 4 réglementaire | I: Dispositions générales | 4 | 71 |
| 7 législative | III: Voyageurs, représentants, placiers… | 1 | 25 |

## Nettoyage et transformations
- **Texte d'un article**: les chunks DINUM sont recollés par `chunk_index` (concaténation de la colonne `text`, sans l'en-tête ajouté dans `chunk_text`). 98,5 % des articles n'ont qu'un chunk.
- **`hierarchy`**: obtenue en découpant `subtitles` sur ` - `, en normalisant les espaces. On garde code (législative/réglementaire), partie, livre, titre, chapitre. Les sections et sous-sections sont retirées. Si un niveau apparaît deux fois (renommage), on garde le dernier intitulé.
- **`url`**: `https://www.legifrance.gouv.fr/codes/article_lc/{doc_id}`.
- **Dates**: `date_debut` / `date_fin` en dates. La version en vigueur a `date_fin = 2999-01-01`.
- **Versions `MODIFIE_MORT_NE`** (14 pour les articles retenus) retirées de `article_versions.parquet`: modifiées avant leur entrée en vigueur, `date_fin` antérieure à `date_debut`, jamais applicables.
- **Embeddings DINUM**: décodés (JSON → liste de floats). Ils ne servent que de baseline; l'index final utilise nos propres choix de découpage et d'embedding.

## Limites connues
- **Montant du SMIC absent**: il est fixé par décret hors Code du travail (seule la méthode de calcul est dedans). Le système doit répondre qu'il ne peut pas citer le montant.
- **Un `doc_id` par version**: chaque version d'un article a son propre identifiant LEGIARTI. La clé stable entre versions est `num`.
- **Trous dans l'historique**: pour 17 articles, il y a une période sans version (article abrogé puis recréé, ex. D6222-33 entre 2019 et 2020). À une date de cette période, aucun texte n'est applicable.
- **Pas de conventions collectives ni de jurisprudence**: seules les règles du Code du travail sont couvertes.
- **Périmètre thématique**: ni rupture pour faute grave ou licenciement économique, ni temps partiel, ni heures supplémentaires (hors jeunes). Le golden set doit rester dans le périmètre.
- **Intitulé du livre III incohérent**: « La formation professionnelle » (législatif) contre « La formation professionnelle continue » (réglementaire) pour le même chapitre. Non harmonisé.
- **Texte brut**: les renvois entre articles ne sont pas résolus, les tableaux éventuels sont aplatis en texte.
- **Fraîcheur**: le corpus reflète `legi-latest` à la date d'extraction; les changements ultérieurs ne sont pas pris en compte.
