# Journal des données (chantier A)

Format: **on observe** → **on fait**. Une ligne par observation, numérotée par étape (1.x, 2.x…). À compléter au fil de l'eau.

Statuts: **Fait** (appliqué dans le code), **Décidé** (validé, pas encore codé), **À faire**.

---

## Étape 1: exploration des données brutes

Script: `src/ingest/explore.py`

| # | On observe | On fait | Statut |
|---|---|---|---|
| 1.1 | Le Code du travail tient dans un seul fichier parquet (262 Mo, 48 993 chunks). Les « 2,2 M de lignes » concernent tous les codes. | Télécharger uniquement `data/legi-latest/legi_code_du_travail/`. | Fait |
| 1.2 | Le dataset a des snapshots datés (`legi-20250905`, …, `legi-latest`). | Utiliser `legi-latest`. Noter la date d'extraction dans la data card. | Fait |
| 1.3 | 47 777 articles (`doc_id`), mais seulement 11 479 en `VIGUEUR`. Le reste: MODIFIE (20 056), ABROGE (13 922), etc. | Corpus = `VIGUEUR` seulement. Les autres versions vont dans `article_versions.parquet`. | Décidé |
| 1.4 | 92 articles `VIGUEUR_DIFF` (en vigueur à une date future). | Les exclure: `etat == VIGUEUR` strict. | Décidé |
| 1.5 | Chaque version d'un article a son propre `doc_id` (LEGIARTI…). Ex.: L6222-1 a 5 `doc_id`. | Garder `num` (ex. L6222-1) comme clé stable entre versions. Le noter dans la data card. | Fait |
| 1.6 | La version en vigueur a `end_date = 2999-01-01`. Les versions se suivent sans trou. | Garder tel quel pour `get_article_at_date`. | Fait |
| 1.7 | 98,5 % des articles n'ont qu'un chunk. 710 en ont plusieurs (max 32). | 1 article = 1 unité de retrieval. Pour les longs, on garde les chunks DINUM. Le texte complet reste dans `articles.parquet`. | Décidé |
| 1.8 | `chunk_index` commence à 1. `text` = fragment propre du chunk. `chunk_text` = en-tête + fragment. | Reconstituer l'article en concaténant `text` par `chunk_index`. Utiliser `chunk_text` pour la baseline DINUM. | Fait |
| 1.9 | `subtitles` est irrégulier: espaces multiples, niveaux dupliqués après renommage, « Chapitre unique » sans titre. | Découper sur `\s+-\s+`, normaliser les espaces, garder le dernier intitulé quand un niveau est dupliqué. | Fait |
| 1.10 | Une « Partie réglementaire ancienne » existe (D118, D322-…), quasi entièrement non en vigueur. | Le filtre `VIGUEUR` l'écarte. Sélectionner par préfixe de `number`, pas par `subtitles` seuls. | Fait |
| 1.11 | Partie 6 livre II (L/R/D 62xx): 1 171 versions, **301 en `VIGUEUR`**. | Ajouter d'autres règles pour viser ~500 (étape 2). | Fait |

---

## Étape 2: sélection du corpus (~500 articles)

Fichiers: `src/ingest/selection.yaml` (règles) et `src/ingest/select.py` (application + log par règle).

### Constats qui ont guidé les règles

| # | On observe | On fait | Statut |
|---|---|---|---|
| 2.1 | **Le livre III de la partie 6 n'est pas le contrat de professionnalisation**: c'est toute la formation professionnelle continue (774 articles VIGUEUR: CPF, financement, contrôle…). | Ne prendre que le chapitre « Contrats de professionnalisation » (L/R/D 6325, 62 articles). | Fait |
| 2.2 | « Jeunes travailleurs »: les numéros sont éparpillés (L3161 à L3164, R3162 à R3165, R4153). | Sélection par `subtitles` (« jeunes travailleurs ») plutôt que par numéro. | Fait |
| 2.3 | Congés payés (L/R/D 3141): 75 articles dont 31 sur les caisses de congés, spécifiques au BTP. | Exclure les `subtitles` « Caisses de congés ». | Fait |
| 2.4 | Période d'essai: 13 articles via `subtitles`, dont 5 hors contrat standard (CDD L1242, intérim L1251, VRP L7313). | Les garder: ils répondent aussi à des questions sur l'essai. | Fait |
| 2.5 | Rupture du contrat: L/R/D 123x fait 354 articles, trop pour le thème. | Restreindre à 3 morceaux (règles `rupture_*`) au lieu de tout prendre. | Fait |

### Résultat par règle

| # | Règle | Critère | Ajoutés | Cumul |
|---|---|---|---|---|
| 2.6 | `apprentissage` | L/R/D 62xx | 301 | 301 |
| 2.7 | `contrat_professionnalisation` | L/R/D 6325 | 62 | 363 |
| 2.8 | `jeunes_travailleurs` | subtitles « jeunes travailleurs » | 37 | 400 |
| 2.9 | `conges_payes` | L/R/D 3141, hors caisses de congés | 44 | 444 |
| 2.10 | `periode_essai` | subtitles « Période d'essai » | 13 | 457 |
| 2.11 | `rupture_cdi` | L/R/D 1231 et 1232 (démission, licenciement pour motif personnel) | 35 | 492 |
| 2.12 | `rupture_cdd` | L/R/D 1243 | 17 | 509 |
| 2.13 | `rupture_conventionnelle` | L1237-11 à L1237-19 | 31 | 540 |

| # | On observe | On fait | Statut |
|---|---|---|---|
| 2.14 | Total: **540 articles**, 540 numéros distincts (pas de doublon). | Dans la fourchette 400-600 du test. À ajuster dans `selection.yaml` si le golden set révèle des trous. | Fait |

---

## Étape 3: fichiers produits et data card

Fichiers: `src/ingest/build.py` (build), `src/ingest/stats.py` (stats), `tests/test_corpus.py`, `docs/data_card.md`.

| # | On observe | On fait | Statut |
|---|---|---|---|
| 3.1 | Reconstitution par `chunk_index`: 540 articles sans texte vide, 542 chunks baseline (2 articles à 2 chunks). | `articles.parquet` = texte recollé. `dinum_baseline_chunks.parquet` garde les chunks d'origine. | Fait |
| 3.2 | `subtitles` parsé: 536 articles ont 4 niveaux (partie, livre, titre, chapitre), 4 n'ont pas de chapitre. | `hierarchy` = « code > partie > livre > titre > chapitre ». Sections et sous-sections retirées. | Fait |
| 3.3 | **14 versions `MODIFIE_MORT_NE`** ont `date_fin` < `date_debut` (modifiées avant d'entrer en vigueur). Trouvé par le test sur les dates. | Les retirer de `article_versions.parquet` (1 066 versions au final). | Fait |
| 3.4 | 97 articles ont un `nota` (précisions d'application). | Colonne `nota` ajoutée à `articles.parquet`. | Fait |
| 3.5 | Tokens BGE-M3: moyenne 138, p99 805, max 1 006, seulement 14 articles > 512. | 1 article = 1 unité de retrieval est viable. Pas de re-découpage nécessaire. | Décidé |
| 3.6 | Livre III de la partie 6: « La formation professionnelle » (législatif) et « …continue » (réglementaire) pour le même chapitre. | Non harmonisé. Noté dans la data card (limites). | Fait |
| 3.7 | Le téléchargement du tokenizer BGE-M3 se bloque avec Xet (comme en 1.1). | Lancer `stats.py` avec `HF_HUB_DISABLE_XET=1`. | Fait |

