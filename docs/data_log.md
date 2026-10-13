# Journal des données (chantier A)

Format: **on observe** → **on fait**. Une ligne par observation. À compléter au fil de l'eau.

| # | On observe | On fait | Statut |
|---|---|---|---|
| 1 | Le Code du travail tient dans un seul fichier parquet (262 Mo, 48 993 chunks). Les « 2,2 M de lignes » concernent tous les codes. | Télécharger uniquement `data/legi-latest/legi_code_du_travail/`. | Fait |
| 2 | Le dataset a des snapshots datés (`legi-20250905`, …, `legi-latest`). | Utiliser `legi-latest`. Noter la date d'extraction dans la data card. | Fait |
| 3 | 47 777 articles (`doc_id`), mais seulement 11 479 en `VIGUEUR`. Le reste: MODIFIE (20 056), ABROGE (13 922), etc. | Corpus = `VIGUEUR` seulement. Les autres versions vont dans `article_versions.parquet`. | Décidé |
| 4 | 92 articles `VIGUEUR_DIFF` (en vigueur à une date future). | Les exclure: `etat == VIGUEUR` strict. | Décidé |
| 5 | Chaque version d'un article a son propre `doc_id` (LEGIARTI…). Ex.: L6222-1 a 5 `doc_id`. | Garder `num` (ex. L6222-1) comme clé stable entre versions. Le noter dans la data card. | À faire |
| 6 | La version en vigueur a `end_date = 2999-01-01`. Les versions se suivent sans trou. | Garder tel quel pour `get_article_at_date`. | À faire |
| 7 | 98,5 % des articles n'ont qu'un chunk. 710 en ont plusieurs (max 32). | 1 article = 1 unité de retrieval. Pour les longs, on garde les chunks DINUM. Le texte complet reste dans `articles.parquet`. | Décidé |
| 8 | `chunk_index` commence à 1. `text` = fragment propre du chunk. `chunk_text` = en-tête + fragment. | Reconstituer l'article en concaténant `text` par `chunk_index`. Utiliser `chunk_text` pour la baseline DINUM. | À faire |
| 9 | `subtitles` est irrégulier: espaces multiples, niveaux dupliqués après renommage, « Chapitre unique » sans titre. | Découper sur `\s+-\s+`, normaliser les espaces, garder le dernier intitulé quand un niveau est dupliqué. | À faire |
| 10 | Une « Partie réglementaire ancienne » existe (D118, D322-…), quasi entièrement non en vigueur. | Le filtre `VIGUEUR` l'écarte. Sélectionner par préfixe de `number`, pas par `subtitles` seuls. | À faire |
| 11 | Partie 6 livre II (L/R/D 62xx): 1 171 versions, **301 en `VIGUEUR`**. | Ajouter le livre III (contrat de professionnalisation) et les articles connexes pour viser ~500. | À faire (étape 2) |
