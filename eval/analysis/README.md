# P4 — Évaluation d’Alter-Ego

Quatre fichiers suffisent :

| Fichier | Utilité |
|---|---|
| `p4_eval.py` | Valider les données, appeler le juge Ollama et produire le rapport. La consigne du juge est dans `SYSTEM`. |
| `questions_draft.jsonl` | Les 40 questions candidates, avec catégorie, date et comportement attendu dans `annotation`. |
| `test_p4_eval.py` | Vérifier les formats, métriques, dates et erreurs sur des données synthétiques. |
| `README.md` | Formats, commandes et méthode. |

## Préparer les références

Les questions restent un brouillon : réponses nulles, sources non vérifiées et marqueur `draft_needs_sources`. Lire les articles, remplacer les placeholders et renseigner les réponses et vrais identifiants de versions avant de créer `eval/golden.jsonl`. Retirer le champ `status` seulement après vérification humaine.

Le golden accepte les quatre champs du PDF. `annotation` est facultatif et regroupe les informations d’évaluation dans la même ligne :

```json
{"id":"q001","question":"Question vérifiée","answer":"Réponse de référence vérifiée","gold_docs":["IDENTIFIANT_VERSION_REELLE"],"annotation":{"category":"historical","as_of":"2017-06-01","expected_behavior":"Citer la version applicable"}}
```

Une abstention attendue utilise `gold_docs: []` après vérification de la couverture. Les catégories sont `factual`, `synthesis`, `calculation`, `historical`, `out_of_scope`, `injection`. Dans `annotation`, `notes` documente les hypothèses, `evidence` associe un identifiant à un extrait vérifié et `expected_calculation` peut conserver une référence indépendante (`value`, `unit` EUR/percent, `tolerance`). Les valeurs décimales sont écrites comme des chaînes.

Figer les 40 références pour le test final et utiliser d’autres questions pour régler le système. Conserver les versions du code, du corpus et des modèles. Ne pas prendre l’outil de rémunération testé comme référence : vérifier séparément le SMIC daté, la grille, les hypothèses et l’arrondi.

## Entrées et commandes

Le script lit directement `data/processed/article_versions.parquet` : identifiants `doc_id` des versions LEGIARTI, `text`, `num`, `etat`, `date_debut`, `date_fin`. Les intervalles sont début inclus, fin exclue ; 2999 représente une fin ouverte.

Le runner de Brian doit ajouter `id` et idéalement `mode` à `asdict(answer(question, mode=mode))`. Les champs natifs sont `answer`, `citations`, `retrieved_ids`, `tokens_in`, `tokens_out`, `latency_ms`, `tool_calls`, `stop_reason`, `cost_eur`. Le runner `eval/run_eval.py` reste à implémenter par P2.

Depuis la racine, après installation des dépendances du projet, avec les fichiers réels :

```bash
uv run python eval/analysis/p4_eval.py validate --questions eval/golden.jsonl --expected-count 40
uv run python eval/analysis/p4_eval.py judge --questions eval/golden.jsonl --results eval/results.jsonl --model NOM_MODELE_JUGE --out eval/analysis/run01/judgments.jsonl
uv run python eval/analysis/p4_eval.py report --questions eval/golden.jsonl --results eval/results.jsonl --judgments eval/analysis/run01/judgments.jsonl --out eval/analysis/run01/report
uv run pytest eval/analysis/test_p4_eval.py
```

`--corpus` remplace le chemin Parquet. `validate --results` vérifie aussi les sorties. `--mode rag` ou `--mode agent` précise le mode s’il manque dans les résultats. Les sorties existantes ne sont pas écrasées : `judge` crée un JSONL, `report` crée `report.md` et `checks.jsonl`.

Ollama doit être démarré et le modèle disponible. Le client lit `OLLAMA_HOST`. Sans `--model`, le juge reprend `OLLAMA_MODEL` ou le défaut du projet. Préférer un juge distinct du modèle répondant ; relire au moins dix jugements et tous les cas litigieux. Les erreurs de service, de parsing ou de contexte restent sans score.

## Interpréter les résultats

Recall@k et MRR concernent seulement le RAG : les `retrieved_ids` de l’agent sont une union des sources d’outils, dont on mesure la couverture. Les questions sans document de référence sont exclues des métriques de recherche. Le coût `cost_eur` est le coût d’inférence déclaré ; zéro en local exclut matériel et électricité.

Une citation existante ne prouve pas qu’elle justifie la réponse. Les traces d’outils sont tronquées à 300 caractères : ni montant complet structuré ni contexte final ne permettent un contrôle automatique exhaustif. Compléter le rapport avec trois erreurs réelles expliquées, les corrections proposées et les risques résiduels : corpus/historique incomplets, conventions absentes, SMIC externe, traces partielles et biais du juge. Les tests synthétiques ne constituent pas une évaluation juridique réelle.
