# P4 — Évaluation d’Alter-Ego

Un script, `p4_eval.py`, pour valider les données, appeler un juge Ollama et produire le rapport. Il utilise les dépendances du projet et les formats réels de P1/P3. Le runner `eval/run_eval.py` reste le chantier de Brian ; il est encore à compléter.

## Fichiers

| Fichier | Rôle |
|---|---|
| `p4_eval.py` | Commandes `validate`, `judge`, `report` |
| `test_p4_eval.py` | Tests des formats, métriques et erreurs avec données synthétiques |
| `questions_draft.jsonl` | 40 propositions, volontairement sans réponse ni source validée |
| `annotations_draft.jsonl` | Catégories, dates et comportements attendus |
| `judge_prompt.md` | Critères et format JSON du juge |
| `README.md` | Formats, protocole et commandes |
| `.gitignore` | Fichiers temporaires Python |

## Entrées de l’équipe

Le corpus est lu directement dans `data/processed/article_versions.parquet`, produit par `src.ingest.build` : `doc_id`, `num`, `text`, `chunk_text`, `hierarchy`, `etat`, `date_debut`, `date_fin`, `url`, `nota`, `is_in_corpus`. Aucun export intermédiaire n’est demandé. Les identifiants `LEGIARTI…` désignent des versions ; un numéro comme `L6222-35` ne suffit pas. Les intervalles sont `[date_debut, date_fin)` ; une fin en 2999 représente une version ouverte.

Le golden final, à remplir après lecture du corpus, contient un objet par ligne :

```json
{"id":"q001","question":"Question vérifiée","answer":"Référence humaine vérifiée","gold_docs":["IDENTIFIANT_VERSION_REELLE"]}
```

Une abstention attendue utilise `gold_docs: []` après confirmation de l’absence de réponse. Ces cas sont exclus du recall/MRR. Une référence inconnue invalide le golden ; une citation inconnue du système est conservée comme erreur à mesurer.

Le runner doit ajouter `id` et idéalement `mode` à `asdict(answer(question, mode=mode))`, sans renommer les champs d’`AnswerTrace` :

```json
{"id":"q001","mode":"rag","answer":"Réponse produite","citations":[],"retrieved_ids":[],"tokens_in":1200,"tokens_out":160,"latency_ms":845.2,"tool_calls":[],"stop_reason":"answered","cost_eur":0.0}
```

Faire un fichier de résultats par mode. En `rag`, `retrieved_ids` est classé : recall@k et MRR sont calculables. En `agent`, c’est une union des sources consultées : seule leur couverture est mesurée. Sans mode connu, le script n’interprète pas cette liste comme un classement ; `report --mode rag` ou `--mode agent` permet de le préciser.

Les annotations finales sont facultatives mais nécessaires à l’analyse par catégorie et date :

```json
{"id":"q001","category":"historical","as_of":"2017-06-01","expected_behavior":"Citer la version applicable, ou reconnaître son absence"}
```

Catégories : `factual`, `synthesis`, `calculation`, `historical`, `out_of_scope`, `injection`. `notes` permet de documenter les hypothèses et `evidence` associe chaque `doc_id` à un extrait vérifié. Les appels de rémunération utilisent `age`, `contract_year`, `licence_pro`, `on_date` ; ils n’acceptent pas un SMIC libre. Conserver dans les annotations la référence indépendante du calcul : date et source officielle du SMIC, taux, hypothèses, formule, arrondi et montant. Ne pas prendre la sortie de l’outil testé comme référence.

## Préparer les 40 références

1. Vérifier la couverture et les versions avec P1 et `docs/data_card.md`. La répartition proposée est 10 factuelles, 6 synthèses, 8 calculs, 8 historiques, 4 hors corpus et 4 injections directes. Réviser les questions selon le corpus réel.
2. Lire les sources, remplacer les placeholders, écrire les réponses et choisir les versions réellement nécessaires. Vérifier les hypothèses conventionnelles et faire relire les calculs/cas ambigus. Garder les extraits justificatifs dans les annotations.
3. Créer `eval/golden.jsonl` et des annotations finales distinctes des brouillons. Les brouillons doivent échouer à la validation ; les renommer ne les rend pas utilisables.
4. Figer le golden et ses versions avant l’évaluation finale. Utiliser d’autres questions pour régler le système ; signaler toute réutilisation des cas finaux pendant les réglages.
5. Noter commit, corpus/index, modèles/version, prompts, machine et paramètres. Exécuter les mêmes questions en RAG puis agent, conserver les erreurs et les cas sans réponse.

## Commandes

Depuis la racine, après `uv sync --group dev`, avec les vrais fichiers :

```bash
uv run python eval/analysis/p4_eval.py validate \
  --questions eval/golden.jsonl --results eval/results.jsonl \
  --annotations eval/analysis/annotations.jsonl --expected-count 40
```

Le corpus natif est utilisé par défaut ; `--corpus CHEMIN` le remplace. Omettre `--results` pour valider les références avant exécution, et `--annotations` si absentes. Le runner de Brian doit d’abord produire `eval/results.jsonl`.

```bash
uv run python eval/analysis/p4_eval.py judge \
  --questions eval/golden.jsonl --results eval/results.jsonl \
  --annotations eval/analysis/annotations.jsonl --model NOM_MODELE_JUGE \
  --out eval/analysis/run01/judgments.jsonl

uv run python eval/analysis/p4_eval.py report \
  --questions eval/golden.jsonl --results eval/results.jsonl \
  --annotations eval/analysis/annotations.jsonl \
  --judgments eval/analysis/run01/judgments.jsonl \
  --k 5 --out eval/analysis/run01/report

uv run pytest eval/analysis/test_p4_eval.py
```

`judge --out` désigne un fichier JSONL neuf ; `report --out` désigne un dossier neuf contenant `report.md` et `checks.jsonl`. Les sorties existantes ne sont pas écrasées.

Ollama doit être démarré et le modèle disponible. Le client lit `OLLAMA_HOST`. Sans `--model`, le juge utilise `OLLAMA_MODEL`, sinon le défaut du projet `gemma4:e4b-it-qat`. Préférer un juge distinct du modèle répondant ; consigner le modèle/version et relire au moins 10 jugements ainsi que tous les cas litigieux. Les données à juger ne peuvent modifier la consigne du juge. Les textes des sources ne sont pas tronqués silencieusement : un dépassement de contexte reste sans score, comme une erreur de service/parsing ; aucun de ces cas ne devient une note zéro ou un succès.

## Mesures et limites

Le rapport distingue classement RAG, couverture des sources agent, existence des citations, validité des dates, scores du juge, tokens, latence et `cost_eur`. Les cas sans source de référence sont exclus des métriques de recherche ; une donnée manquante n’est pas remplacée par zéro. `cost_eur = 0` pour Ollama décrit le coût API déclaré, sans chiffrer le matériel ni l’électricité.

Les traces actuelles tronquent les résultats d’outils à 300 caractères, n’exportent pas le contexte final et ne fournissent pas de montant calculé structuré. Le script ne peut donc pas certifier automatiquement les montants ni attribuer une erreur au retrait d’un passage du contexte. Une citation existante ne prouve pas son soutien textuel ; juge et relecture contrôlent ce point.

Compléter le rapport avec trois erreurs réelles, leur cause étayée ou hypothèse, le responsable de la correction et les risques résiduels : corpus limité, historique/SMIC incomplets, conventions absentes, trace partielle, biais du juge, petit échantillon. Les tests synthétiques ne prouvent ni l’exactitude juridique ni la qualité réelle. Aucun score réel n’est disponible avant corpus, sorties et juge opérationnels.

Livrer golden validé, annotations, configuration du juge, jugements, métriques et analyse. Pour les 2 minutes du pitch : 30 s de méthode, 40 s de résultats, 30 s sur un échec et 20 s sur les limites.
