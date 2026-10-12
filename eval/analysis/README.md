# P4 — Évaluation d’Alter-Ego

Ce dossier contient le chantier P4 : préparation du golden set, juge LLM, analyse d’erreurs et risques résiduels. Les scripts utilisent la bibliothèque standard de Python ; ils ne changent ni le corpus, ni le moteur de recherche, ni l’agent. Aucun service de modèle n’est fourni par ces fichiers.

**Les 40 questions présentes sont un brouillon, pas un golden set validé.** `questions_draft.jsonl` contient volontairement des réponses nulles, des listes de sources vides et le statut `draft_needs_sources`. `annotations_draft.jsonl` décrit les comportements à vérifier. Des placeholders restent à remplir. Il ne faut ni calculer des scores avec ces fichiers, ni les renommer directement en `eval/golden.jsonl`.

## Ressources à récupérer auprès de l’équipe

| Responsable | Ressource nécessaire | Utilisation |
|---|---|---|
| Enzo / P1 | Export exact des articles utilisés : identifiants de versions, texte intégral, numéro, dates de validité et limites connues du corpus | Écrire des réponses de référence vérifiables et valider les citations et dates |
| Enzo / P1 | Data card et confirmation des versions historiques disponibles | Déterminer les questions historiques réellement répondables |
| Brian / P2 | Sorties JSONL d’une exécution sur les questions finales, avec classement des documents et contexte transmis au modèle | Mesurer recall@k, MRR et diagnostiquer les omissions |
| Keira / P3 | Réponse finale, citations, trace des outils, tokens, latence, raison d’arrêt | Vérifier les calculs, l’usage des versions et le tracking |
| Équipe | Date de référence, valeur officielle du SMIC, hypothèses contractuelles et conventionnelles, sources de la grille | Calcul indépendant des montants attendus |
| Utilisateur du juge | Service Ollama ou API compatible OpenAI, nom/version du modèle et, pour l’API, variable secrète déjà injectée | Exécuter le juge LLM |

Les exportations originales peuvent avoir d’autres champs : normaliser une copie pour P4 après accord sur les correspondances. `doc_id` doit identifier une **version précise**, et correspondre aux identifiants des sorties P2/P3. Un numéro comme `L6222-35` ne suffit pas à identifier une version historique. Ne pas remplacer les données de P1. Une source de référence inconnue invalide le golden ; une citation inconnue produite par le système reste une erreur à mesurer, avec avertissement de validation, pas un motif pour exclure sa réponse de l’analyse.

## Formats d’échange

Les chemins ci-dessous sont des exemples à remplacer par les exportations réelles. Les exemples JSON sont des contrats de format, pas des données de référence.

Corpus JSONL :

```json
{"doc_id":"IDENTIFIANT_VERSION_REEL","text":"Texte vérifié de la source","article_number":"L…","valid_from":"AAAA-MM-JJ","valid_to":null}
```

`doc_id` et `text` sont obligatoires. Les métadonnées sont nécessaires aux vérifications qu’elles permettent : en particulier les bornes de validité doivent être présentes pour contrôler une réponse historique. Un article ouvert a `valid_to: null` ; une version historique a une date de fin réelle. Le protocole définit la convention des intervalles et sa conversion depuis les métadonnées LEGI.

Golden set final, après vérification humaine :

```json
{"id":"q001","question":"Question datée et sans placeholder","answer":"Réponse de référence vérifiée, avec hypothèses utiles","gold_docs":["IDENTIFIANT_VERSION_REEL"]}
```

Pour une abstention attendue après confirmation de l’absence de réponse dans le corpus, `gold_docs` vaut `[]` et `answer` décrit l’abstention attendue. Les questions de cette sorte ne participent pas aux métriques de retrieval. La présence d’une liste vide ne dispense pas de cette vérification humaine.

Résultats P2/P3, un enregistrement par question :

```json
{"id":"q001","answer":"Réponse produite","citations":["IDENTIFIANT_VERSION_REEL"],"retrieved_doc_ids":["IDENTIFIANT_VERSION_REEL"],"context_doc_ids":["IDENTIFIANT_VERSION_REEL"],"tokens_in":1200,"tokens_out":160,"latency_ms":845.2,"tool_calls":[],"stop_reason":"stop"}
```

`retrieved_doc_ids` est la liste classée après retrieval/reranking, selon le point de mesure convenu avec Brian. `context_doc_ids` contient les documents effectivement transmis au modèle. Les deux listes ne sont pas interchangeables. Une trace vide n’équivaut pas à une trace absente ; les commandes ne peuvent vérifier l’exhaustivité d’un tracking que l’équipe ne fournit pas. Omettre les champs numériques de tracking inconnus plutôt que les remplir avec zéro ou `null` ; sans ces champs, la validation stricte échoue mais une analyse non stricte peut signaler les valeurs manquantes. P3 doit retirer les secrets et données personnelles inutiles des traces en amont. Conserver aussi l’export brut dans un emplacement approprié pour audit.

Pour le contrôle arithmétique, ajouter à la sortie un champ structuré `calculation` :

```json
{"value":"123.45","unit":"EUR"}
```

La valeur illustre le format uniquement. Utiliser une chaîne décimale exacte et une unité `EUR` ou `percent`. Le contrôle compare ce champ à la référence indépendante des annotations ; il n’extrait pas de nombres du texte de la réponse et ne recalcule pas la législation. Sans ce champ, le calcul reste non évalué.

Annotations finales facultatives pour les contrôles propres à Alter-Ego :

```json
{"id":"q001","category":"historical","as_of":"2017-06-01","expected_behavior":"Utiliser la bonne version et citer la source"}
```

```json
{"id":"q017","category":"calculation","as_of":"2024-01-01","expected_calculation":{"value":"123.45","unit":"EUR","tolerance":"0.01"},"expected_behavior":"Calcul indépendant documenté"}
```

Le montant `123.45` et la date illustrent uniquement le format ; ils ne constituent aucune référence légale. Les catégories sont `factual`, `synthesis`, `calculation`, `historical`, `out_of_scope` et `injection`. `as_of` vaut une date ISO réelle ou `null` si elle n’est pas pertinente ; les dates requises pour les cas temporels doivent être remplies. `value` et `tolerance` sont des chaînes décimales exactes, et non des nombres JSON flottants. L’unité des valeurs de calcul est `EUR` ou `percent`. Un seul taux ou montant structuré est comparé par question ; choisir cette cible et contrôler l’autre élément dans la référence/relecture. Documenter les hypothèses, sources et extraits vérifiés dans un fichier adjacent lorsque le format JSONL ne les représente pas.

## Travail à faire avant une première mesure

1. Lire le corpus réel et remplacer les questions qui ne correspondent pas à sa couverture. Vérifier l’absence de réponse pour les cas hors corpus.
2. Fixer les dates, le SMIC et les hypothèses des calculs. Remplacer tous les placeholders.
3. Pour chaque question, rédiger la réponse attendue depuis les textes et sélectionner les identifiants de versions réels. Faire relire les cas ambigus par un autre membre de l’équipe.
4. Créer `eval/golden.jsonl` avec seulement `id`, `question`, `answer`, `gold_docs`. Créer les annotations finales, séparément des brouillons. Conserver les extraits justificatifs et un calcul indépendant.
5. Geler cette version et la réserver à l’évaluation finale. Utiliser d’autres questions pour régler P2/P3.
6. Demander à P2/P3 une exécution du runner de l’équipe sur ce fichier. Ce dossier ne fournit pas `eval/run_eval.py` : il analyse les sorties de ce runner.

## Commandes

Depuis `/workspace/french-data-rag`, avec les fichiers réels aux chemins indiqués :

```bash
python eval/analysis/validate.py \
  --questions eval/golden.jsonl \
  --corpus /workspace/exports/corpus.jsonl \
  --results /workspace/exports/results.jsonl \
  --expected-count 40 \
  --strict-tracking
```

Les brouillons et les références invalides doivent échouer à la validation. `--expected-count 40` contrôle la taille finale ; son omission permet un jeu de développement plus petit. Commencer sans résultats si l’objectif est de contrôler le golden et le corpus avant l’exécution de l’équipe ; consulter `--help` pour les entrées facultatives.

Avec un modèle Ollama déjà installé et servi localement :

```bash
python eval/analysis/judge.py \
  --questions eval/golden.jsonl \
  --results /workspace/exports/results.jsonl \
  --corpus /workspace/exports/corpus.jsonl \
  --annotations /workspace/exports/annotations.jsonl \
  --provider ollama \
  --model NOM_MODELE \
  --out eval/analysis/run01/judgments.jsonl
```

L’URL locale Ollama par défaut est `http://127.0.0.1:11434`. Utiliser `--base-url` si nécessaire.

Pour une API compatible OpenAI :

```bash
python eval/analysis/judge.py \
  --questions eval/golden.jsonl \
  --results /workspace/exports/results.jsonl \
  --corpus /workspace/exports/corpus.jsonl \
  --annotations /workspace/exports/annotations.jsonl \
  --provider openai \
  --model NOM_MODELE \
  --base-url https://api.openai.com/v1 \
  --api-key-env JUDGE_API_KEY \
  --out eval/analysis/run01/judgments.jsonl
```

Configurer la clé dans les paramètres secrets de l’environnement. Ne jamais inscrire une valeur de clé dans un fichier, une commande documentée ou une conversation. `--api-key-env` désigne le nom de la variable, pas sa valeur ; il peut être omis pour un service compatible local qui ne demande pas d’authentification. L’API doit accepter Chat Completions, `response_format: json_object`, `max_tokens` et une température de zéro.

Le fournisseur hébergé reçoit la question, la réponse de référence, la réponse produite, les annotations et les textes complets des sources de référence, citées et présentes dans le contexte. Les traces brutes d’outils ne sont pas transmises au juge. Utiliser des données dont cet usage a été autorisé. Les noms de modèles doivent être remplacés par les identifiants disponibles, et leurs versions figées consignées dans le rapport. L’accès au service et la qualité du juge devront être validés sur les données réelles. Le prompt effectif est `judge_prompt.md` par défaut ; la sortie du modèle doit contenir un objet `scores` et une chaîne `reason`, comme son exemple. Le script enregistre le jugement validé et ses métadonnées, pas la réponse brute du fournisseur.

Une erreur d’appel ou de parsing est enregistrée avec `status: "error"`, entraîne une sortie non nulle du script et ne crée aucun score automatique. Une erreur HTTP, réseau ou délai dépassé du service interrompt les appels suivants ; les cas non exécutés restent non jugés. Les sorties existantes ne sont pas écrasées : choisir un nouveau chemin de run. Un contexte dépassant la limite explicite `--max-payload-chars` est rejeté avant le premier appel ; aucune source n’est tronquée silencieusement.

Analyse après validation et, si disponible, jugement :

```bash
python eval/analysis/analyze.py \
  --questions eval/golden.jsonl \
  --results /workspace/exports/results.jsonl \
  --corpus /workspace/exports/corpus.jsonl \
  --annotations /workspace/exports/annotations.jsonl \
  --judgments eval/analysis/run01/judgments.jsonl \
  --out eval/analysis/run01/report \
  --k 5
```

Tests techniques avec des données synthétiques :

```bash
python -m unittest discover -s eval/analysis/tests -v
```

Ces tests vérifient le code d’évaluation ; ils ne prouvent ni l’exactitude juridique des références ni la qualité d’Alter-Ego. Aucun score réel n’est annoncé tant que le corpus, les réponses et le modèle du juge ne sont pas disponibles.

## Livraison et pitch P4

Livrer le golden validé, les annotations, les paramètres/version du juge, ses résultats, la synthèse des métriques, trois erreurs expliquées et les risques résiduels. Le protocole complet figure dans `protocol.md`, la consigne du juge dans `judge_prompt.md`.

Pour les deux minutes de présentation : 30 secondes sur le jeu de questions, 40 secondes sur les résultats mesurés, 30 secondes sur un échec instructif et 20 secondes sur les limites. Ne pas convertir des brouillons ou des tests synthétiques en résultats de démonstration.

Le document de cours `project-instructions.pdf` demande environ 40 questions, le format du golden et un runner d’équipe `uv run python eval/run_eval.py`. Le présent dossier fournit les éléments P4 de cette chaîne. La répartition des responsables et le périmètre de fichiers viennent de votre message ; le corpus Alter-Ego vient de votre description du projet. Le pitch est organisé selon vos cinq créneaux de deux minutes.
