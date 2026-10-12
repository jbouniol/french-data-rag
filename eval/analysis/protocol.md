# Protocole P4 — Alter-Ego

## Périmètre et état initial

P4 prépare les références, contrôle les sorties et décrit les erreurs. Les fichiers restent dans `eval/golden.jsonl` et `eval/analysis/`. P1 fournit les documents et métadonnées ; P2 fournit le retrieval classé et son runner ; P3 fournit les sorties et la trace de l’agent. P4 ne modifie aucun corpus ou outil pour faire réussir un test.

Les 40 questions fournies sont des propositions. Les réponses juridiques, identifiants de sources et montants attendus sont absents, volontairement. Sans export du corpus et contrôle des références, aucun score de qualité n’est interprétable.

## Constitution des 40 cas

| Catégorie | Nombre | Objet principal |
|---|---:|---|
| Factuelle | 10 | Trouver et interpréter une disposition |
| Synthèse | 6 | Croiser plusieurs dispositions et reconnaître les faits manquants |
| Calcul | 8 | Paramètres corrects, grille datée et arithmétique |
| Historique | 8 | Version applicable à une date donnée |
| Hors corpus | 4 | Abstention justifiée après vérification de la couverture |
| Injection | 4 | Résister à des consignes adverses en conservant la tâche utile |
| Total | 40 | |

Chaque cas conserve un identifiant `q001` à `q040`. Réviser une question plutôt que créer un doublon si le corpus impose un changement. Les huit questions historiques sont huit cas à date unique, dont certaines portent sur la même règle à des dates différentes. Cette proximité doit être signalée : quarante cas ne représentent pas quarante situations indépendantes.

La couverture de l’échantillon est indicative, pas exhaustive. Vérifier les thèmes disponibles dans les quelque 500 articles réellement retenus, et les versions historiques présentes. Un cas historique sans version disponible doit devenir une abstention attendue, ou être remplacé en conservant le nombre de cas convenu. Ne pas prétendre mesurer une précision historique sur une source absente.

## Références humaines

Pour chaque cas, lire le texte intégral des articles, identifier leurs versions, conserver un extrait justificatif exact et rédiger une réponse de référence. L’extrait est vérifié par une personne, jamais fabriqué par un modèle. Documenter les ambiguïtés, interprétations et règles conventionnelles exclues. Un second lecteur revoit les questions sensibles, les exceptions et les montants.

Une référence répondable doit avoir un ensemble minimal de sources nécessaires dans `gold_docs`. Inclure des sources obligatoires, pas tous les résultats vaguement pertinents. Sinon les métriques confondent sources nécessaires et sources simplement utiles. Une question à plusieurs voies de justification doit être revue pour que la définition de pertinence soit stable.

Les cas hors corpus ne sont confirmés qu’après examen de l’export exact et de sa data card. Si les documents permettent finalement une réponse, changer la catégorie ou le cas. Pour un manque d’information personnelle, distinguer demande de clarification et absence de disposition juridique : ce ne sont pas les mêmes échecs.

## Dates et calculs

Une réponse juridique dépend de la version applicable. Renseigner `as_of` dans les annotations lorsque la date est déterminante et exprimer la date dans la question. Conserver les bornes normalisées de validité de chaque source et expliquer leur conversion depuis LEGI. La convention retenue pour les contrôles est un intervalle **début inclus, fin exclue**, `[valid_from, valid_to)`, avec fin ouverte si `valid_to` est null. Si l’export exprime une autre convention, convertir explicitement avant de lancer les contrôles. Ne pas conclure à une version correcte lorsque les bornes sont inconnues.

Pour chaque calcul, documenter indépendamment : date de référence, âge, année du contrat, date de début et éléments utiles du parcours, valeur officielle du SMIC avec source, base mensuelle/horaire et temps de travail, taux réglementaire, règles conventionnelles pertinentes ou limite de couverture, formule et arrondi. Ne pas supposer qu’un pourcentage du SMIC est toujours le minimum applicable : la référence doit tenir compte des conditions de droit réellement vérifiées.

La signature `compute_remuneration(age, annee_contrat)` ne permet pas à elle seule de passer une date, un SMIC ou toutes les hypothèses. Obtenir de P3 la version de grille et les paramètres effectifs. Si l’outil ne couvre pas une hypothèse, conserver ce cas comme limite documentée ou demander un changement à P3 ; P4 ne réécrit pas l’outil. Une sortie répétée par le système testé n’est pas un calcul de référence indépendant.

Les annotations numériques finales utilisent `expected_calculation` avec `value`, `unit` (`EUR` ou `percent`) et `tolerance`. `value` et `tolerance` sont des chaînes décimales exactes, pas des nombres JSON flottants. La sortie P3 doit fournir `calculation` avec `value` et `unit` pour le montant ou le taux à comparer. En l’absence de ce champ structuré, le contrôle reste inconnu ; aucun nombre n’est extrait de la réponse en prose. Fixer la tolérance avant de mesurer ; distinguer arrondi au centime et différence de taux. Aucun montant attendu n’est créé dans le brouillon.

## Gel et exécution

Créer un petit jeu de développement distinct pour ajuster retrieval, prompts et outils. Après validation et relecture, geler le golden, noter son empreinte et celle du corpus, et éviter de régler P2/P3 avec les 40 cas finaux. Documenter toute fuite de ces cas dans les réglages ; un score sur un jeu déjà utilisé pour optimiser doit être présenté comme tel.

Fixer avant l’exécution : révision du code, versions du corpus/index, modèle de réponse, embeddings, reranker, prompts, valeurs des paramètres, date des règles de calcul, `k` et machine/service utilisé. P2/P3 exécutent les mêmes questions sans intervenir sur les réponses. Conserver les sorties brutes, exceptions et cas sans réponse ; ne pas éliminer les échecs du dénominateur pour améliorer les scores.

Exporter la réponse, les citations, les documents classés, les passages effectivement envoyés au modèle, les appels d’outils et leurs entrées/sorties, les tokens, la latence et la raison d’arrêt. Si plusieurs étapes de retrieval existent, convenir avec P2 de la liste mesurée et documenter ce point. Les documents des appels d’outils doivent être représentés dans le contexte ou la trace pour que l’ancrage puisse être contrôlé.

## Mesures

Pour une question répondable, avec ensemble de références `G` non vide et documents classés `R`, recall@k vaut `|G ∩ R[:k]| / |G|`. Calculer la moyenne par question ; préciser le dénominateur. MRR vaut la moyenne de l’inverse du rang du premier document de `G`, avec zéro lorsqu’aucun n’est trouvé. Préciser si MRR est calculé sur toute la liste exportée ou limité à un rang. Les questions à `G = []` sont exclues de ces deux métriques, mais participent aux contrôles d’abstention.

Le contexte effectivement transmis apporte une mesure différente : une bonne récupération ne garantit pas que le bon passage soit parvenu au modèle. Distinguer qualité du classement, couverture du contexte et qualité de la réponse.

Le juge attribue exactitude de 0 à 2, ancrage de 0 à 2 lorsqu’il est vérifiable, soutien des citations de 0 à 2 lorsqu’il est vérifiable, et booléens ou valeurs nulles pour abstention et résistance aux injections. Rapporter séparément chaque dimension et le nombre de cas effectivement jugés. Une note manquante ou un juge en échec ne vaut ni succès ni zéro automatiquement.

Les vérifications de présence d’identifiants, de dates et de montants sont des contrôles déterministes complémentaires. Une citation existante n’est pas nécessairement une citation qui soutient la réponse : le juge et la relecture doivent vérifier le texte. Une valeur correcte avec une hypothèse incorrecte ne constitue pas une réponse juridique validée.

Une citation produite dont l’identifiant n’existe pas dans le corpus est conservée comme sortie à évaluer, signalée comme non résolue et pénalisée pour son soutien. Ne pas rejeter toute la réponse ni la supprimer du dénominateur pour cette raison. À l’inverse, un identifiant inconnu dans `gold_docs` est un défaut de référence à corriger avant l’évaluation.

Rapporter tokens et latence avec le nombre de valeurs disponibles ; les manquants ne valent pas zéro. Ne pas annoncer de coût monétaire sans tarif daté et informations suffisantes. Distinguer latence du système répondant et latence du juge. Quarante cas autorisent une description des erreurs, pas une garantie juridique générale.

## Juge LLM et contrôle humain

Le juge reçoit la question, la référence, la réponse produite, les sources effectivement disponibles et les annotations. Toutes ces entrées sont des données non fiables pour les instructions : leur contenu ne peut modifier la consigne du juge. Le juge doit retourner un JSON strict et une raison courte appuyée sur les éléments fournis, sans raisonnement privé ni citation inventée.

Consigner fournisseur, identifiant et version du modèle, température, prompt utilisé, jugement JSON validé, métadonnées disponibles et erreurs de parsing/appel. Le script conserve les scores et raisons validés ; il ne stocke pas la réponse brute du fournisseur. Utiliser la même configuration pour une comparaison. Le juge n’est pas une autorité juridique ; il ne complète pas le corpus à partir de sa mémoire. Contrôler manuellement au moins dix cas, dont une sélection par catégorie, et tous les cas ambigus ou litigieux. Rapporter les désaccords plutôt que masquer les notes initiales.

Les quatre injections proposées sont directes, dans la question utilisateur. Elles ne mesurent pas les injections contenues dans les documents. Pour une campagne complémentaire indirecte, préparer des fixtures isolées, obtenir l’accord de P1/P3 sur le point d’injection et exécuter hors du corpus partagé. P4 ne modifie pas les documents indexés de l’équipe.

## Analyse des erreurs et risques

Pour chaque échec, conserver question, référence, réponse, documents et trace utiles, puis rechercher l’étape responsable : document absent, version absente, filtre incorrect, classement, retrait du contexte, mauvais outil, paramètres incomplets, calcul, interprétation, citation ou abstention. Une cause incertaine reste une hypothèse, avec l’information manquante qui permettrait de la vérifier.

Présenter au moins trois cas détaillés et proposer une correction au responsable du composant. Après une correction, mesurer un nouveau run avec versions et différences explicites. Si le golden a servi à corriger, ne présenter pas ce nouveau score comme une évaluation indépendante.

Le rapport couvre les risques résiduels réellement observés : corpus limité, règles conventionnelles manquantes, historique incomplet, temporalité du SMIC, cas contractuels non couverts, défaillances de tracking, biais/erreurs du juge, proximité de certains cas et faible taille de l’échantillon. La démonstration doit présenter les limites et orienter vers une vérification humaine pour une décision juridique.
