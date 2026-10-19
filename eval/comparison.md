# Comparaison des backends

| Backend | N | Erreurs | F1 réponse | Rappel docs | Rappel citations | Latence moy. (ms) | P50 (ms) | P95 (ms) | Coût total (€) | Tokens entrée | Tokens sortie |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local-rag | 40 | 0 | 46.3% | 76.4% | 43.5% | 28685.06 | 27478.25 | 46020.97 | 0.00 | 109485 | 3886 |

F1 réponse = F1 lexical par tokens entre la réponse du modèle et `answer` du golden. Rappel docs/citations sont non définis (`N/A` dans le tableau) quand `gold_docs` est vide.
