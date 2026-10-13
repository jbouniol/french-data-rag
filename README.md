# Alter-Ego: l'assistant juridique de l'alternant

Assistant RAG + agent sur le Code du travail (apprentissage), exécuté en local avec Ollama.

## Installation

```bash
uv sync
cp .env.example .env
```

## Données (chantier A)

```bash
uv run python -m src.ingest.build
```

## Évaluation

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

## Tests

```bash
uv run pytest
```
