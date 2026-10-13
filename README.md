# Alter-Ego: l'assistant juridique de l'alternant

Assistant RAG + agent sur le Code du travail (apprentissage), exécuté en local avec Ollama.

## Installation

```bash
uv sync
cp .env.example .env
```

## Données (chantier A)

Source: [`AgentPublic/legi`](https://huggingface.co/datasets/AgentPublic/legi) (Etalab 2.0), uniquement le Code du travail (262 Mo). Détails dans [docs/data_card.md](docs/data_card.md).

```bash
# 1. Télécharger les données brutes dans data/raw (ignoré par git)
HF_HUB_DISABLE_XET=1 uv run hf download AgentPublic/legi --repo-type dataset \
  --include "data/legi-latest/legi_code_du_travail/*" --local-dir data/raw

# 2. Construire data/processed/ (articles, versions, baseline DINUM)
uv run python -m src.ingest.build
```

`HF_HUB_DISABLE_XET=1` évite un blocage du téléchargement observé sur ce dataset. Un `HF_TOKEN` dans `.env` accélère le téléchargement mais n'est pas obligatoire.

## Évaluation

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

## Tests

```bash
uv run pytest
```
