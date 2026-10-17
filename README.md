# Alter-Ego: the apprentice's legal assistant

RAG + agent assistant on the French Code du travail (apprenticeship), running locally with Ollama.

## Presentation

[![Canva Presentation](https://img.shields.io/badge/Canva-Presentation-00C4CC?style=for-the-badge&logo=canva&logoColor=white)](https://canva.link/wtn42r9nyh7shne)

View the presentation on Canva: [https://canva.link/wtn42r9nyh7shne](https://canva.link/wtn42r9nyh7shne)

<div style="position: relative; width: 100%; height: 0; padding-top: 56.2500%; padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden; border-radius: 8px; will-change: transform;">
  <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0; margin: 0;"
    src="https://www.canva.com/design/DAHXz6TTVvA/DL1GiFc2DgeDZdOgsxPMRQ/view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
  </iframe>
</div>

## Installation

```bash
uv sync
cp .env.example .env
```

## Data (workstream A)

Source: [`AgentPublic/legi`](https://huggingface.co/datasets/AgentPublic/legi) (Etalab 2.0), Code du travail only (262 MB). Details in [docs/data_card.md](docs/data_card.md).

```bash
# 1. Download the raw data into data/raw (git-ignored)
HF_HUB_DISABLE_XET=1 uv run hf download AgentPublic/legi --repo-type dataset \
  --include "data/legi-latest/legi_code_du_travail/*" --local-dir data/raw

# 2. Build data/processed/ (articles, versions, DINUM baseline)
uv run python -m src.ingest.build
```

`HF_HUB_DISABLE_XET=1` avoids a download stall observed on this dataset. An `HF_TOKEN` in `.env` speeds up the download but is not required.

## Evaluation

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

## Tests

```bash
uv run pytest
```
