# Alter-Ego: the apprentice's legal assistant

RAG + agent assistant on the French Code du travail (apprenticeship), running locally with Ollama.

## Presentation

[![Canva Presentation](https://img.shields.io/badge/Canva-Presentation-00C4CC?style=for-the-badge&logo=canva&logoColor=white)](https://canva.link/wtn42r9nyh7shne)

View the presentation on Canva: [https://canva.link/wtn42r9nyh7shne](https://canva.link/wtn42r9nyh7shne)

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

`eval/golden.jsonl` contains one JSON object per line. Add quality annotations to measure
quality; without them, only the expected answer status is scored.

```json
{"id":"q001","question":"Quelle est la durée maximale d'un contrat d'apprentissage ?","expected_status":"answered","must_include":["trois ans"],"expected_citations":["LEGIARTI..."],"reference_answer":"La durée maximale est de trois ans."}
```

Supported annotations: `expected_status` (`answered` by default), `must_include` (answer
substrings, case-insensitive), `expected_citations` (expected `LEGIARTI...` IDs), and
`reference_answer` (used for a lexical token F1, not semantic grading).

```bash
uv run python -m eval.run_eval --questions eval/golden.jsonl --out results/ --backend local --local-mode rag
```

Each run creates a timestamped directory containing predictions (`predictions.jsonl`),
aggregates (`summary.csv`), and a comparison table (`comparison.md`). Run the agent instead
with `--local-mode agent`. The table reports status accuracy, keyword/citation recall,
optional answer token F1, latency percentiles, token counts, and cost. The quality score is the mean of the available keyword recall, citation recall, and
lexical answer F1 for annotated questions. Status accuracy is reported separately.
This score is a transparent deterministic proxy, not a semantic or human evaluation.

For API comparison, implement a Python callable taking `question: str` and returning a dict
with the same fields as `AnswerTrace` (`answer`, `citations`, `retrieved_ids`, `tokens_in`,
`tokens_out`, `latency_ms`, `cost_eur`, `stop_reason`). Then run:

```bash
uv run python -m eval.run_eval --questions eval/golden.jsonl --out results/ \
  --backend both --local-mode rag --api-adapter my_api_adapter:answer
```

The adapter is intentionally provider-specific; the repository does not currently include an
API backend. Set `cost_eur` to `None` when pricing is unavailable so the report shows `N/A`.
The local cost is the model's direct usage cost (zero); electricity and hardware are not included.

## Tests

```bash
uv run pytest
```
