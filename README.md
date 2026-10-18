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

`eval/golden.jsonl` contains 40 verified legal questions on the apprenticeship corpus (`factual`, `synthesis`, `calculation`, `historical`, `out_of_scope`, `injection`):

```json
{"id":"q001","question":"Quel est l’objet du contrat d’apprentissage et quelle formation permet-il d’obtenir ?","answer":"...","gold_docs":["LEGIARTI000037385863"]}
```

`gold_docs` contains the `LEGIARTI...` version IDs of documents that support the answer (`[]` when the corpus does not contain the answer and abstention is expected).

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
```

This writes `eval/results.jsonl`, one row per question and backend, with the generated answer,
`citations`, `retrieved_ids`, token counts, latency, tool calls, stop reason, and
answer F1 / document recall. `summary.csv` and `comparison.md` report aggregate quality,
latency, tokens, and cost. Run the local agent loop instead with `--local-mode agent`.

To validate the golden dataset, run the Ollama LLM-as-a-judge, and generate the P4 analysis report:

```bash
uv run python eval/analysis/p4_eval.py validate --questions eval/golden.jsonl --expected-count 40
uv run python eval/analysis/p4_eval.py judge --questions eval/golden.jsonl --results eval/results.jsonl --out eval/analysis/run01/judgments.jsonl
uv run python eval/analysis/p4_eval.py report --questions eval/golden.jsonl --results eval/results.jsonl --judgments eval/analysis/run01/judgments.jsonl --out eval/analysis/run01/report
```

For API comparison, implement a Python callable taking `question: str` and returning a dict
with the same fields as `AnswerTrace` (`answer`, `citations`, `retrieved_ids`, `tokens_in`,
`tokens_out`, `latency_ms`, `tool_calls`, `cost_eur`, `stop_reason`). Then run:

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ \
  --backend both --local-mode rag --api-adapter my_api_adapter:answer
```

The adapter is intentionally provider-specific; the repository does not currently include an
API backend. Set `cost_eur` to `None` when pricing is unavailable so the report shows `N/A`.
The local cost is the model's direct usage cost (zero); electricity and hardware are not included.

## Tests

```bash
uv run pytest
```
