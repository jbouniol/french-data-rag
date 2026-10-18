# Alter-Ego: the apprentice's legal assistant

RAG + agent assistant on the French Code du travail (apprenticeship), running locally with Ollama.

## Presentation

[![Canva Presentation](https://img.shields.io/badge/Canva-Presentation-00C4CC?style=for-the-badge&logo=canva&logoColor=white)](https://canva.link/wtn42r9nyh7shne)

View the presentation on Canva: [https://canva.link/wtn42r9nyh7shne](https://canva.link/wtn42r9nyh7shne)

## Models & Versions

All retrieval and generation components run locally on a laptop by default:

| Component | Model Name | Version / Package | Configuration |
|---|---|---|---|
| **Dense Embeddings** | `BAAI/bge-m3` | `sentence-transformers>=5.7.0`, `tokenizers>=0.20` | 1024-d, `max_seq_length=2048`, L2-normalized |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-v2-m3` | `sentence-transformers>=5.7.0` | `max_length=2048`, reranks top-30 RRF pool (`k_rrf=60`) |
| **Lexical Search** | `BM25Okapi` | `rank-bm25>=0.2.2` | Custom tokenizer keeping article numbers (`l6222-35`) as single tokens |
| **Local LLM (Ollama)** | `gemma4:e4b-it-qat` *(default)* / `gemma4:12b` | `ollama>=0.6.3` (`OLLAMA_MODEL`) | `num_ctx=8192`, `temperature=0`, `seed=0`, `think=False`, JSON schema decoding |
| **Cloud API Comparison** | `google/gemini-2.5-flash`, `openai/gpt-4o-mini`, `openai/gpt-4o`, `mistralai/mistral-small-3.2-24b-instruct`, `claude-sonnet-4-6`, `claude-haiku-5-5` | OpenRouter & Anthropic Direct (`app/api_llm.py`) | `temperature=0`, tool calling & structured JSON output, EUR cost tracking |

## Installation

```bash
uv sync
cp .env.example .env
```

## Web Demo UI (workstream E)

Launch the interactive web application (answer + Legifrance sources + live agent trace + deterministic tool sandbox + Local vs API model switcher):

```bash
uv run python -m app --open
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
{"id":"q001","question":"Quel est le rôle du contrat d’apprentissage selon les articles sélectionnés du Code du travail ?","answer":"...","gold_docs":["LEGIARTI000006903996"]}
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

For the Local vs Cloud API comparison (with `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` set in `.env`), use the built-in adapter `app.api_llm:api_answer` (or any custom `package.module:function` callable returning an `AnswerTrace` dict):

```bash
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/ \
  --backend both --local-mode rag --api-adapter app.api_llm:api_answer
```

Set `cost_eur` to `None` when pricing is unavailable so the report shows `N/A`.
The local cost is the model's direct usage cost (`0.00 EUR`); electricity and hardware are not included.

## Tests

```bash
uv run pytest
```
