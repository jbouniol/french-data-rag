"""Run evaluation questions and write one AnswerTrace-style JSONL record per run."""
import argparse
from collections import Counter
import csv
import importlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Callable


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {err.msg}") from err
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if not isinstance(item.get("id"), str) or not item["id"].strip():
                raise ValueError(f"{path}:{line_number}: expected a non-empty 'id' string")
            if not isinstance(item.get("question"), str) or not item["question"].strip():
                raise ValueError(f"{path}:{line_number}: expected a non-empty 'question' string")
            if not isinstance(item.get("answer"), str):
                raise ValueError(f"{path}:{line_number}: expected an 'answer' string")
            if not isinstance(item.get("gold_docs"), list) or not all(
                isinstance(doc_id, str) for doc_id in item["gold_docs"]
            ):
                raise ValueError(f"{path}:{line_number}: expected 'gold_docs' to be a list of document IDs")
            questions.append(item)
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {path}")
    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def load_callable(spec: str) -> Callable[[str], Any]:
    """Load an API adapter written as 'package.module:function'."""
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("API adapter must use the format 'package.module:function'")
    adapter = getattr(importlib.import_module(module_name), function_name)
    if not callable(adapter):
        raise TypeError(f"API adapter is not callable: {spec}")
    return adapter


def run_local(question: str, mode: str) -> dict[str, Any]:
    from src.agent.answer import answer

    result = answer(question, mode=mode)
    return result.__dict__


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def token_f1(reference: str, prediction: str) -> float:
    expected, actual = tokens(reference), tokens(prediction)
    if not expected or not actual:
        return float(not expected and not actual)
    overlap = sum((Counter(expected) & Counter(actual)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(actual), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def score(item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    gold_docs = set(item["gold_docs"])
    retrieved = set(result.get("retrieved_doc_ids") or [])
    cited = set(result.get("cited_sources") or [])
    return {
        "answer_token_f1": token_f1(item["answer"], str(result.get("answer", ""))),
        "retrieved_doc_recall": len(gold_docs & retrieved) / len(gold_docs) if gold_docs else None,
        "cited_source_recall": len(gold_docs & cited) / len(gold_docs) if gold_docs else None,
    }


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms") is not None]
    costs_complete = all(row.get("cost_eur") is not None for row in rows)
    costs = [float(row["cost_eur"]) for row in rows if row.get("cost_eur") is not None]
    def mean_metric(name: str) -> float | None:
        values = [row[name] for row in rows if row[name] is not None]
        return statistics.mean(values) if values else None

    summary: dict[str, Any] = {
        "questions": len(rows),
        "errors": sum(row["stop_reason"] == "error" for row in rows),
        "answer_token_f1": mean_metric("answer_token_f1"),
        "retrieved_doc_recall": mean_metric("retrieved_doc_recall"),
        "cited_source_recall": mean_metric("cited_source_recall"),
        "avg_latency_ms": statistics.mean(latencies) if latencies else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "total_cost_eur": sum(costs) if costs and costs_complete else None,
        "avg_cost_eur": statistics.mean(costs) if costs and costs_complete else None,
        "tokens_in": sum(int(row.get("tokens_in") or 0) for row in rows),
        "tokens_out": sum(int(row.get("tokens_out") or 0) for row in rows),
    }
    return summary


def display(value: Any, *, percentage: bool = False) -> str:
    if value is None:
        return "N/A"
    if percentage:
        return f"{value:.1%}"
    return f"{value:.2f}"


def write_outputs(out_dir: Path, rows_by_backend: dict[str, list[dict[str, Any]]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = [row for rows in rows_by_backend.values() for row in rows]
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as target:
        for row in flat_rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

    summaries = {backend: summarize(rows) for backend, rows in rows_by_backend.items()}
    metric_names = list(next(iter(summaries.values())).keys())
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["backend", *metric_names])
        writer.writeheader()
        for backend, summary in summaries.items():
            writer.writerow({"backend": backend, **summary})

    columns = (
        ("backend", "Backend"),
        ("questions", "N"),
        ("errors", "Erreurs"),
        ("answer_token_f1", "F1 réponse"),
        ("retrieved_doc_recall", "Rappel docs"),
        ("cited_source_recall", "Rappel citations"),
        ("avg_latency_ms", "Latence moy. (ms)"),
        ("p50_latency_ms", "P50 (ms)"),
        ("p95_latency_ms", "P95 (ms)"),
        ("total_cost_eur", "Coût total (€)"),
        ("tokens_in", "Tokens entrée"),
        ("tokens_out", "Tokens sortie"),
    )
    percentage_columns = {"answer_token_f1", "retrieved_doc_recall", "cited_source_recall"}
    with (out_dir / "comparison.md").open("w", encoding="utf-8") as target:
        target.write("# Comparaison des backends\n\n")
        target.write("| " + " | ".join(label for _, label in columns) + " |\n")
        target.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for backend, summary in summaries.items():
            cells = [backend]
            for key, _ in columns[1:]:
                value = summary[key]
                if key in percentage_columns:
                    cells.append(display(value, percentage=True))
                elif key in {"questions", "errors", "tokens_in", "tokens_out"}:
                    cells.append("N/A" if value is None else str(value))
                else:
                    cells.append(display(value))
            target.write("| " + " | ".join(cells) + " |\n")
        target.write(
            "\nF1 réponse = F1 lexical par tokens entre la réponse du modèle et `answer` du golden. "
            "Rappel docs/citations sont non définis (`N/A` dans le tableau) quand `gold_docs` est vide.\n"
        )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="Fichier JSONL des questions de référence")
    parser.add_argument("--out", type=Path, required=True, help="Dossier de sortie (écrit results.jsonl)")
    parser.add_argument("--backend", choices=("local", "api", "both"), default="local")
    parser.add_argument("--local-mode", choices=("rag", "agent"), default="rag")
    parser.add_argument(
        "--api-adapter",
        help="Callable Python au format package.module:function, appelée avec question et renvoyant un dict AnswerTrace",
    )
    args = parser.parse_args()

    try:
        questions = load_questions(args.questions)
        backends: dict[str, Callable[[str], Any]] = {}
        if args.backend in {"local", "both"}:
            backends[f"local-{args.local_mode}"] = lambda question: run_local(question, args.local_mode)
        if args.backend in {"api", "both"}:
            if not args.api_adapter:
                parser.error("--api-adapter is required when --backend is api or both")
            backends["api"] = load_callable(args.api_adapter)

        rows_by_backend = {}
        for backend, run in backends.items():
            rows = []
            for item in questions:
                question = item["question"]
                try:
                    result = run(question)
                    if hasattr(result, "__dict__"):
                        result = result.__dict__
                    if not isinstance(result, dict):
                        raise TypeError("backend must return a dict or dataclass with AnswerTrace fields")
                    result = {
                        "answer": result.get("answer", ""),
                        "cited_sources": result.get("citations", result.get("cited_sources", [])),
                        "retrieved_doc_ids": result.get("retrieved_ids", result.get("retrieved_doc_ids", [])),
                        "tokens_in": result.get("tokens_in", 0),
                        "tokens_out": result.get("tokens_out", 0),
                        "latency_ms": result.get("latency_ms"),
                        "cost_eur": result.get("cost_eur"),
                        "tool_calls": result.get("tool_calls", []),
                        "stop_reason": result.get("stop_reason", "error"),
                    }
                except Exception as err:
                    result = {
                        "answer": "",
                        "cited_sources": [],
                        "retrieved_doc_ids": [],
                        "tokens_in": 0,
                        "tokens_out": 0,
                        "latency_ms": None,
                        "cost_eur": None,
                        "tool_calls": [],
                        "stop_reason": "error",
                        "error": f"{type(err).__name__}: {err}",
                    }
                rows.append({
                    "backend": backend,
                    "id": item["id"],
                    "question": question,
                    "gold_answer": item["answer"],
                    "gold_docs": item["gold_docs"],
                    **result,
                    **score(item, result),
                })
            rows_by_backend[backend] = rows

        out_dir = write_outputs(args.out, rows_by_backend)
    except (OSError, ValueError, ImportError, AttributeError, TypeError) as err:
        parser.error(str(err))

    print(f"Résultats enregistrés dans : {out_dir / 'results.jsonl'}")
    print(f"Tableau comparatif : {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
