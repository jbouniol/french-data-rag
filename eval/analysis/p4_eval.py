"""P4 evaluation: native Parquet sources, AnswerTrace results and an Ollama judge.

Usage: uv run python eval/analysis/p4_eval.py {validate,judge,report} --help
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


CORPUS = Path("data/processed/article_versions.parquet")
PROMPT = Path(__file__).with_name("judge_prompt.md")
JUDGE_CTX = 16384
FAILURES = {"error", "invalid_output", "max_iterations", "budget_exceeded", "stuck"}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    gold_docs: list[str]


class Trace(BaseModel):
    """AnswerTrace from P3, with the question ID and optional runner metadata."""
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    id: str = Field(min_length=1)
    answer: str
    citations: list[str]
    retrieved_ids: list[str]
    tokens_in: int = Field(ge=0, strict=True)
    tokens_out: int = Field(ge=0, strict=True)
    latency_ms: float = Field(ge=0, strict=True)
    tool_calls: list[dict]
    stop_reason: Literal["answered", "not_found", "out_of_scope", "invalid_output", "error",
                         "max_iterations", "budget_exceeded", "stuck"]
    cost_eur: float | None = Field(default=None, ge=0, strict=True)
    mode: Literal["rag", "agent"] | None = None


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    correctness: StrictInt = Field(ge=0, le=2)
    groundedness: StrictInt | None = Field(ge=0, le=2)
    citation_support: StrictInt | None = Field(ge=0, le=2)
    abstention_ok: StrictBool | None
    injection_resisted: StrictBool | None


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    scores: Scores
    reason: str = Field(min_length=1, max_length=4000)


class CalculationReference(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    value: Decimal
    unit: Literal["EUR", "percent"]
    tolerance: Decimal = Field(default=Decimal("0.01"), ge=0)


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: Literal["factual", "synthesis", "calculation", "historical", "out_of_scope", "injection"]
    as_of: date | None = None
    expected_behavior: str = ""
    expected_calculation: CalculationReference | None = None
    notes: str = ""
    evidence: dict[str, str] = Field(default_factory=dict)


class ContextLimitError(ValueError):
    """The complete evidence cannot fit the judge's configured context."""


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_jsonl(path):
    rows = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line, object_pairs_hook=_object)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError("expected an object with a nonempty id")
            if row["id"] in rows:
                raise ValueError(f"duplicate id: {row['id']}")
            rows[row["id"]] = row
        except ValueError as err:
            raise ValueError(f"{path}:{number}: {err}") from err
    if not rows:
        raise ValueError(f"{path}: empty input")
    return rows


def write_jsonl(path, rows):
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _ids(values):
    if any(not value.strip() for value in values) or len(set(values)) != len(values):
        raise ValueError("document IDs must be nonempty and unique")


def load_questions(path):
    questions = {}
    for key, row in read_jsonl(path).items():
        q = Question.model_validate(row).model_dump()
        if q["id"] != key or re.search(r"\[[A-Z]+_[A-Z0-9_]+\]", q["question"] + q["answer"]):
            raise ValueError(f"{key}: complete the draft placeholders before evaluation")
        _ids(q["gold_docs"])
        questions[key] = q
    return questions


def load_results(path, questions, mode=None):
    results = {key: Trace.model_validate(row).model_dump() for key, row in read_jsonl(path).items()}
    if results.keys() != questions.keys():
        raise ValueError("result IDs must exactly match the golden set")
    for key, row in results.items():
        if mode and row["mode"] and row["mode"] != mode:
            raise ValueError(f"{key}: --mode conflicts with the recorded mode")
        row["mode"] = row["mode"] or mode
        _ids(row["citations"])
        _ids(row["retrieved_ids"])
    return results


def _day(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    return value if isinstance(value, date) else date.fromisoformat(value)


def load_corpus(path):
    """Keep the actual ingestion column names, including the year-2999 sentinel."""
    frame = pd.read_parquet(path)
    required = {"doc_id", "num", "text", "etat", "date_debut", "date_fin"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError(f"corpus requires nonempty native Parquet columns: {sorted(required)}")
    corpus = {}
    for row in frame.to_dict("records"):
        key = row["doc_id"]
        if not isinstance(key, str) or not key.strip() or key in corpus:
            raise ValueError("corpus has empty or duplicate doc_id")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"{key}: missing source text")
        row["date_debut"], row["date_fin"] = _day(row["date_debut"]), _day(row["date_fin"])
        if row["date_debut"] and row["date_fin"] and row["date_debut"] > row["date_fin"]:
            raise ValueError(f"{key}: reversed validity interval")
        corpus[key] = row
    return corpus


def load_annotations(path, questions):
    if path is None:
        return {}
    rows = {key: Annotation.model_validate(row).model_dump(mode="json", exclude_none=True)
            for key, row in read_jsonl(path).items()}
    if rows.keys() - questions.keys():
        raise ValueError("annotations contain unknown question IDs")
    return rows


def _references(questions, results, corpus):
    for key, question in questions.items():
        required = question["gold_docs"] + results.get(key, {}).get("retrieved_ids", [])
        if set(required) - corpus.keys():
            raise ValueError(f"{key}: unknown golden or retrieved document IDs")
    # Unknown model citations are failures to measure, not reference-data errors.


def checks(question, result, corpus, annotation=None, k=5):
    if k < 1:
        raise ValueError("k must be positive")
    annotation = annotation or {}
    gold, retrieved, cited = set(question["gold_docs"]), result["retrieved_ids"], result["citations"]
    ranked = result.get("mode") == "rag"
    coverage = len(gold.intersection(retrieved)) / len(gold) if gold else None
    recall = len(gold.intersection(retrieved[:k])) / len(gold) if gold and ranked else None
    mrr = next((1 / n for n, key in enumerate(retrieved, 1) if key in gold), 0.0) if gold and ranked else None
    as_of = _day(annotation.get("as_of"))
    validity = []
    if as_of and cited:
        for key in cited:
            doc = corpus.get(key, {})
            start, end = doc.get("date_debut"), doc.get("date_fin")
            validity.append(start <= as_of < end and doc.get("etat") != "ABROGE"
                            if start and end else None)
    return {
        "id": question["id"], "mode": result.get("mode"), "category": annotation.get("category", "unknown"),
        "source_coverage": coverage, "recall_at_k": recall, "mrr": mrr,
        "citation_resolvable": all(key in corpus for key in cited) if cited else None,
        "date_valid": False if False in validity else (True if validity and None not in validity else None),
        "operational_failure": result["stop_reason"] in FAILURES, "stop_reason": result["stop_reason"],
        "tool_errors": sum(call.get("error") is True for call in result["tool_calls"]),
    }


def judge_payload(question, result, corpus, annotation):
    ids = list(dict.fromkeys(question["gold_docs"] + result["citations"]))
    evidence = []
    for key in ids:
        doc = corpus.get(key)
        evidence.append({"doc_id": key, "resolvable": doc is not None,
                         **({field: doc.get(field) for field in
                             ("num", "text", "hierarchy", "nota", "etat", "date_debut", "date_fin")} if doc else {})})
    return {"question": question["question"], "reference_answer": question["answer"],
            "gold_docs": question["gold_docs"], "candidate": {key: result[key] for key in ("answer", "citations", "stop_reason")},
            "annotation": annotation, "evidence": evidence}


def judge_one(payload, model, timeout):
    """Use the same SDK and decoding options as src.agent.llm; judge usage is separate."""
    import ollama
    prompt = PROMPT.read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, default=str)
    # A conservative byte bound leaves space for the chat template and output.
    # Refuse long evidence rather than letting Ollama discard source text.
    if len((prompt + data).encode("utf-8")) + 3072 > JUDGE_CTX:
        raise ContextLimitError("complete evidence exceeds the judge context")
    response = ollama.Client(timeout=timeout).chat(
        model=model, messages=[{"role": "system", "content": prompt}, {"role": "user", "content": data}],
        format=Verdict.model_json_schema(), think=False, keep_alive="30m",
        options={"temperature": 0, "seed": 0, "num_ctx": JUDGE_CTX, "num_predict": 1024},
    )
    if not response.done or response.done_reason == "length":
        raise ValueError("judge response did not complete")
    verdict = Verdict.model_validate(json.loads(response.message.content, object_pairs_hook=_object))
    return {**verdict.model_dump(), "judge": {
        "model": response.model, "model_requested": model, "temperature": 0, "seed": 0,
        "num_ctx": JUDGE_CTX,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tokens_in": response.prompt_eval_count, "tokens_out": response.eval_count,
    }}


def _mean(values):
    return f"{statistics.mean(values):.3f} (n={len(values)})" if values else "not evaluated (n=0)"


def make_report(questions, results, corpus, annotations, judgments, k=5):
    _references(questions, results, corpus)
    if results.keys() != questions.keys() or judgments.keys() - questions.keys():
        raise ValueError("question/result/judgment IDs do not match")
    rows = []
    for key, question in questions.items():
        row = checks(question, results[key], corpus, annotations.get(key), k)
        judgment = judgments.get(key, {})
        row["judge_status"] = judgment.get("status", "missing")
        if row["judge_status"] == "error":
            row["judge_error"] = judgment.get("error_type") or judgment.get("reason") or "unknown error"
        if row["judge_status"] == "ok":
            verdict = Verdict.model_validate({name: judgment.get(name) for name in ("scores", "reason")})
            row.update(verdict.model_dump())
        elif row["judge_status"] not in {"missing", "error"}:
            raise ValueError(f"{key}: unknown judgment status")
        rows.append(row)
    n = len(rows)
    lines = ["# P4 evaluation report", "", f"Questions: {n}. Judge status: {dict(Counter(r['judge_status'] for r in rows))}.",
             "", "| Metric | Result |", "| --- | --- |"]
    for name in ("source_coverage", "recall_at_k", "mrr"):
        lines.append(f"| {name}{f' (k={k})' if name == 'recall_at_k' else ''} | {_mean([r[name] for r in rows if r[name] is not None])} |")
    for name in ("correctness", "groundedness", "citation_support"):
        lines.append(f"| judge {name} (0–2) | {_mean([r['scores'][name] for r in rows if r.get('scores', {}).get(name) is not None])} |")
    lines += ["", "Recall@k and MRR apply only to known RAG runs. Agent retrieved_ids is an ordered union of tool sources, not a ranking.",
              "Source coverage uses all returned IDs; it does not prove the source was used correctly. Questions without gold_docs are excluded.",
              "", "| Check | True | False | Unknown / N/A |", "| --- | --- | --- | --- |"]
    for name in ("citation_resolvable", "date_valid", "operational_failure", "abstention_ok", "injection_resisted"):
        values = [r.get("scores", {}).get(name) if name in {"abstention_ok", "injection_resisted"} else r[name] for r in rows]
        yes, no = values.count(True), values.count(False)
        lines.append(f"| {name} | {yes} | {no} | {n - yes - no} |")
    lines += ["", "| Tracking | Total | Observed / questions |", "| --- | --- | --- |"]
    for name in ("tokens_in", "tokens_out", "cost_eur"):
        values = [r[name] for r in results.values() if r.get(name) is not None]
        lines.append(f"| {name} | {sum(values) if values else 'unknown'} | {len(values)} / {n} |")
    latencies = sorted(r["latency_ms"] for r in results.values())
    lines += [f"| latency_ms | p50={statistics.median(latencies):.1f}; p95={latencies[math.ceil(.95 * n) - 1]:.1f} | {n} / {n} |",
              "", "Cost is the declared inference fee; local 0 EUR excludes hardware and energy. Judge tokens are separate.",
              f"Stop reasons: {dict(Counter(r['stop_reason'] for r in rows))}. Tool errors: {sum(r['tool_errors'] for r in rows)}.",
              "", "| Category | Questions | Judge correctness (0–2) |", "| --- | --- | --- |"]
    for category in sorted({r["category"] for r in rows}):
        members = [r for r in rows if r["category"] == category]
        values = [r["scores"]["correctness"] for r in members if "scores" in r]
        lines.append(f"| {category} | {len(members)} | {_mean(values)} |")
    lines += ["", "## Cases to review", "", "| Question | Observations |", "| --- | --- |"]
    for row in rows:
        observations = [name for name in ("citation_resolvable", "date_valid") if row[name] is False]
        if row["source_coverage"] is not None and row["source_coverage"] < 1:
            observations.append("incomplete source coverage")
        if row["operational_failure"]:
            observations.append(row["stop_reason"])
        if row.get("scores", {}).get("correctness", 2) < 2:
            observations.append(row["reason"].replace("|", "\\|").replace("\n", " "))
        if row["judge_status"] != "ok":
            observations.append("judge " + row["judge_status"] + (": " + row["judge_error"] if "judge_error" in row else ""))
        if observations:
            lines.append(f"| {row['id']} | {'; '.join(observations)} |")
    lines += ["", "## Residual risks and manual analysis", "",
              "Review at least ten judgments and every disputed case; add three explained failures and proposed fixes.",
              "Investigate corpus coverage, missing historical versions, external SMIC references and judge bias.",
              "Agent tool observations are clipped at 300 characters: no full calculation or final-context check is inferred.",
              "One article number can overwrite another version in the agent citation map; inspect multi-date cases.",
              "These are limits to investigate, not incidents proven by this report. Small samples do not establish legal reliability.", ""]
    return rows, "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "judge", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--questions", type=Path, required=True)
        sub.add_argument("--corpus", type=Path, default=CORPUS)
        sub.add_argument("--results", type=Path, required=name != "validate")
        sub.add_argument("--annotations", type=Path)
        sub.add_argument("--mode", choices=("rag", "agent"))
        if name == "validate":
            sub.add_argument("--expected-count", type=int)
        else:
            sub.add_argument("--out", type=Path, required=True)
        if name == "judge":
            sub.add_argument("--model")
            sub.add_argument("--timeout", type=float, default=180)
            sub.add_argument("--max-chars", type=int, default=120000)
        if name == "report":
            sub.add_argument("--judgments", type=Path)
            sub.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        questions = load_questions(args.questions)
        corpus = load_corpus(args.corpus)
        results = load_results(args.results, questions, args.mode) if args.results else {}
        annotations = load_annotations(args.annotations, questions)
        _references(questions, results, corpus)
        if args.command == "validate":
            if args.expected_count is not None and (args.expected_count < 1 or len(questions) != args.expected_count):
                raise ValueError("golden size does not match positive --expected-count")
            unknown = {key: [doc for doc in row["citations"] if doc not in corpus] for key, row in results.items()}
            print(json.dumps({"questions": len(questions), "documents": len(corpus), "results": len(results),
                              "unresolved_citations": {key: ids for key, ids in unknown.items() if ids}}))
        elif args.command == "judge":
            if args.out.exists() or args.timeout <= 0 or args.max_chars < 1:
                raise ValueError("choose a new output file and positive timeout/context limit")
            model = args.model
            if not model:
                sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
                from src.agent.llm import MODEL
                model = MODEL
            payloads = {key: judge_payload(q, results[key], corpus, annotations.get(key, {})) for key, q in questions.items()}
            if any(len(json.dumps(p, default=str)) > args.max_chars for p in payloads.values()):
                raise ValueError("evidence exceeds --max-chars; no sources have been silently truncated")
            output = []
            for key, payload in payloads.items():
                try:
                    row = {"id": key, "status": "ok", **judge_one(payload, model, args.timeout)}
                except Exception as err:
                    row = {"id": key, "status": "error", "error_type": type(err).__name__}
                    output.append(row)
                    # A malformed model response may be case-specific; service failures abort.
                    if not isinstance(err, ValueError):
                        break
                    continue
                output.append(row)
            write_jsonl(args.out, output)
            failed = sum(row["status"] == "error" for row in output)
            print(json.dumps({"judged": len(output) - failed, "errors": failed, "unrun": len(questions) - len(output)}))
            return int(failed > 0)
        else:
            if args.k < 1:
                raise ValueError("k must be positive")
            judgments = read_jsonl(args.judgments) if args.judgments else {}
            rows, report = make_report(questions, results, corpus, annotations, judgments, args.k)
            if (args.out / "report.md").exists() or (args.out / "checks.jsonl").exists():
                raise ValueError("choose a new report directory")
            write_jsonl(args.out / "checks.jsonl", rows)
            with (args.out / "report.md").open("x", encoding="utf-8") as handle:
                handle.write(report)
            print(args.out / "report.md")
    except (ValueError, OSError, ImportError) as err:
        print(f"Evaluation error: {err}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
