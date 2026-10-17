"""P4 contracts on fictitious articles and traces; no Ollama or legal evaluation."""
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
import sys

import httpx
import ollama
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import p4_eval as E
from src.agent.answer import AnswerTrace


def question(id="q1", gold_docs=None, **fields):
    return {"id": id, "question": "Question fictive ?", "answer": "Référence vérifiée.",
            "gold_docs": ["v1"] if gold_docs is None else gold_docs, **fields}


def result(id="q1", **fields):
    trace = AnswerTrace(answer="Réponse fictive [L1].", citations=["v1"],
                        retrieved_ids=["v2", "v1"], tokens_in=10, tokens_out=5,
                        latency_ms=12.5, stop_reason="answered")
    return {"id": id, **asdict(trace), **fields}


def judgment(id="q1", correctness=2, **fields):
    return {"id": id, "status": "ok", "scores": {
        "correctness": correctness, "groundedness": 2, "citation_support": 2,
        "abstention_ok": None, "injection_resisted": None},
        "reason": "Justification fictive.", **fields}


@pytest.fixture
def files(tmp_path):
    """The corpus matches src.ingest.build.ARTICLE_COLUMNS, including date(2999)."""
    rows = []
    for id, start, end, state in (
        ("v1", date(2021, 1, 1), date(2999, 1, 1), "VIGUEUR"),
        ("v2", date(2010, 1, 1), date(2021, 1, 1), "MODIFIE"),
    ):
        rows.append({"doc_id": id, "num": "L1", "text": f"Texte fictif {id}.",
                     "chunk_text": f"Code du travail - Article L1\nTexte fictif {id}.",
                     "hierarchy": "Partie législative > Livre I", "etat": state,
                     "date_debut": start, "date_fin": end,
                     "url": f"https://www.legifrance.gouv.fr/codes/article_lc/{id}",
                     "nota": "", "is_in_corpus": id == "v1"})
    corpus_path = tmp_path / "article_versions.parquet"
    pd.DataFrame(rows).to_parquet(corpus_path, index=False)
    questions_path, results_path = tmp_path / "golden.jsonl", tmp_path / "results.jsonl"
    E.write_jsonl(questions_path, [question()])
    E.write_jsonl(results_path, [result(mode="rag")])
    return questions_path, results_path, corpus_path


def test_native_parquet_and_answer_trace_need_no_conversion(files):
    questions_path, results_path, corpus_path = files
    questions = E.load_questions(questions_path)
    results = E.load_results(results_path, questions)
    corpus = E.load_corpus(corpus_path)
    assert results["q1"]["retrieved_ids"] == ["v2", "v1"]
    assert results["q1"]["stop_reason"] == "answered"
    assert results["q1"]["cost_eur"] == 0
    assert corpus["v1"]["date_fin"] == date(2999, 1, 1)
    assert corpus["v2"]["date_debut"] == date(2010, 1, 1)


def test_four_field_golden_needs_no_annotation(tmp_path):
    path = tmp_path / "golden.jsonl"
    original = question()
    E.write_jsonl(path, [original])
    assert E.load_questions(path) == {"q1": original}


@pytest.mark.parametrize("annotation", [
    {"category": "unknown"}, {"category": "historical", "as_of": "not-a-date"},
    {"category": "historical", "as_of": "2025-02-30"},
    {"category": "factual", "unexpected": True}, {"as_of": "2025-01-01"},
])
def test_invalid_nested_annotations_are_rejected(tmp_path, annotation):
    path = tmp_path / "bad-metadata.jsonl"
    E.write_jsonl(path, [question(annotation=annotation)])
    with pytest.raises(ValueError):
        E.load_questions(path)


@pytest.mark.parametrize("changes", [
    {"status": "draft"}, {"status": "draft_needs_sources"}, {"answer": None},
    {"question": "Date [DATE_DE_REFERENCE] ?"},
    {"answer": "[SMIC_MENSUEL_BRUT_EUR]"}, {"answer": " "},
    {"gold_docs": ["v1", "v1"]}, {"gold_docs": "v1"},
])
def test_drafts_and_incomplete_references_are_not_golden(tmp_path, changes):
    path = tmp_path / "bad.jsonl"
    E.write_jsonl(path, [question(**changes)])
    with pytest.raises(ValueError):
        E.load_questions(path)


@pytest.mark.parametrize("rows", [[], [question(), question()]])
def test_golden_must_be_nonempty_and_unique(tmp_path, rows):
    path = tmp_path / "bad.jsonl"
    E.write_jsonl(path, rows)
    with pytest.raises(ValueError):
        E.load_questions(path)


def test_out_of_corpus_question_can_expect_abstention(tmp_path):
    path = tmp_path / "outside.jsonl"
    E.write_jsonl(path, [question(gold_docs=[], answer="Reconnaître l'absence de source.")])
    assert E.load_questions(path)["q1"]["gold_docs"] == []


@pytest.mark.parametrize("changes", [
    {"tokens_in": True}, {"tokens_out": -1}, {"latency_ms": -1},
    {"cost_eur": -1}, {"retrieved_ids": ["v1", "v1"]}, {"tool_calls": "[]"},
])
def test_malformed_tracking_is_rejected(tmp_path, changes):
    path = tmp_path / "bad.jsonl"
    E.write_jsonl(path, [result(**changes)])
    with pytest.raises(ValueError):
        E.load_results(path, {"q1": question()})


def test_results_match_golden_and_mode_is_explicit(tmp_path):
    path = tmp_path / "results.jsonl"
    E.write_jsonl(path, [result()])
    assert E.load_results(path, {"q1": question()}, mode="agent")["q1"]["mode"] == "agent"
    with pytest.raises(ValueError):
        E.load_results(path, {"other": question(id="other")})


@pytest.mark.parametrize("mode, expected_recall, expected_mrr", [
    ("rag", 1, 0.5), ("agent", None, None), (None, None, None),
])
def test_rank_metrics_only_for_rag_and_coverage_for_every_mode(files, mode, expected_recall, expected_mrr):
    corpus = E.load_corpus(files[2])
    row = E.checks(question(), result(mode=mode), corpus, {}, k=5)
    assert row["source_coverage"] == 1
    assert row["recall_at_k"] == expected_recall
    assert row["mrr"] == expected_mrr


def test_unanswerable_questions_are_excluded_from_retrieval_metrics(files):
    row = E.checks(question(gold_docs=[]), result(mode="rag"), E.load_corpus(files[2]), {})
    assert row["source_coverage"] is None
    assert row["recall_at_k"] is None
    assert row["mrr"] is None


def test_recall_cutoff_does_not_change_all_source_coverage(files):
    row = E.checks(question(), result(mode="rag"), E.load_corpus(files[2]), {}, k=1)
    assert row["source_coverage"] == 1
    assert row["recall_at_k"] == 0
    assert row["mrr"] == 0.5


def test_invented_citation_is_a_measured_failure(files):
    row = E.checks(question(), result(citations=["invented"]), E.load_corpus(files[2]), {})
    assert row["citation_resolvable"] is False


def test_unknown_gold_reference_is_rejected(files):
    with pytest.raises(ValueError):
        E.make_report({"q1": question(gold_docs=["invented"])}, {"q1": result()},
                      E.load_corpus(files[2]), {}, {})


@pytest.mark.parametrize("doc, as_of, expected", [
    ("v1", "2021-01-01", True), ("v1", "2020-12-31", False),
    ("v2", "2020-12-31", True), ("v2", "2021-01-01", False),
])
def test_validity_interval_includes_start_and_excludes_end(files, doc, as_of, expected):
    row = E.checks(question(), result(citations=[doc]), E.load_corpus(files[2]), {"as_of": as_of})
    assert row["date_valid"] is expected


def test_abrogated_source_is_not_current_and_missing_dates_are_unknown(files):
    corpus = E.load_corpus(files[2])
    assert E.checks(question(), result(), corpus, {})["date_valid"] is None
    corpus["v1"] = {**corpus["v1"], "etat": "ABROGE"}
    assert E.checks(question(), result(), corpus, {"as_of": "2025-01-01"})["date_valid"] is False
    corpus["v1"] = {**corpus["v1"], "etat": "VIGUEUR", "date_debut": None}
    assert E.checks(question(), result(), corpus, {"as_of": "2025-01-01"})["date_valid"] is None


@pytest.mark.parametrize("stop_reason, failed", [
    ("answered", False), ("not_found", False), ("out_of_scope", False),
    ("stuck", True), ("budget_exceeded", True), ("max_iterations", True), ("error", True),
])
def test_operational_failure_is_not_successful_abstention(files, stop_reason, failed):
    row = E.checks(question(gold_docs=[]), result(answer="", citations=[], stop_reason=stop_reason),
                   E.load_corpus(files[2]), {})
    assert row["operational_failure"] is failed


def test_calculation_reference_is_not_a_parser_for_the_model_answer(files, tmp_path):
    path = tmp_path / "annotated-golden.jsonl"
    E.write_jsonl(path, [question(annotation={"category": "calculation", "as_of": "2025-01-01",
                    "expected_calculation": {"value": "954.95", "unit": "EUR", "tolerance": "0.01"},
                    "notes": "SMIC et taux vérifiés indépendamment.", "evidence": {"v1": "Texte fictif v1."}})])
    q = E.load_questions(path)["q1"]
    annotation = q["annotation"]
    row = E.checks(q, result(answer="Montant 1 000,00 €, après comparaison avec 954,95 €."),
                   E.load_corpus(files[2]), annotation)
    # A number appearing anywhere in a free answer does not establish a correct calculation.
    assert row.get("calculation_correct") is None
    assert annotation["expected_calculation"]["value"] == "954.95"
    assert annotation["expected_calculation"]["tolerance"] == "0.01"
    assert annotation["as_of"] == "2025-01-01"
    assert annotation["evidence"]["v1"] == "Texte fictif v1."
    payload = E.judge_payload(q, result(), E.load_corpus(files[2]), annotation)
    assert payload["annotation"]["expected_calculation"]["value"] == "954.95"
    assert payload["evidence"][0]["text"] == "Texte fictif v1."


@pytest.mark.parametrize("invalid", [True, 3, None, "2"])
def test_invalid_judge_scores_are_not_averaged(files, invalid):
    with pytest.raises(ValueError):
        E.make_report({"q1": question()}, {"q1": result(mode="rag")},
                      E.load_corpus(files[2]), {}, {"q1": judgment(correctness=invalid)})


def test_judge_errors_are_missing_scores_and_zero_cost_is_observed(files):
    questions = {"q1": question(), "q2": question(id="q2")}
    results = {"q1": result(mode="rag", cost_eur=0),
               "q2": result(id="q2", mode="rag", cost_eur=None)}
    checks, report = E.make_report(questions, results, E.load_corpus(files[2]), {}, {
        "q1": judgment(), "q2": {"id": "q2", "status": "error", "reason": "timeout"}})
    assert len(checks) == 2
    assert "cost_eur" in report and "tokens_in" in report and "latency_ms" in report
    assert "2.000" in report  # Only the valid judgment, rather than a fabricated zero.
    assert "1 / 2" in report or "1/2" in report
    assert checks[1]["judge_status"] == "error"
    assert "timeout" in report


def test_jsonl_writer_preserves_existing_results(tmp_path):
    path = tmp_path / "saved.jsonl"
    E.write_jsonl(path, [{"id": "original"}])
    original = path.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        E.write_jsonl(path, [{"id": "replacement"}])
    assert path.read_bytes() == original


def test_cli_workflow_uses_native_traces_and_refuses_overwrite(files, tmp_path, monkeypatch):
    _, results_path, corpus_path = files
    questions_path = tmp_path / "annotated-golden.jsonl"
    E.write_jsonl(questions_path, [question(annotation={"category": "calculation", "as_of": "2025-01-01",
                    "expected_calculation": {"value": "954.95", "unit": "EUR"}})])
    common = ["--questions", str(questions_path), "--corpus", str(corpus_path),
              "--results", str(results_path)]
    assert E.main(["validate", *common, "--expected-count", "1"]) == 0

    calls = []

    def fake_judge(payload, model, timeout):
        calls.append((payload, model, timeout))
        row = judgment()
        row.pop("id")
        row.pop("status")
        return row

    monkeypatch.setattr(E, "judge_one", fake_judge)
    judgments_path = tmp_path / "judgments.jsonl"
    judge_args = ["judge", *common, "--model", "synthetic-judge", "--out", str(judgments_path)]
    assert E.main(judge_args) == 0
    assert calls[0][0]["annotation"]["category"] == "calculation"
    assert calls[0][0]["annotation"]["as_of"] == "2025-01-01"
    assert calls[0][0]["annotation"]["expected_calculation"]["value"] == "954.95"
    assert calls[0][1] == "synthetic-judge"
    assert calls[0][2] == 180
    assert json.loads(judgments_path.read_text().strip())["status"] == "ok"
    original = judgments_path.read_bytes()
    assert E.main(judge_args) != 0
    assert judgments_path.read_bytes() == original
    assert len(calls) == 1  # Existing output is checked before paying for another judge call.

    out = tmp_path / "report"
    report_args = ["report", *common, "--judgments", str(judgments_path), "--out", str(out)]
    assert E.main(report_args) == 0
    assert (out / "report.md").is_file()
    checks = E.read_jsonl(out / "checks.jsonl")
    assert len(checks) == 1
    assert checks["q1"]["category"] == "calculation"
    assert checks["q1"]["date_valid"] is True
    assert "| calculation | 1 |" in (out / "report.md").read_text()
    original = (out / "report.md").read_bytes()
    assert E.main(report_args) != 0
    assert (out / "report.md").read_bytes() == original


@pytest.mark.parametrize("error, expected_calls", [(ValueError("malformed verdict"), 2),
                                                   (ConnectionError("service unavailable"), 1)])
def test_judge_cli_records_errors_without_fabricating_scores(files, tmp_path, monkeypatch, error, expected_calls):
    questions_path, results_path = tmp_path / "two-questions.jsonl", tmp_path / "two-results.jsonl"
    E.write_jsonl(questions_path, [question(), question(id="q2")])
    E.write_jsonl(results_path, [result(mode="rag"), result(id="q2", mode="rag")])
    common = ["--questions", str(questions_path), "--results", str(results_path), "--corpus", str(files[2])]
    calls = []

    def failing_judge(payload, model, timeout):
        calls.append(payload)
        raise error

    monkeypatch.setattr(E, "judge_one", failing_judge)
    judgments_path = tmp_path / "failed-judgments.jsonl"
    assert E.main(["judge", *common, "--model", "synthetic", "--out", str(judgments_path)]) != 0
    rows = [json.loads(line) for line in judgments_path.read_text().splitlines()]
    assert len(calls) == expected_calls
    assert len(rows) == expected_calls
    assert all(row["status"] == "error" and "scores" not in row for row in rows)
    assert all(row["error_type"] == type(error).__name__ for row in rows)

    out = tmp_path / "failed-report"
    assert E.main(["report", *common, "--judgments", str(judgments_path), "--out", str(out)]) == 0
    checks = [json.loads(line) for line in (out / "checks.jsonl").read_text().splitlines()]
    assert all("scores" not in row for row in checks)
    assert checks[0]["judge_status"] == "error"
    assert checks[1]["judge_status"] == ("error" if expected_calls == 2 else "missing")
    assert "not evaluated (n=0)" in (out / "report.md").read_text()


@pytest.mark.parametrize("reference", [
    {"value": "NaN", "unit": "EUR"}, {"value": "Infinity", "unit": "EUR"},
    {"value": "10.00", "unit": "EUR", "tolerance": "-0.01"},
    {"value": "10.00", "unit": "EUR", "tolerance": "NaN"},
])
def test_calculation_references_must_be_finite_with_nonnegative_tolerance(tmp_path, reference):
    path = tmp_path / "bad-reference.jsonl"
    E.write_jsonl(path, [question(annotation={"category": "calculation", "expected_calculation": reference})])
    with pytest.raises(ValueError):
        E.load_questions(path)


def test_judge_uses_real_ollama_sdk_with_native_schema_and_dates(files, monkeypatch):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        verdict = judgment()
        verdict = {key: verdict[key] for key in ("scores", "reason")}
        return httpx.Response(200, json={
            "model": "synthetic:resolved", "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": json.dumps(verdict)},
            "done": True, "done_reason": "stop", "prompt_eval_count": 23, "eval_count": 8})

    client = ollama.Client
    monkeypatch.setattr(ollama, "Client", lambda **kwargs: client(
        host="http://p4.test", transport=httpx.MockTransport(handler), **kwargs))
    payload = E.judge_payload(question(), result(), E.load_corpus(files[2]), {})
    row = E.judge_one(payload, "synthetic:requested", timeout=2)
    assert row["scores"]["correctness"] == 2
    assert row["reason"] == "Justification fictive."
    assert row["judge"]["model"] == "synthetic:resolved"
    assert row["judge"]["model_requested"] == "synthetic:requested"
    assert row["judge"]["tokens_in"] == 23 and row["judge"]["tokens_out"] == 8
    assert len(row["judge"]["prompt_sha256"]) == 64
    assert len(row["judge"]["input_sha256"]) == 64
    assert requests[0]["options"]["num_ctx"] == E.JUDGE_CTX
    assert requests[0]["options"]["temperature"] == 0
    assert requests[0]["format"]["$defs"]["Scores"]["properties"]["correctness"]["type"] == "integer"
    sent = json.loads(requests[0]["messages"][1]["content"])
    assert sent["evidence"][0]["date_fin"] == "2999-01-01"
    assert sent["evidence"][0]["text"] == "Texte fictif v1."


def test_judge_refuses_long_evidence_before_calling_ollama(monkeypatch):
    def unexpected_client(**kwargs):
        pytest.fail("No SDK client or request is needed when complete sources cannot fit.")

    monkeypatch.setattr(ollama, "Client", unexpected_client)
    with pytest.raises(E.ContextLimitError):
        E.judge_one({"evidence": "x" * E.JUDGE_CTX}, "synthetic", timeout=2)
