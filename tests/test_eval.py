import json

import pytest

from eval.run_eval import load_questions, score, summarize, token_f1, write_outputs


def test_load_questions_reports_invalid_json_line(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question":"Question valide"}\n{invalid}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"questions\.jsonl:2: invalid JSON"):
        load_questions(path)


def test_load_questions_adds_ids_and_rejects_empty_files(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"question":"Question ?"}\n', encoding="utf-8")
    assert load_questions(path)[0]["id"] == "q001"

    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No questions found"):
        load_questions(path)


def test_score_calculates_status_keyword_citation_and_answer_quality():
    item = {
        "expected_status": "answered",
        "must_include": ["trois ans", "contrat"],
        "expected_citations": ["LEGIARTI1"],
        "reference_answer": "Le contrat dure trois ans.",
    }
    result = {
        "answer": "Le contrat dure trois ans.",
        "citations": ["LEGIARTI1"],
        "stop_reason": "answered",
    }

    scored = score(item, result)
    assert scored["status_correct"] is True
    assert scored["keyword_recall"] == 1
    assert scored["citation_recall"] == 1
    assert scored["answer_token_f1"] == 1
    assert scored["quality_score"] == 1


def test_score_leaves_unannotated_quality_metrics_unavailable():
    scored = score({"question": "Question ?"}, {"answer": "", "stop_reason": "answered"})
    assert scored["status_correct"] is True
    assert scored["quality_score"] is None
    assert scored["keyword_recall"] is None
    assert scored["citation_recall"] is None
    assert scored["answer_token_f1"] is None


def test_token_f1_uses_token_overlap():
    assert token_f1("chat noir", "chat blanc") == pytest.approx(0.5)


def test_summarize_and_write_comparison(tmp_path):
    row = {
        "backend": "local-rag",
        "id": "q001",
        "question": "Question ?",
        "answer": "Réponse",
        "citations": [],
        "retrieved_ids": [],
        "tokens_in": 10,
        "tokens_out": 4,
        "latency_ms": 100.0,
        "cost_eur": 0.0,
        "stop_reason": "answered",
        "status_correct": True,
        "quality_score": 1.0,
        "keyword_recall": None,
        "citation_recall": None,
        "answer_token_f1": None,
    }

    summary = summarize([row])
    assert summary["total_cost_eur"] == 0
    assert summary["p95_latency_ms"] == 100
    run_dir = write_outputs(tmp_path, "test-run", {"local-rag": [row]})
    assert (run_dir / "comparison.md").exists()
    assert "N/A" in (run_dir / "comparison.md").read_text(encoding="utf-8")
    assert json.loads((run_dir / "predictions.jsonl").read_text(encoding="utf-8"))["id"] == "q001"


def test_summary_does_not_underreport_partial_costs():
    rows = [
        {"latency_ms": 1, "cost_eur": 0.01, "stop_reason": "answered", "status_correct": True,
         "quality_score": None, "keyword_recall": None, "citation_recall": None, "answer_token_f1": None},
        {"latency_ms": 2, "cost_eur": None, "stop_reason": "answered", "status_correct": True,
         "quality_score": None, "keyword_recall": None, "citation_recall": None, "answer_token_f1": None},
    ]
    summary = summarize(rows)
    assert summary["total_cost_eur"] is None
    assert summary["avg_cost_eur"] is None
    assert summary["quality_score"] is None
