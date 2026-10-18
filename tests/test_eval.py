import json

import pytest

from eval.run_eval import load_questions, score, summarize, token_f1, write_outputs


def test_load_questions_validates_official_schema(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"id":"q001","question":"Question ?","answer":"Réponse","gold_docs":["DOC1"]}\n',
        encoding="utf-8",
    )
    assert load_questions(path) == [
        {"id": "q001", "question": "Question ?", "answer": "Réponse", "gold_docs": ["DOC1"]}
    ]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ('{"question":"Question ?","answer":"Réponse","gold_docs":[]}', "non-empty 'id' string"),
        ('{"id":"q1","answer":"Réponse","gold_docs":[]}', "non-empty 'question' string"),
        ('{"id":"q1","question":"Question ?","gold_docs":[]}', "an 'answer' string"),
        ('{"id":"q1","question":"Question ?","answer":"Réponse","gold_docs":"DOC1"}', "list of document IDs"),
    ],
)
def test_load_questions_rejects_missing_or_malformed_fields(tmp_path, data, message):
    path = tmp_path / "questions.jsonl"
    path.write_text(data + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_questions(path)


def test_load_questions_reports_invalid_json_and_duplicate_ids(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id":"q001","question":"Question ?","answer":"Réponse","gold_docs":[]}\n{invalid}\n',
                    encoding="utf-8")
    with pytest.raises(ValueError, match=r"questions\.jsonl:2: invalid JSON"):
        load_questions(path)

    path.write_text(
        '{"id":"q001","question":"Q1","answer":"R1","gold_docs":[]}\n'
        '{"id":"q001","question":"Q2","answer":"R2","gold_docs":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate question IDs"):
        load_questions(path)


def test_load_questions_rejects_empty_files(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No questions found"):
        load_questions(path)


def test_score_calculates_answer_f1_and_document_recall():
    item = {"answer": "Le contrat dure trois ans.", "gold_docs": ["DOC1", "DOC2"]}
    result = {
        "answer": "Le contrat dure trois ans.",
        "citations": ["DOC1"],
        "retrieved_ids": ["DOC1", "DOC3"],
    }

    scored = score(item, result)
    assert scored["answer_token_f1"] == 1
    assert scored["retrieved_doc_recall"] == 0.5
    assert scored["cited_source_recall"] == 0.5


def test_score_leaves_document_recall_undefined_for_empty_gold_docs():
    scored = score({"answer": "Je ne sais pas.", "gold_docs": []}, {
        "answer": "Je ne sais pas.",
        "citations": [],
        "retrieved_ids": ["DOC1"],
    })
    assert scored["answer_token_f1"] == 1
    assert scored["retrieved_doc_recall"] is None
    assert scored["cited_source_recall"] is None


def test_token_f1_uses_token_overlap():
    assert token_f1("chat noir", "chat blanc") == pytest.approx(0.5)


def test_summarize_and_write_official_results_jsonl(tmp_path):
    import p4_eval

    row = {
        "backend": "local-rag",
        "mode": "rag",
        "id": "q001",
        "question": "Question ?",
        "gold_answer": "Réponse",
        "gold_docs": ["DOC1"],
        "answer": "Réponse",
        "citations": ["DOC1"],
        "retrieved_ids": ["DOC1"],
        "tokens_in": 10,
        "tokens_out": 4,
        "latency_ms": 100.0,
        "cost_eur": 0.0,
        "tool_calls": [{"name": "search_articles"}],
        "stop_reason": "answered",
        "answer_token_f1": 1.0,
        "retrieved_doc_recall": 1.0,
        "cited_source_recall": 1.0,
    }

    summary = summarize([row])
    assert summary["total_cost_eur"] == 0
    assert summary["p95_latency_ms"] == 100
    write_outputs(tmp_path, {"local-rag": [row]})

    saved = json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8"))
    assert saved["answer"] == "Réponse"
    assert saved["citations"] == ["DOC1"]
    assert saved["retrieved_ids"] == ["DOC1"]
    assert saved["tool_calls"] == [{"name": "search_articles"}]
    assert (tmp_path / "comparison.md").exists()

    validated = p4_eval.load_results(tmp_path / "results.jsonl", {"q001": {"id": "q001"}})
    assert validated["q001"]["citations"] == ["DOC1"]
    assert validated["q001"]["retrieved_ids"] == ["DOC1"]
    assert validated["q001"]["mode"] == "rag"


def test_summary_does_not_underreport_partial_costs():
    rows = [
        {"latency_ms": 1, "cost_eur": 0.01, "stop_reason": "answered", "tokens_in": 1,
         "tokens_out": 1, "answer_token_f1": None, "retrieved_doc_recall": None, "cited_source_recall": None},
        {"latency_ms": 2, "cost_eur": None, "stop_reason": "answered", "tokens_in": 1,
         "tokens_out": 1, "answer_token_f1": None, "retrieved_doc_recall": None, "cited_source_recall": None},
    ]
    summary = summarize(rows)
    assert summary["total_cost_eur"] is None
    assert summary["avg_cost_eur"] is None
    assert summary["answer_token_f1"] is None
