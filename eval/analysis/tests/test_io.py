"""Contract tests use fictitious data, never legal answers or real credentials."""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p4.io import index_by_id, load_corpus, load_golden, load_results, read_jsonl, write_jsonl
from validate import main as validate_main, validate_files


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def fixture(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        write_jsonl(path, rows)
        return path

    @staticmethod
    def question(**changes: object) -> dict:
        return {"id": "q1", "question": "Question fictive ?", "answer": "Réponse témoin.", "gold_docs": ["v1"], **changes}

    @staticmethod
    def result(**changes: object) -> dict:
        return {"id": "q1", "answer": "Réponse.", "citations": ["v1"], "retrieved_doc_ids": ["v2", "v1"], **changes}

    def test_jsonl_roundtrip_unicode_and_refuses_overwrite(self) -> None:
        path = self.fixture("unicode.jsonl", [{"id": "q", "question": "Été ?"}])
        self.assertEqual(read_jsonl(path), [{"id": "q", "question": "Été ?"}])
        with self.assertRaises(FileExistsError):
            write_jsonl(path, [{"id": "changed"}])
        self.assertEqual(read_jsonl(path)[0]["id"], "q")

    def test_invalid_json_reports_file_and_line(self) -> None:
        path = self.root / "bad.jsonl"
        for malformed in ('{"id":', '{"id":"a","id":"b"}', '{"score":NaN}', '{"score":1e999}', '[]', ''):
            with self.subTest(malformed=malformed):
                path.write_text('{}\n' + malformed + '\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, r"bad.jsonl: line 2"):
                    read_jsonl(path)

    def test_serialization_failure_leaves_no_file(self) -> None:
        target = self.root / "invalid.jsonl"
        with self.assertRaises(ValueError):
            write_jsonl(target, [{"number": float("nan")}])
        self.assertFalse(target.exists())

    def test_duplicate_record_ids_and_missing_ids_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            index_by_id([{"id": "q"}, {"id": "q"}])
        with self.assertRaisesRegex(ValueError, "missing required"):
            index_by_id([{}])

    def test_golden_allows_abstention_and_preserves_annotations(self) -> None:
        path = self.fixture("gold.jsonl", [self.question(gold_docs=[], category="abstention")])
        self.assertEqual(load_golden(path)["q1"]["category"], "abstention")
        self.assertEqual(load_golden(path)["q1"]["gold_docs"], [])

    def test_empty_golden_and_corpus_are_not_valid_evaluation_inputs(self) -> None:
        for filename, loader in (("empty-gold.jsonl", load_golden), ("empty-corpus.jsonl", load_corpus)):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "at least one"):
                    loader(self.fixture(filename, []))

    def test_golden_rejects_incomplete_or_ambiguous_references(self) -> None:
        bad_rows = [self.question(question=" "), self.question(gold_docs=["v1", "v1"]), self.question(gold_docs="v1")]
        incomplete = self.question()
        del incomplete["answer"]
        bad_rows.append(incomplete)
        for number, row in enumerate(bad_rows):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    load_golden(self.fixture(f"bad-gold-{number}.jsonl", [row]))

    def test_drafts_and_unfilled_legal_assumptions_cannot_be_promoted(self) -> None:
        for status in ("draft", "draft_needs_sources"):
            with self.subTest(status=status):
                with self.assertRaisesRegex(ValueError, "source-verified"):
                    load_golden(self.fixture(f"{status}.jsonl", [self.question(status=status)]))
        for number, token in enumerate(("[DATE_DE_REFERENCE]", "[SMIC_MENSUEL_BRUT_EUR]", "[HYPOTHESES_CONTRAT]")):
            for field in ("question", "answer"):
                with self.subTest(token=token, field=field):
                    with self.assertRaisesRegex(ValueError, "unfilled placeholder"):
                        load_golden(self.fixture(f"placeholder-{number}-{field}.jsonl", [self.question(**{field: "À renseigner " + token})]))

    def test_corpus_dates_and_unique_version_ids(self) -> None:
        old = {"doc_id": "v1", "text": "Texte fictif ancien.", "article_number": "X1", "valid_from": "2020-01-01", "valid_to": "2021-01-01"}
        current = {"doc_id": "v2", "text": "Texte fictif actuel.", "valid_from": "2021-01-01", "valid_to": None}
        self.assertEqual(set(load_corpus(self.fixture("corpus.jsonl", [old, current]))), {"v1", "v2"})
        bad_dates = [("2020-02-30", None), ("2020-1-01", None), ("2021-01-01", "2021-01-01"), ("2022-01-01", "2021-01-01")]
        for number, (start, end) in enumerate(bad_dates):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    load_corpus(self.fixture(f"bad-date-{number}.jsonl", [{**old, "valid_from": start, "valid_to": end}]))
        with self.assertRaisesRegex(ValueError, "duplicate doc_id"):
            load_corpus(self.fixture("duplicate-corpus.jsonl", [old, old]))

    def test_results_require_ranks_and_normalize_only_documented_aliases(self) -> None:
        row = self.result()
        row["cited_sources"] = row.pop("citations")
        row["retrieved_document_ids"] = row.pop("retrieved_doc_ids")
        normalized = load_results(self.fixture("alias.jsonl", [row]))["q1"]
        self.assertEqual(normalized["retrieved_doc_ids"], ["v2", "v1"])
        self.assertEqual(normalized["citations"], ["v1"])
        self.assertIn("cited_sources", normalized)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            load_results(self.fixture("conflict.jsonl", [self.result(cited_sources=["v2"])]))
        incomplete = self.result()
        del incomplete["retrieved_doc_ids"]
        with self.assertRaisesRegex(ValueError, "retrieved_doc_ids"):
            load_results(self.fixture("no-ranks.jsonl", [incomplete]))

    def test_empty_generation_is_kept_and_optional_tracking_is_validated(self) -> None:
        path = self.fixture("failed.jsonl", [self.result(answer="", tokens_in=0, tokens_out=0, latency_ms=0, tool_calls=[], stop_reason="error")])
        self.assertEqual(load_results(path, strict_tracking=True)["q1"]["answer"], "")
        with self.assertRaisesRegex(ValueError, "tokens_in"):
            load_results(self.fixture("no-tracking.jsonl", [self.result()]), strict_tracking=True)
        bad_fields = [{"tokens_in": True}, {"tokens_out": -1}, {"latency_ms": -0.1}, {"tool_calls": "[]"}, {"stop_reason": ""}, {"retrieved_doc_ids": ["v1", "v1"]}]
        for number, changes in enumerate(bad_fields):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    load_results(self.fixture(f"bad-result-{number}.jsonl", [self.result(**changes)]))

    def test_validation_checks_source_ids_and_exact_question_coverage(self) -> None:
        golden = self.fixture("gold.jsonl", [self.question()])
        corpus = self.fixture("corpus.jsonl", [{"doc_id": "v1", "text": "Source fictive."}, {"doc_id": "v2", "text": "Autre source fictive."}])
        results = self.fixture("results.jsonl", [self.result(context_doc_ids=["v1"])])
        self.assertTrue(validate_files(golden, corpus, results)["valid"])
        unknown_gold = self.fixture("unknown-gold.jsonl", [self.question(gold_docs=["unknown"])])
        with self.assertRaisesRegex(ValueError, "gold_docs absent"):
            validate_files(unknown_gold, corpus)
        for number, field in enumerate(("retrieved_doc_ids", "context_doc_ids")):
            unknown_result = self.fixture(f"unknown-result-{number}.jsonl", [self.result(**{field: ["unknown"]})])
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field + " absent"):
                    validate_files(golden, corpus, unknown_result)
        missing = self.fixture("missing.jsonl", [])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_files(golden, corpus, missing)
        unexpected = self.fixture("unexpected.jsonl", [self.result(id="other")])
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_files(golden, corpus, unexpected)

    def test_invented_citations_are_measurable_warnings_not_invalid_inputs(self) -> None:
        golden = self.fixture("citation-gold.jsonl", [self.question()])
        corpus = self.fixture("citation-corpus.jsonl", [{"doc_id": "v1", "text": "Source fictive."}])
        results = self.fixture("citation-results.jsonl", [self.result(retrieved_doc_ids=["v1"], citations=["v1", "invented"] )])
        summary = validate_files(golden, corpus, results)
        self.assertTrue(summary["valid"])
        self.assertTrue(summary["citation_reference_check_performed"])
        self.assertEqual(summary["unresolved_citations"], [{"id": "q1", "doc_ids": ["invented"]}])
        self.assertEqual(summary["unresolved_citation_count"], 1)
        bad_golden = self.fixture("bad-citation-gold.jsonl", [self.question(gold_docs=["invented"])])
        with self.assertRaisesRegex(ValueError, "gold_docs absent"):
            validate_files(bad_golden, corpus, results)
        no_corpus = validate_files(golden, results=results)
        self.assertFalse(no_corpus["citation_reference_check_performed"])

    def test_expected_count_is_optional_and_requires_a_positive_exact_count(self) -> None:
        golden = self.fixture("count-gold.jsonl", [self.question()])
        self.assertEqual(validate_files(golden)["questions"], 1)
        self.assertEqual(validate_files(golden, expected_count=1)["expected_count"], 1)
        with self.assertRaisesRegex(ValueError, "expected 40 questions, found 1"):
            validate_files(golden, expected_count=40)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    validate_files(golden, expected_count=count)
        for value in ("0", "-1", "1.5", "many"):
            with self.subTest(value=value):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                    validate_main(["--questions", str(golden), "--expected-count", value])
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
