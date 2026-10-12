"""Reporting tests with synthetic runs, never model or legal evaluations."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import build_report, main


def question(identifier, docs):
    return {"id": identifier, "question": "Synthetic question", "answer": "Synthetic reference", "gold_docs": docs}


def result(identifier, docs, **fields):
    return {"id": identifier, "answer": "Synthetic answer", "citations": [], "retrieved_doc_ids": docs, **fields}


def judgment(identifier, correctness, **scores):
    return {"id": identifier, "status": "ok", "scores": {"correctness": correctness, "groundedness": None, "citation_support": None, "abstention_ok": None, "injection_resisted": None, **scores}, "reason": "Synthetic reason"}


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.golden = {"q1": question("q1", ["d1"]), "q2": question("q2", ["d2"]), "q3": question("q3", [])}
        self.results = {"q1": result("q1", ["d1"], tokens_in=0, latency_ms=0), "q2": result("q2", [], latency_ms=20), "q3": result("q3", [], answer="")}

    def test_retrieval_excludes_unanswerable_and_keeps_observed_zero(self):
        checks, report = build_report(self.golden, self.results)
        self.assertIn("| Recall@5 | 0.500 | 2 / 2 | 0 |", report)
        self.assertIn("| MRR | 0.500 | 2 / 2 | 0 |", report)
        self.assertIsNone(checks[2]["recall_at_k"])
        self.assertIn("réponses vides : **1/3**", report)

    def test_missing_judge_scores_are_unknown_not_zero(self):
        judgments = {"q1": judgment("q1", 2), "q2": {"id": "q2", "status": "error", "reason": "Synthetic timeout"}}
        checks, report = build_report(self.golden, self.results, judgments=judgments)
        self.assertIn("| Exactitude | 2.000 | 1 / 3 | 2 |", report)
        self.assertIn("erreurs du juge : **1** ; jugements absents : **1**", report)
        self.assertNotIn("judge_scores", checks[1])
        self.assertIn("judge_error", report)
        self.assertIn("0 = 0, 1 = 0, 2 = 1", report)

    def test_no_judgments_never_claim_accuracy(self):
        _, report = build_report(self.golden, self.results)
        self.assertIn("| Exactitude | non évalué | 0 / 3 | 3 |", report)

    def test_results_must_match_every_expected_id(self):
        for results in ({"q1": self.results["q1"]}, {**self.results, "extra": result("extra", [])}):
            with self.assertRaisesRegex(ValueError, "results:"):
                build_report(self.golden, results)
        with self.assertRaises(ValueError):
            build_report({}, {})

    def test_optional_artifacts_cannot_introduce_unknown_ids(self):
        with self.assertRaisesRegex(ValueError, "annotations:"):
            build_report(self.golden, self.results, annotations={"extra": {"id": "extra"}})
        with self.assertRaisesRegex(ValueError, "judgments:"):
            build_report(self.golden, self.results, judgments={"extra": judgment("extra", 2)})

    def test_invalid_judge_values_are_not_averaged(self):
        for invalid in (3, True, None, "2"):
            with self.assertRaisesRegex(ValueError, "correctness"):
                build_report(self.golden, self.results, judgments={"q1": judgment("q1", invalid)})

    def test_incomplete_verdict_or_empty_reason_is_rejected(self):
        for record in ({"id": "q1", "status": "ok", "scores": {"correctness": 2}, "reason": "Synthetic"}, {**judgment("q1", 2), "reason": " "}):
            with self.assertRaisesRegex(ValueError, "judgment q1"):
                build_report(self.golden, self.results, judgments={"q1": record})

    def test_malformed_annotations_are_not_treated_as_missing_evidence(self):
        for fields in ({"as_of": "2025-02-30"}, {"as_of": "20250101"}, {"expected_calculation": {"unit": "EUR", "value": "NaN"}}, {"expected_calculation": {"unit": "EUR", "value": "1", "tolerance": "-1"}}):
            with self.assertRaisesRegex(ValueError, "annotation q1"):
                build_report(self.golden, self.results, annotations={"q1": {"id": "q1", **fields}})

    def test_corpus_references_must_resolve_except_application_citations(self):
        corpus = {"d1": {"doc_id": "d1", "text": "synthetic"}, "d2": {"doc_id": "d2", "text": "synthetic"}}
        for field in ("retrieved_doc_ids", "context_doc_ids"):
            altered = {**self.results, "q1": {**self.results["q1"], field: ["unknown"]}}
            with self.assertRaisesRegex(ValueError, "unknown corpus IDs"):
                build_report(self.golden, altered, corpus)
        with self.assertRaisesRegex(ValueError, "gold_docs references unknown corpus IDs"):
            build_report(self.golden, self.results, {"d1": corpus["d1"]})
        altered = {**self.results, "q1": {**self.results["q1"], "citations": ["unknown"]}}
        checks, _ = build_report(self.golden, altered, corpus)
        self.assertFalse(checks[0]["citation_resolvable"])

    def test_partial_context_does_not_suggest_correct_generation_inputs(self):
        golden = {"q1": question("q1", ["d1", "d2"])}
        results = {"q1": result("q1", ["d1", "d2"], context_doc_ids=["d1"])}
        checks, _ = build_report(golden, results, judgments={"q1": judgment("q1", 0)})
        self.assertEqual(checks[0]["judge_observations"], ["judge_incorrect"])
        self.assertIn("gold_partial_in_final_context", checks[0]["observations"])

    def test_tracking_keeps_zero_and_missing_distinct(self):
        _, report = build_report(self.golden, self.results)
        self.assertIn("| tokens_in | total 0 ; moyenne 0.000 | 1 / 3 | 2 |", report)
        self.assertIn("| tokens_out | non évalué | 0 / 3 | 3 |", report)
        self.assertIn("| latency_ms | p50 0.00 ; p95 20.00 | 2 / 3 | 1 |", report)
        self.assertIn("petit échantillon (n=2)", report)

    def test_cli_writes_new_artifacts_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            questions_path, results_path = base / "questions.jsonl", base / "results.jsonl"
            for path, records in ((questions_path, self.golden), (results_path, self.results)):
                path.write_text("".join(json.dumps(row) + "\n" for row in records.values()), encoding="utf-8")
            argv = ["--questions", str(questions_path), "--results", str(results_path), "--out", str(base / "out")]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 0)
                original = (base / "out" / "report.md").read_bytes()
                self.assertEqual(main(argv), 2)
            self.assertEqual((base / "out" / "report.md").read_bytes(), original)
            checks = [json.loads(line) for line in (base / "out" / "checks.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(checks), 3)
            self.assertEqual(checks[2]["judge_status"], "missing")


if __name__ == "__main__":
    unittest.main()
