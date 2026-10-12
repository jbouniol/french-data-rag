"""Synthetic evidence checks, deliberately independent of legal assertions."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.checks import evaluate_checks


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.golden = {"id": "q1", "question": "Synthetic?", "answer": "Synthetic answer", "gold_docs": ["d1", "d2"]}
        self.result = {"id": "q1", "answer": "Synthetic answer", "citations": ["d1"], "retrieved_doc_ids": ["other", "d1", "d2"]}
        self.corpus = {
            "d1": {"doc_id": "d1", "text": "synthetic", "valid_from": "2020-01-01", "valid_to": "2021-01-01"},
            "d2": {"doc_id": "d2", "text": "synthetic", "valid_from": "2021-01-01", "valid_to": None},
        }

    def check(self, *, golden=None, result=None, corpus=None, annotation=None, k=5):
        return evaluate_checks(golden or self.golden, result or self.result, self.corpus if corpus is None else corpus, annotation, k=k)

    def test_recall_at_k_and_full_ranking_mrr_have_distinct_scopes(self):
        observed = self.check(k=1)
        self.assertEqual(observed["recall_at_k"], 0)
        self.assertEqual(observed["mrr"], .5)
        self.assertEqual(self.check(k=2)["recall_at_k"], .5)
        self.assertEqual(self.check()["recall_at_k"], 1)

    def test_first_occurrence_is_used_without_double_counting(self):
        observed = self.check(result={**self.result, "retrieved_doc_ids": ["other", "other", "d1", "d1", "d2"]}, k=2)
        self.assertEqual(observed["recall_at_k"], .5)
        self.assertEqual(observed["mrr"], .5)

    def test_missing_ranking_is_unknown_empty_ranking_is_zero(self):
        missing = self.result.copy()
        del missing["retrieved_doc_ids"]
        self.assertIsNone(self.check(result=missing)["recall_at_k"])
        self.assertIsNone(self.check(result=missing)["mrr"])
        empty = self.check(result={**self.result, "retrieved_doc_ids": []})
        self.assertEqual(empty["recall_at_k"], 0)
        self.assertEqual(empty["mrr"], 0)

    def test_questions_without_gold_docs_are_not_retrieval_failures(self):
        observed = self.check(golden={**self.golden, "gold_docs": []})
        self.assertIsNone(observed["recall_at_k"])
        self.assertIsNone(observed["mrr"])
        self.assertNotIn("no_gold_document_at_k", observed["observations"])

    def test_resolvable_citation_does_not_claim_semantic_support(self):
        observed = self.check()
        self.assertTrue(observed["citation_resolvable"])
        self.assertNotIn("citation_support", observed)
        self.assertFalse(self.check(result={**self.result, "citations": ["unknown"]})["citation_resolvable"])
        self.assertIsNone(self.check(result={**self.result, "citations": []})["citation_resolvable"])
        self.assertIsNone(evaluate_checks(self.golden, self.result, None)["citation_resolvable"])

    def test_date_start_inclusive_end_exclusive(self):
        self.assertTrue(self.check(annotation={"as_of": "2020-01-01"})["temporal_validity"])
        self.assertFalse(self.check(annotation={"as_of": "2021-01-01"})["temporal_validity"])

    def test_missing_and_double_null_dates_remain_unknown(self):
        for document in ({"text": "synthetic"}, {"valid_from": "2020-01-01"}, {"valid_from": None, "valid_to": None}):
            observed = self.check(corpus={"d1": document}, annotation={"as_of": "2020-07-01"})
            self.assertIsNone(observed["temporal_validity"])
            self.assertEqual(observed["temporal_unknown_doc_ids"], ["d1"])
        open_end = self.check(result={**self.result, "citations": ["d2"]}, annotation={"as_of": "2021-01-01"})
        self.assertTrue(open_end["temporal_validity"])

    def test_proven_temporal_failure_survives_other_unknown_citations(self):
        observed = self.check(result={**self.result, "citations": ["d1", "unknown"]}, annotation={"as_of": "2021-01-01"})
        self.assertFalse(observed["temporal_validity"])
        self.assertEqual(observed["temporal_invalid_doc_ids"], ["d1"])
        self.assertEqual(observed["temporal_unknown_doc_ids"], ["unknown"])

    def test_retrieved_documents_are_not_assumed_to_be_final_context(self):
        self.assertIsNone(self.check()["gold_in_context"])
        self.assertIsNone(self.check()["context_gold_recall"])
        partial = self.check(result={**self.result, "context_doc_ids": ["d1"]})
        self.assertFalse(partial["gold_in_context"])
        self.assertEqual(partial["context_gold_recall"], .5)
        self.assertIn("gold_partial_in_final_context", partial["observations"])
        self.assertNotIn("gold_absent_from_final_context", partial["observations"])
        self.assertTrue(self.check(result={**self.result, "context_doc_ids": ["d1", "d2"]})["gold_in_context"])

    def test_decimal_calculation_inclusive_tolerance_and_unit(self):
        expected = {"value": "100.00", "unit": "EUR", "tolerance": "0.01"}
        for value, correct in (("100.01", True), ("99.99", True), ("100.011", False)):
            observed = self.check(result={**self.result, "calculation": {"value": value, "unit": "EUR"}}, annotation={"expected_calculation": expected})
            self.assertIs(observed["calculation_correct"], correct)
        wrong_unit = self.check(result={**self.result, "calculation": {"value": "100", "unit": "percent"}}, annotation={"expected_calculation": expected})
        self.assertFalse(wrong_unit["calculation_correct"])

    def test_no_calculation_is_extracted_from_answer(self):
        observed = self.check(result={**self.result, "answer": "100.00 EUR"}, annotation={"expected_calculation": {"value": "100.00", "unit": "EUR"}})
        self.assertIsNone(observed["calculation_correct"])
        nonfinite = self.check(result={**self.result, "calculation": {"value": "NaN", "unit": "EUR"}}, annotation={"expected_calculation": {"value": "100", "unit": "EUR"}})
        self.assertIsNone(nonfinite["calculation_correct"])

    def test_empty_answer_remains_visible_and_invalid_k_is_rejected(self):
        observed = self.check(result={**self.result, "answer": " "})
        self.assertFalse(observed["answered"])
        self.assertIn("unanswered", observed["observations"])
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                self.check(k=value)


if __name__ == "__main__":
    unittest.main()
