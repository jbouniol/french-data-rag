"""Checks whose evidence is machine-readable, without an LLM or API calls.

These checks do not establish that a citation supports a claim.  In particular,
an existing document and a version valid on the requested date can still be
irrelevant to the answer.  Missing evidence yields None, never an invented zero.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


def _unique_ids(value: Any) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return None
    return list(dict.fromkeys(value))


def _iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.isoformat() == value else None


def _version_at_date(document: Mapping[str, Any], as_of: date) -> bool | None:
    # Missing keys are not equivalent to explicit open bounds.  Both open
    # bounds carry no temporal evidence and therefore also remain unknown.
    if "valid_from" not in document or "valid_to" not in document:
        return None
    start_raw, end_raw = document["valid_from"], document["valid_to"]
    if start_raw is None and end_raw is None:
        return None
    start = _iso_date(start_raw) if start_raw is not None else None
    end = _iso_date(end_raw) if end_raw is not None else None
    if (start_raw is not None and start is None) or (end_raw is not None and end is None):
        return None
    if start is not None and end is not None and start >= end:
        return None
    return (start is None or start <= as_of) and (end is None or as_of < end)


def _decimal(value: Any) -> Decimal | None:
    # Structured decimal strings and integers are supported.  Floats are
    # excluded: the producer should retain its exact calculation as a string.
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        return None
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _calculation(expected: Any, actual: Any) -> bool | None:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return None
    expected_unit, actual_unit = expected.get("unit"), actual.get("unit")
    if expected_unit not in {"EUR", "percent"} or actual_unit not in {"EUR", "percent"}:
        return None
    target, observed = _decimal(expected.get("value")), _decimal(actual.get("value"))
    tolerance = _decimal(expected.get("tolerance", "0.01"))
    if target is None or observed is None or tolerance is None or tolerance < 0:
        return None
    if expected_unit != actual_unit:
        return False
    return abs(observed - target) <= tolerance


def evaluate_checks(
    golden: Mapping[str, Any],
    result: Mapping[str, Any],
    corpus: Mapping[str, Mapping[str, Any]] | None,
    annotation: Mapping[str, Any] | None = None,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Return retrieval, citation, date, calculation and context observations.

    Recall is measured at k; MRR uses the entire returned ranking.  Retrieval
    excludes questions without gold documents.  Date intervals are [from, to).
    Only cited versions are checked for dates; retrieved candidates are not
    implicitly treated as cited or as final generation context.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    annotation = annotation or {}
    gold_ids = _unique_ids(golden.get("gold_docs"))
    if gold_ids is None:
        raise ValueError("gold_docs must be a list of document IDs")
    gold_set = set(gold_ids)
    retrieved = _unique_ids(result.get("retrieved_doc_ids"))
    recall = mrr = None
    if gold_set and retrieved is not None:
        recall = len(gold_set.intersection(retrieved[:k])) / len(gold_set)
        mrr = next((1.0 / rank for rank, doc_id in enumerate(retrieved, 1) if doc_id in gold_set), 0.0)

    citations = _unique_ids(result.get("citations"))
    citation_resolvable = None
    unresolved: list[str] = []
    if citations and corpus is not None:
        unresolved = [doc_id for doc_id in citations if doc_id not in corpus]
        citation_resolvable = not unresolved

    temporal_validity = None
    invalid_dates: list[str] = []
    unknown_dates: list[str] = []
    as_of = _iso_date(annotation.get("as_of"))
    if as_of is not None and citations and corpus is not None:
        for doc_id in citations:
            validity = _version_at_date(corpus[doc_id], as_of) if doc_id in corpus else None
            if validity is False:
                invalid_dates.append(doc_id)
            elif validity is None:
                unknown_dates.append(doc_id)
        temporal_validity = False if invalid_dates else (None if unknown_dates else True)

    context_ids = _unique_ids(result.get("context_doc_ids"))
    context_gold_recall = len(gold_set.intersection(context_ids)) / len(gold_set) if gold_set and context_ids is not None else None
    gold_in_context = context_gold_recall == 1 if context_gold_recall is not None else None
    calculation_correct = _calculation(annotation.get("expected_calculation"), result.get("calculation"))
    answered = isinstance(result.get("answer"), str) and bool(result["answer"].strip())
    observations: list[str] = []
    if not answered:
        observations.append("unanswered")
    if recall == 0:
        observations.append("no_gold_document_at_k")
    elif recall is not None and recall < 1:
        observations.append("partial_gold_recall_at_k")
    if context_gold_recall == 0:
        observations.append("gold_absent_from_final_context")
    elif context_gold_recall is not None and context_gold_recall < 1:
        observations.append("gold_partial_in_final_context")
    if citation_resolvable is False:
        observations.append("unresolved_citation")
    if temporal_validity is False:
        observations.append("citation_invalid_at_requested_date")
    if calculation_correct is False:
        observations.append("structured_calculation_mismatch")
    return {
        "id": golden["id"],
        "k": k,
        "answerable": bool(gold_set),
        "answered": answered,
        "retrieval_available": retrieved is not None,
        "recall_at_k": recall,
        "mrr": mrr,
        "citation_count": len(citations) if citations is not None else None,
        "citation_resolvable": citation_resolvable,
        "unresolved_citation_ids": unresolved,
        "temporal_validity": temporal_validity,
        "temporal_invalid_doc_ids": invalid_dates,
        "temporal_unknown_doc_ids": unknown_dates,
        "calculation_correct": calculation_correct,
        "gold_in_context": gold_in_context,
        "context_gold_recall": context_gold_recall,
        "observations": observations,
    }
