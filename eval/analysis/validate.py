#!/usr/bin/env python3
"""Validate P4 input contracts and source references before evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from p4.io import load_corpus, load_golden, load_results


def validate_files(
    questions: str | Path,
    corpus: str | Path | None = None,
    results: str | Path | None = None,
    *,
    strict_tracking: bool = False,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate input contracts and return counts plus unresolved citation warnings.

    Unknown references in the golden set, retrieval, or generation context mean
    inputs are inconsistent. Invented citations are an application failure to
    measure, so they produce warnings without preventing the evaluation.
    """
    if expected_count is not None and (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    golden = load_golden(questions)
    if expected_count is not None and len(golden) != expected_count:
        raise ValueError(f"expected {expected_count} questions, found {len(golden)}")
    sources = load_corpus(corpus) if corpus is not None else None
    outputs = load_results(results, strict_tracking=strict_tracking) if results is not None else None
    unresolved_citations: list[dict[str, Any]] = []
    if sources is not None:
        for identifier, row in golden.items():
            missing = [doc for doc in row["gold_docs"] if doc not in sources]
            if missing:
                raise ValueError(f"question {identifier!r}: gold_docs absent from corpus: {missing}")
    if outputs is not None:
        missing = sorted(golden.keys() - outputs.keys())
        unexpected = sorted(outputs.keys() - golden.keys())
        if missing or unexpected:
            raise ValueError(
                f"result IDs must exactly match question IDs; missing={missing}, unexpected={unexpected}"
            )
        if sources is not None:
            for identifier, row in outputs.items():
                for field in ("retrieved_doc_ids", "context_doc_ids"):
                    missing_sources = [doc for doc in row.get(field, []) if doc not in sources]
                    if missing_sources:
                        raise ValueError(
                            f"result {identifier!r}: {field} absent from corpus: {missing_sources}"
                        )
                unknown_citations = [doc for doc in row["citations"] if doc not in sources]
                if unknown_citations:
                    unresolved_citations.append({"id": identifier, "doc_ids": unknown_citations})
    return {
        "valid": True,
        "questions": len(golden),
        "documents": len(sources) if sources is not None else None,
        "results": len(outputs) if outputs is not None else None,
        "strict_tracking": strict_tracking,
        "expected_count": expected_count,
        "citation_reference_check_performed": sources is not None and outputs is not None,
        "unresolved_citations": unresolved_citations,
        "unresolved_citation_count": sum(len(item["doc_ids"]) for item in unresolved_citations),
    }


def _positive_int(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Verified golden JSONL")
    parser.add_argument("--corpus", help="Versioned source JSONL (doc_id, text, optional dates)")
    parser.add_argument("--results", help="Runner output JSONL")
    parser.add_argument(
        "--expected-count", type=_positive_int, help="Require exactly this many verified questions (e.g. 40)"
    )
    parser.add_argument(
        "--strict-tracking", action="store_true", help="Require token, latency, tool and stop tracking"
    )
    args = parser.parse_args(argv)
    try:
        summary = validate_files(
            args.questions,
            args.corpus,
            args.results,
            strict_tracking=args.strict_tracking,
            expected_count=args.expected_count,
        )
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
