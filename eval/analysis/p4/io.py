"""Strict, dependency-free JSONL contracts for the P4 evaluation files.

Identifiers always refer to a *versioned corpus record*, not an article number.
Corpus validity intervals use an inclusive start and an exclusive end.
No environment variables, credentials, or API responses are read here.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping


TRACKING_FIELDS = ("tokens_in", "tokens_out", "latency_ms", "tool_calls", "stop_reason")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_DRAFT_STATUSES = ("draft", "draft_needs_sources")
_UNFILLED_PLACEHOLDERS = (
    "[DATE_DE_REFERENCE]",
    "[SMIC_MENSUEL_BRUT_EUR]",
    "[HYPOTHESES_CONTRAT]",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number {value!r}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number is outside the finite numeric range: {value!r}")
    return parsed


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSON objects, rejecting malformed or ambiguous input.

    Blank lines, repeated object keys and non-finite JSON numbers are errors.
    File and line locations accompany all JSON format errors.
    """
    source = Path(path)
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            location = f"{source}: line {line_number}"
            if not line.strip():
                raise ValueError(f"{location}: expected a JSON object, found a blank line")
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=_unique_keys,
                    parse_constant=_reject_constant,
                    parse_float=_finite_float,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{location}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{location}: expected a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write UTF-8 JSONL; never overwrite a previous result.

    Serialize before creating the file, so serialization failures do not leave
    a partial output. Existing parent directories must already be present.
    """
    target = Path(path)
    lines: list[str] = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"output row {number}: expected a JSON object")
        try:
            lines.append(json.dumps(dict(row), ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"output row {number}: {exc}") from exc
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def _string(value: Any, field: str, location: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise ValueError(f"{location}: {field} must be {qualifier}")
    return value


def _strings(value: Any, field: str, location: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{location}: {field} must be a list of strings")
    for item in value:
        _string(item, f"{field} item", location)
    if len(value) != len(set(value)):
        raise ValueError(f"{location}: {field} contains duplicate identifiers")
    return value


def _required(row: Mapping[str, Any], fields: Iterable[str], location: str) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{location}: missing required field(s): {', '.join(missing)}")


def index_by_id(
    rows: Iterable[dict[str, Any]], label: str = "records"
) -> dict[str, dict[str, Any]]:
    """Index rows by their nonempty `id`, rejecting duplicate identifiers."""
    indexed: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        location = f"{label}: record {number}"
        if not isinstance(row, dict):
            raise ValueError(f"{location}: expected an object")
        _required(row, ("id",), location)
        identifier = _string(row["id"], "id", location)
        if identifier in indexed:
            raise ValueError(f"{location}: duplicate id {identifier!r}")
        indexed[identifier] = row
    return indexed


def load_golden(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load verified questions; an empty `gold_docs` denotes abstention."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: golden set must contain at least one question")
    for number, row in enumerate(rows, 1):
        location = f"{path}: line {number}"
        _required(row, ("id", "question", "answer", "gold_docs"), location)
        for field in ("id", "question", "answer"):
            _string(row[field], field, location)
        if row.get("status") in _DRAFT_STATUSES:
            raise ValueError(
                f"{location}: draft question must be source-verified before becoming golden"
            )
        for field in ("question", "answer"):
            unfilled = [token for token in _UNFILLED_PLACEHOLDERS if token in row[field]]
            if unfilled:
                raise ValueError(f"{location}: {field} contains unfilled placeholder(s): {unfilled}")
        _strings(row["gold_docs"], "gold_docs", location)
    return index_by_id(rows, str(path))


def _date(value: Any, field: str, location: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise ValueError(f"{location}: {field} must be YYYY-MM-DD or null")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{location}: {field} is not a valid calendar date") from exc


def load_corpus(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load versioned source texts without changing or inventing identifiers."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: corpus must contain at least one document")
    indexed: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, 1):
        location = f"{path}: line {number}"
        _required(row, ("doc_id", "text"), location)
        identifier = _string(row["doc_id"], "doc_id", location)
        _string(row["text"], "text", location)
        if row.get("article_number") is not None:
            _string(row["article_number"], "article_number", location)
        start = _date(row.get("valid_from"), "valid_from", location)
        end = _date(row.get("valid_to"), "valid_to", location)
        if start is not None and end is not None and start >= end:
            raise ValueError(f"{location}: valid_from must be earlier than valid_to")
        if identifier in indexed:
            raise ValueError(f"{location}: duplicate doc_id {identifier!r}")
        indexed[identifier] = row
    return indexed


def _tracking(row: dict[str, Any], location: str, strict: bool) -> None:
    if strict:
        _required(row, TRACKING_FIELDS, location)
    for field in ("tokens_in", "tokens_out"):
        if field in row:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{location}: {field} must be a nonnegative integer")
    if "latency_ms" in row:
        value = row["latency_ms"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or value < 0
        ):
            raise ValueError(f"{location}: latency_ms must be a finite nonnegative number")
    if "tool_calls" in row:
        value = row["tool_calls"]
        if not isinstance(value, list) or any(not isinstance(call, dict) for call in value):
            raise ValueError(f"{location}: tool_calls must be a list of objects")
    if "stop_reason" in row:
        _string(row["stop_reason"], "stop_reason", location)


def load_results(
    path: str | Path, *, strict_tracking: bool = False
) -> dict[str, dict[str, Any]]:
    """Load outputs with explicit retrieval ranks and cited source IDs.

    `cited_sources` and `retrieved_document_ids` are accepted only when their
    canonical field is absent. The originals are preserved alongside normalized
    canonical fields. No guessed field mappings or implicit empty lists are used.
    Empty answers are accepted so failed generations remain in the evaluation.
    """
    rows = read_jsonl(path)
    for number, row in enumerate(rows, 1):
        location = f"{path}: line {number}"
        for canonical, alias in (
            ("citations", "cited_sources"),
            ("retrieved_doc_ids", "retrieved_document_ids"),
        ):
            if canonical not in row and alias in row:
                row[canonical] = row[alias]
            elif canonical in row and alias in row and row[canonical] != row[alias]:
                raise ValueError(f"{location}: conflicting {canonical} and {alias}")
        _required(row, ("id", "answer", "citations", "retrieved_doc_ids"), location)
        _string(row["id"], "id", location)
        _string(row["answer"], "answer", location, allow_empty=True)
        for field in ("citations", "retrieved_doc_ids"):
            _strings(row[field], field, location)
        if "context_doc_ids" in row:
            _strings(row["context_doc_ids"], "context_doc_ids", location)
        _tracking(row, location, strict_tracking)
    return index_by_id(rows, str(path))
