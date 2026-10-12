#!/usr/bin/env python3
"""Produce a transparent evaluation report from existing run artifacts.

This command performs no model calls and does not run the RAG application.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from p4.checks import evaluate_checks
from p4.io import index_by_id, load_corpus, load_golden, load_results, read_jsonl, write_jsonl
from p4.judging import validate_verdict


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _mean(values: list[float | int]) -> str:
    return f"{sum(values) / len(values):.3f}" if values else "non évalué"


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _quantile(values: list[float | int], probability: float) -> str:
    if not values:
        return "non évalué"
    ordered = sorted(values)
    return f"{ordered[max(0, math.ceil(probability * len(ordered)) - 1)]:.2f}"


def _validate_ids(label: str, records: Mapping[str, Any], expected: set[str], *, complete: bool) -> None:
    extra = set(records) - expected
    missing = expected - set(records) if complete else set()
    if extra or missing:
        detail = []
        if missing:
            detail.append("missing IDs: " + ", ".join(sorted(missing)))
        if extra:
            detail.append("unexpected IDs: " + ", ".join(sorted(extra)))
        raise ValueError(f"{label}: {'; '.join(detail)}")


def _validate_judgments(judgments: Mapping[str, Mapping[str, Any]]) -> None:
    for question_id, record in judgments.items():
        status = record.get("status")
        if status not in {"ok", "error"}:
            raise ValueError(f"judgment {question_id}: status must be ok or error")
        if status == "error":
            continue
        try:
            validate_verdict(dict(record))
        except ValueError as exc:
            raise ValueError(f"judgment {question_id}: {exc}") from exc


def _validate_annotations(annotations: Mapping[str, Mapping[str, Any]]) -> None:
    for question_id, record in annotations.items():
        if record.get("category") is not None and (not isinstance(record["category"], str) or not record["category"].strip()):
            raise ValueError(f"annotation {question_id}: category must be a nonempty string")
        as_of = record.get("as_of")
        if as_of is not None:
            try:
                parsed = date.fromisoformat(as_of)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"annotation {question_id}: as_of must be YYYY-MM-DD or null") from exc
            if parsed.isoformat() != as_of:
                raise ValueError(f"annotation {question_id}: as_of must be YYYY-MM-DD or null")
        calculation = record.get("expected_calculation")
        if calculation is None:
            continue
        if not isinstance(calculation, Mapping) or calculation.get("unit") not in {"EUR", "percent"}:
            raise ValueError(f"annotation {question_id}: expected_calculation requires value and unit EUR or percent")
        for field in ("value", "tolerance"):
            raw = calculation.get(field, "0.01" if field == "tolerance" else None)
            if not isinstance(raw, str):
                raise ValueError(f"annotation {question_id}: expected_calculation.{field} must be a decimal string")
            try:
                value = Decimal(raw)
            except InvalidOperation as exc:
                raise ValueError(f"annotation {question_id}: expected_calculation.{field} must be finite") from exc
            if not value.is_finite() or (field == "tolerance" and value < 0):
                raise ValueError(f"annotation {question_id}: expected_calculation.{field} is invalid")


def _validate_corpus_references(
    golden: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    corpus: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    if corpus is None:
        return
    for label, records, fields in (("golden", golden, ("gold_docs",)), ("results", results, ("retrieved_doc_ids", "context_doc_ids"))):
        for question_id, record in records.items():
            for field in fields:
                if field not in record:
                    continue
                ids = record[field]
                if not isinstance(ids, list) or any(not isinstance(doc_id, str) for doc_id in ids):
                    raise ValueError(f"{label} {question_id}: {field} must be a list of document IDs")
                missing = [doc_id for doc_id in ids if doc_id not in corpus]
                if missing:
                    raise ValueError(f"{label} {question_id}: {field} references unknown corpus IDs: {', '.join(missing)}")


def _judge_observations(check: Mapping[str, Any], scores: Mapping[str, Any]) -> list[str]:
    observations: list[str] = []
    if scores.get("correctness") == 0:
        observations.append("judge_incorrect_with_gold_context" if check["gold_in_context"] is True else "judge_incorrect")
    elif scores.get("correctness") == 1:
        observations.append("judge_partially_correct")
    if scores.get("groundedness") == 0:
        observations.append("judge_ungrounded")
    if scores.get("citation_support") == 0:
        observations.append("judge_citation_unsupported")
    if scores.get("abstention_ok") is False:
        observations.append("judge_abstention_inappropriate")
    if scores.get("injection_resisted") is False:
        observations.append("judge_injection_followed")
    return observations


def _bool_row(label: str, values: list[Any]) -> str:
    yes = sum(value is True for value in values)
    no = sum(value is False for value in values)
    return f"| {label} | {yes} | {no} | {len(values) - yes - no} |"


def build_report(
    golden: Mapping[str, Mapping[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    corpus: Mapping[str, Mapping[str, Any]] | None = None,
    annotations: Mapping[str, Mapping[str, Any]] | None = None,
    judgments: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    k: int = 5,
) -> tuple[list[dict[str, Any]], str]:
    """Build report text and records; all expected run IDs must be present."""
    if not golden:
        raise ValueError("The golden set is empty")
    annotations, judgments = annotations or {}, judgments or {}
    expected = set(golden)
    _validate_ids("results", results, expected, complete=True)
    _validate_ids("annotations", annotations, expected, complete=False)
    _validate_ids("judgments", judgments, expected, complete=False)
    _validate_annotations(annotations)
    _validate_judgments(judgments)
    _validate_corpus_references(golden, results, corpus)
    checks: list[dict[str, Any]] = []
    for question_id, reference in golden.items():
        check = evaluate_checks(reference, results[question_id], corpus, annotations.get(question_id), k=k)
        check["category"] = annotations.get(question_id, {}).get("category") or "non renseignée"
        judgment = judgments.get(question_id)
        check["judge_status"] = judgment["status"] if judgment else "missing"
        check["judge_observations"] = []
        if judgment and judgment["status"] == "ok":
            check["judge_scores"] = dict(judgment["scores"])
            check["judge_observations"] = _judge_observations(check, judgment["scores"])
        if judgment and isinstance(judgment.get("reason"), str):
            check["judge_reason"] = judgment["reason"]
        checks.append(check)

    total = len(checks)
    answerable = [row for row in checks if row["answerable"]]
    recall = [row["recall_at_k"] for row in answerable if row["recall_at_k"] is not None]
    reciprocal_rank = [row["mrr"] for row in answerable if row["mrr"] is not None]
    answered = sum(row["answered"] for row in checks)
    judge_statuses = Counter(row["judge_status"] for row in checks)
    lines = [
        "# Rapport d’évaluation P4",
        "",
        f"Questions attendues : **{total}** ; résultats présents : **{len(results)}**. Les identifiants correspondent exactement.",
        f"Réponses non vides : **{answered}/{total}** ; réponses vides : **{total - answered}/{total}**. Une réponse non vide peut être une abstention et ne prouve pas son exactitude.",
        f"Questions avec documents de référence : **{len(answerable)}** ; sans document de référence : **{total - len(answerable)}**.",
        "",
        "## Retrieval",
        "",
        "| Mesure | Moyenne | Évalués / éligibles | Inconnus |",
        "| --- | --- | --- | --- |",
        f"| Recall@{k} | {_mean(recall)} | {len(recall)} / {len(answerable)} | {len(answerable) - len(recall)} |",
        f"| MRR | {_mean(reciprocal_rank)} | {len(reciprocal_rank)} / {len(answerable)} | {len(answerable) - len(reciprocal_rank)} |",
        "",
        f"Recall@{k} mesure la fraction des documents de référence présents dans les {k} premiers résultats. MRR utilise le rang du premier document de référence dans toute la liste retournée. Les questions sans document de référence sont exclues des deux moyennes ; un zéro observé est conservé.",
        "",
        "## Vérifications déterministes",
        "",
        "| Vérification | Vrai | Faux | Inconnu ou non applicable |",
        "| --- | --- | --- | --- |",
        _bool_row("Tous les documents de référence dans le contexte final", [row["gold_in_context"] for row in checks]),
        _bool_row("Toutes les citations identifiables dans le corpus", [row["citation_resolvable"] for row in checks]),
        _bool_row("Versions citées valides à la date demandée", [row["temporal_validity"] for row in checks]),
        _bool_row("Calcul structuré conforme à la référence", [row["calculation_correct"] for row in checks]),
        "",
        "Chaque ligne porte sur les questions, pas sur le nombre de citations. Une citation identifiable ne prouve pas qu’elle soutient une affirmation. Les dates suivent [valid_from, valid_to) ; les bornes manquantes et deux bornes nulles restent inconnues. Les valeurs de calcul proviennent uniquement des champs structurés, jamais de nombres extraits de la réponse. Le contexte final n’est pas déduit des documents récupérés.",
        "",
        "## LLM-judge",
        "",
        f"Jugements réussis : **{judge_statuses['ok']}/{total}** ; erreurs du juge : **{judge_statuses['error']}** ; jugements absents : **{judge_statuses['missing']}**. Une erreur ou une absence ne devient pas une note zéro.",
        "",
        "| Note (0–2) | Moyenne | Évalués / questions | Inconnus |",
        "| --- | --- | --- | --- |",
    ]
    for score_name, label in (("correctness", "Exactitude"), ("groundedness", "Fondement dans les sources"), ("citation_support", "Soutien par les citations")):
        values = [row["judge_scores"][score_name] for row in checks if row["judge_status"] == "ok" and row["judge_scores"].get(score_name) is not None]
        lines.append(f"| {label} | {_mean(values)} | {len(values)} / {total} | {total - len(values)} |")
    correctness = [row["judge_scores"]["correctness"] for row in checks if row["judge_status"] == "ok"]
    distribution = Counter(correctness)
    lines.extend([
        "",
        f"Distribution de l’exactitude parmi {len(correctness)} jugements : 0 = {distribution[0]}, 1 = {distribution[1]}, 2 = {distribution[2]}.",
        "",
        "| Jugement binaire | Vrai | Faux | Inconnu ou non applicable |",
        "| --- | --- | --- | --- |",
        _bool_row("Abstention appropriée", [row.get("judge_scores", {}).get("abstention_ok") for row in checks]),
        _bool_row("Injection ignorée", [row.get("judge_scores", {}).get("injection_resisted") for row in checks]),
        "",
        "Les notes du juge sont des estimations à relire humainement. Les erreurs ci-dessous sont des observations ; elles ne suffisent pas à attribuer une cause à l’index, au reranker ou à l’agent.",
        "",
        "## Résultats par catégorie",
        "",
        "| Catégorie | Questions | Avec documents de référence | Recall moyen (n) | Exactitude moyenne (n) |",
        "| --- | --- | --- | --- | --- |",
    ])
    categories = sorted({str(row["category"]) for row in checks})
    for category in categories:
        members = [row for row in checks if str(row["category"]) == category]
        cat_recall = [row["recall_at_k"] for row in members if row["recall_at_k"] is not None]
        cat_correctness = [row["judge_scores"]["correctness"] for row in members if row["judge_status"] == "ok"]
        lines.append(f"| {_cell(category)} | {len(members)} | {sum(row['answerable'] for row in members)} | {_mean(cat_recall)} ({len(cat_recall)}) | {_mean(cat_correctness)} ({len(cat_correctness)}) |")

    lines.extend(["", "## Tracking", "", "| Champ | Résultat sur valeurs disponibles | Observés / questions | Inconnus |", "| --- | --- | --- | --- |"])
    for field in ("tokens_in", "tokens_out"):
        values = [row[field] for row in results.values() if _is_number(row.get(field))]
        displayed = f"total {sum(values)} ; moyenne {_mean(values)}" if values else "non évalué"
        lines.append(f"| {field} | {displayed} | {len(values)} / {total} | {total - len(values)} |")
    latencies = [row["latency_ms"] for row in results.values() if _is_number(row.get("latency_ms"))]
    lines.append(f"| latency_ms | p50 {_quantile(latencies, .5)} ; p95 {_quantile(latencies, .95)} | {len(latencies)} / {total} | {total - len(latencies)} |")
    for field in ("tool_calls", "stop_reason"):
        count = sum(field in row and row[field] is not None for row in results.values())
        lines.append(f"| {field} | présence du champ | {count} / {total} | {total - count} |")
    lines.extend(["", "Les quantiles de latence utilisent le rang le plus proche sur les seules observations disponibles. Aucune valeur manquante n’est remplacée par zéro ; les totaux de tokens ne couvrent que les observations présentes."])
    if len(latencies) < 20:
        lines.append(f"Latence : petit échantillon (n={len(latencies)}) ; les quantiles ne permettent pas une conclusion robuste sur les performances.")

    lines.extend(["", "## Observations à examiner", "", "| Question | Observations | Motif du juge, si disponible |", "| --- | --- | --- |"])
    flagged = [row for row in checks if row["observations"] or row["judge_observations"] or row["judge_status"] == "error"]
    for row in flagged:
        labels = row["observations"] + row["judge_observations"]
        if row["judge_status"] == "error":
            labels = labels + ["judge_error"]
        lines.append(f"| {_cell(row['id'])} | {_cell(', '.join(labels))} | {_cell(row.get('judge_reason', '')[:500])} |")
    if not flagged:
        lines.append("| — | Aucune anomalie détectée par les vérifications disponibles | Cela ne prouve pas l’absence d’erreurs |")
    lines.extend([
        "",
        "## Limites et risques à examiner",
        "",
        "Les points suivants sont des limites possibles à documenter, pas des incidents observés dans ce rapport : couverture du corpus juridique et des conventions collectives ; disponibilité des versions historiques ; date et hypothèses du SMIC ; représentativité du golden set ; biais du LLM-judge et besoin de relecture humaine. Une réussite sur ce jeu ne garantit pas une réponse juridique fiable pour une autre situation.",
        "",
        "Aucune comparaison de coût en euros n’est calculée sans tarifs et versions de modèles. Les champs absents, le corpus non fourni et les jugements non exécutés limitent explicitement les conclusions.",
        "",
    ])
    return checks, "\n".join(lines)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output directory; existing report/checks are never overwritten")
    parser.add_argument("--k", default=5, type=_positive_integer)
    args = parser.parse_args(argv)
    try:
        golden = load_golden(args.questions)
        results = load_results(args.results)
        corpus = load_corpus(args.corpus) if args.corpus else None
        annotations = index_by_id(read_jsonl(args.annotations)) if args.annotations else {}
        judgments = index_by_id(read_jsonl(args.judgments)) if args.judgments else {}
        checks, report = build_report(golden, results, corpus, annotations, judgments, k=args.k)
        checks_path, report_path = args.out / "checks.jsonl", args.out / "report.md"
        if checks_path.exists() or report_path.exists():
            raise FileExistsError(f"Output already exists in {args.out}; choose a new --out directory")
        args.out.mkdir(parents=True, exist_ok=True)
        write_jsonl(checks_path, checks)
        with report_path.open("x", encoding="utf-8") as handle:
            handle.write(report)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Report: {report_path}")
    print(f"Checks: {checks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
