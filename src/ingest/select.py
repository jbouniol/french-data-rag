"""Sélection des articles du corpus à partir de selection.yaml.

Usage: uv run python -m src.ingest.select
"""
import re

import pandas as pd
import yaml

from src.ingest.common import load_chunks

SELECTION_PATH = "src/ingest/selection.yaml"


def normalize_subtitles(s: str | None) -> str:
    """Réduit les espaces multiples de la hiérarchie."""
    return re.sub(r"\s+", " ", s or "").strip()


def load_rules(path: str = SELECTION_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)["rules"]
    for r in rules:
        if not (r.get("number_regex") or r.get("subtitles_regex")):
            raise ValueError(f"Règle {r['name']}: number_regex ou subtitles_regex requis")
    return rules


def rule_mask(articles: pd.DataFrame, rule: dict) -> pd.Series:
    """Articles qui correspondent à une règle."""
    mask = pd.Series(True, index=articles.index)
    if rule.get("number_regex"):
        mask &= articles.number.str.contains(rule["number_regex"], regex=True, na=False)
    if rule.get("subtitles_regex"):
        mask &= articles.subtitles.str.contains(rule["subtitles_regex"], regex=True, na=False)
    if rule.get("exclude_subtitles_regex"):
        mask &= ~articles.subtitles.str.contains(rule["exclude_subtitles_regex"], regex=True, na=False)
    return mask


def select_articles(articles: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Retourne les articles VIGUEUR retenus, avec la règle qui les a ajoutés (colonne `rule`)."""
    arts = articles[articles.status == "VIGUEUR"].copy()
    arts["subtitles"] = arts.subtitles.map(normalize_subtitles)
    selected: dict[str, str] = {}
    for rule in rules:
        matched = arts[rule_mask(arts, rule)]
        new = [d for d in matched.doc_id if d not in selected]
        selected.update({d: rule["name"] for d in new})
        print(f"{rule['name']:<30} matchés={len(matched):>4}  nouveaux={len(new):>4}  total={len(selected):>4}")
    out = arts[arts.doc_id.isin(selected)].copy()
    out["rule"] = out.doc_id.map(selected)
    return out


def main() -> None:
    chunks = load_chunks(["doc_id", "status", "number", "subtitles"])
    articles = chunks.drop_duplicates("doc_id")
    selected = select_articles(articles, load_rules())
    print(f"\nArticles sélectionnés: {len(selected)} (numéros distincts: {selected.number.nunique()})")


if __name__ == "__main__":
    main()
