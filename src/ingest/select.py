"""Select the corpus articles from selection.yaml.

Usage: uv run python -m src.ingest.select
"""
import re

import pandas as pd
import yaml

from src.ingest.common import load_chunks

SELECTION_PATH = "src/ingest/selection.yaml"


def normalize_subtitles(s: str | None) -> str:
    """Collapse the repeated whitespace of the hierarchy."""
    return re.sub(r"\s+", " ", s or "").strip()


def load_rules(path: str = SELECTION_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rules = yaml.safe_load(f)["rules"]
    for r in rules:
        if not (r.get("number_regex") or r.get("subtitles_regex")):
            raise ValueError(f"Rule {r['name']}: number_regex or subtitles_regex required")
    return rules


def rule_mask(articles: pd.DataFrame, rule: dict) -> pd.Series:
    """Articles matching a rule."""
    mask = pd.Series(True, index=articles.index)
    if rule.get("number_regex"):
        mask &= articles.number.str.contains(rule["number_regex"], regex=True, na=False, case=False)
    if rule.get("subtitles_regex"):
        mask &= articles.subtitles.str.contains(rule["subtitles_regex"], regex=True, na=False)
    if rule.get("exclude_subtitles_regex"):
        mask &= ~articles.subtitles.str.contains(rule["exclude_subtitles_regex"], regex=True, na=False, case=False)
    return mask


def select_articles(articles: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Return the selected VIGUEUR articles, with the rule that added them (`rule` column)."""
    arts = articles[articles.status == "VIGUEUR"].copy()
    arts["subtitles"] = arts.subtitles.map(normalize_subtitles)
    selected: dict[str, str] = {}
    for rule in rules:
        matched = arts[rule_mask(arts, rule)]
        new = [d for d in matched.doc_id if d not in selected]
        selected.update({d: rule["name"] for d in new})
        print(f"{rule['name']:<30} matched={len(matched):>4}  new={len(new):>4}  total={len(selected):>4}")
    out = arts[arts.doc_id.isin(selected)].copy()
    out["rule"] = out.doc_id.map(selected)
    return out


def main() -> None:
    chunks = load_chunks(["doc_id", "status", "number", "subtitles"])
    articles = chunks.drop_duplicates("doc_id")
    selected = select_articles(articles, load_rules())
    print(f"\nSelected articles: {len(selected)} (distinct numbers: {selected.number.nunique()})")


if __name__ == "__main__":
    main()
