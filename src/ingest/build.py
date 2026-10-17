"""Build the data/processed/ files from the raw parquet.

Usage: uv run python -m src.ingest.build
"""
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.ingest.common import load_chunks
from src.ingest.select import load_rules, normalize_subtitles, select_articles

OUT_DIR = Path("data/processed")
CHUNK_COLUMNS = [
    "doc_id", "chunk_index", "status", "number", "subtitles",
    "start_date", "end_date", "nota", "text",
]
ARTICLE_COLUMNS = [
    "doc_id", "num", "text", "chunk_text", "hierarchy", "etat",
    "date_debut", "date_fin", "url", "nota", "is_in_corpus",
]
LEVEL_RE = re.compile(r"^(Livre|Titre|Chapitre)\b")
PARTIE_RE = re.compile(r"^\w+ partie\b")


def parse_hierarchy(subtitles: str) -> str:
    """'Partie législative - Sixième partie : … - Livre II : …' -> 'Partie législative > Sixième partie : … > Livre II : …'.

    Keeps code (législative/réglementaire), partie, livre, titre, chapitre.
    When a level appears twice (renamed), the last label wins.
    """
    levels: dict[str, str] = {}
    for part in re.split(r"\s+-\s+", normalize_subtitles(subtitles)):
        label = part.strip().rstrip(".").strip()
        if label.startswith("Partie "):
            levels["code"] = label
        elif PARTIE_RE.match(label):
            levels["partie"] = label
        elif m := LEVEL_RE.match(label):
            levels[m.group(1)] = label
    order = ["code", "partie", "Livre", "Titre", "Chapitre"]
    return " > ".join(levels[k] for k in order if k in levels)


def format_chunk(num: str, hierarchy: str, text: str, nota: str) -> str:
    """Build the contextualized text for retrieval (article number, clean hierarchy, body, and optional nota)."""
    parts = [f"Code du travail - Article {num}", hierarchy, text]
    if nota:
        parts.append(f"Nota : {nota}")
    return "\n".join(p for p in parts if p)


def build_articles(chunks: pd.DataFrame) -> pd.DataFrame:
    """One row per doc_id: text rebuilt by concatenating the chunks in order."""
    chunks = chunks.sort_values(["doc_id", "chunk_index"])
    arts = chunks.drop_duplicates("doc_id").rename(
        columns={"number": "num", "status": "etat"}
    ).copy()
    arts["text"] = chunks.groupby("doc_id", sort=False).text.agg("\n".join).values
    arts["hierarchy"] = arts.subtitles.map(parse_hierarchy)
    arts["date_debut"] = arts.start_date.map(date.fromisoformat)
    arts["date_fin"] = arts.end_date.map(date.fromisoformat)
    arts["nota"] = arts.nota.fillna("")
    arts["chunk_text"] = [
        format_chunk(n, h, t, no)
        for n, h, t, no in zip(arts.num, arts.hierarchy, arts.text, arts.nota)
    ]
    arts["url"] = "https://www.legifrance.gouv.fr/codes/article_lc/" + arts.doc_id
    return arts.reset_index(drop=True)


def build_baseline(doc_ids: set[str]) -> pd.DataFrame:
    """DINUM chunks (with BGE-M3 embeddings) of the corpus articles."""
    cols = ["chunk_id", "doc_id", "chunk_index", "number", "chunk_text", "embeddings_bge-m3"]
    df = load_chunks(cols, filters=[("doc_id", "in", doc_ids)]).sort_values(["doc_id", "chunk_index"])
    df["embedding"] = df["embeddings_bge-m3"].map(json.loads)
    return df.drop(columns="embeddings_bge-m3").rename(columns={"number": "num"}).reset_index(drop=True)


def main() -> None:
    chunks = load_chunks(CHUNK_COLUMNS)
    articles = chunks.drop_duplicates("doc_id")
    selected = select_articles(articles, load_rules())
    corpus_ids = set(selected.doc_id)
    corpus_nums = set(selected.number)

    # MODIFIE_MORT_NE: modified before it ever came into force (date_fin < date_debut), never applicable.
    kept = chunks[chunks.number.isin(corpus_nums) & (chunks.status != "MODIFIE_MORT_NE")]
    versions = build_articles(kept)
    versions["is_in_corpus"] = versions.doc_id.isin(corpus_ids)
    corpus = versions[versions.is_in_corpus]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus[ARTICLE_COLUMNS].to_parquet(OUT_DIR / "articles.parquet", index=False)
    versions[ARTICLE_COLUMNS].to_parquet(OUT_DIR / "article_versions.parquet", index=False)
    baseline = build_baseline(corpus_ids)
    baseline.to_parquet(OUT_DIR / "dinum_baseline_chunks.parquet", index=False)

    print(f"\narticles.parquet: {len(corpus)} articles")
    print(f"article_versions.parquet: {len(versions)} versions ({versions.num.nunique()} article numbers)")
    print(f"dinum_baseline_chunks.parquet: {len(baseline)} chunks")


if __name__ == "__main__":
    main()
