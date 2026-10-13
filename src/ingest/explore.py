"""Exploration du dossier Code du travail de AgentPublic/legi (lecture seule).

Usage: uv run python -m src.ingest.explore
"""
import pandas as pd
import pyarrow.parquet as pq

RAW_PATH = "data/raw/data/legi-latest/legi_code_du_travail/legi_code_du_travail_part_0.parquet"
# On exclut les colonnes lourdes (embeddings, chunk_text) pour l'exploration.
COLUMNS = [
    "chunk_id", "doc_id", "chunk_index", "status", "subtitles",
    "number", "start_date", "end_date", "text",
]
CHECKED_NUMBERS = ["L6222-1", "L6222-18", "L6222-35", "D6222-26"]
# Partie 6 livre II (apprentissage): numéros L62xx, R62xx, D62xx.
LIVRE_II_REGEX = r"^[LRD]62\d\d(?:-|$)"


def load() -> pd.DataFrame:
    return pq.read_table(RAW_PATH, columns=COLUMNS).to_pandas()


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def counts(df: pd.DataFrame) -> None:
    section("Comptages")
    print(f"chunks: {len(df)}")
    print(f"articles (doc_id uniques): {df.doc_id.nunique()}")
    print(f"numéros d'articles distincts: {df.number.nunique()}")
    per_doc = df.drop_duplicates("doc_id").status.value_counts()
    per_chunk = df.status.value_counts()
    print(pd.DataFrame({"articles": per_doc, "chunks": per_chunk}).fillna(0).astype(int))
    multi = df.groupby("doc_id").size()
    print(f"articles avec >1 chunk: {(multi > 1).sum()} (max {multi.max()} chunks)")
    print(f"chunk_index min/max: {df.chunk_index.min()} / {df.chunk_index.max()}")


def versions(df: pd.DataFrame) -> None:
    section("Versions historiques")
    for num in CHECKED_NUMBERS:
        sub = df[df.number == num].drop_duplicates("doc_id").sort_values("start_date")
        print(f"\n{num}: {len(sub)} version(s)")
        if len(sub):
            print(sub[["doc_id", "status", "start_date", "end_date"]].to_string(index=False))


def sample_livre_ii(df: pd.DataFrame, n: int = 20) -> None:
    section(f"{n} articles aléatoires (partie 6 livre II, regex {LIVRE_II_REGEX})")
    arts = df.drop_duplicates("doc_id")
    sub = arts[arts.number.str.match(LIVRE_II_REGEX, na=False)]
    print(f"{len(sub)} versions d'articles au total dans cette zone; "
          f"{sub[sub.status == 'VIGUEUR'].shape[0]} en VIGUEUR")
    for _, r in sub.sample(n, random_state=42).iterrows():
        print(f"\n[{r.number}] {r.doc_id} ({r.status}, {r.start_date} -> {r.end_date})")
        print(f"  subtitles: {r.subtitles}")
        print(f"  text: {r.text[:120].replace(chr(10), ' ')}...")


def main() -> None:
    df = load()
    counts(df)
    versions(df)
    sample_livre_ii(df)


if __name__ == "__main__":
    main()
