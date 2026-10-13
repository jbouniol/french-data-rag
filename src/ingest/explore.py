"""Read-only exploration of the Code du travail folder of AgentPublic/legi.

Usage: uv run python -m src.ingest.explore
"""
import pandas as pd

from src.ingest.common import load_chunks

# Heavy columns (embeddings, chunk_text) are left out for exploration.
COLUMNS = [
    "chunk_id", "doc_id", "chunk_index", "status", "subtitles",
    "number", "start_date", "end_date", "text",
]
CHECKED_NUMBERS = ["L6222-1", "L6222-18", "L6222-35", "D6222-26"]
# Partie 6 livre II (apprenticeship): numbers L62xx, R62xx, D62xx.
LIVRE_II_REGEX = r"^[LRD]62\d\d(?:-|$)"


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def counts(df: pd.DataFrame) -> None:
    section("Counts")
    print(f"chunks: {len(df)}")
    print(f"articles (unique doc_id): {df.doc_id.nunique()}")
    print(f"distinct article numbers: {df.number.nunique()}")
    per_doc = df.drop_duplicates("doc_id").status.value_counts()
    per_chunk = df.status.value_counts()
    print(pd.DataFrame({"articles": per_doc, "chunks": per_chunk}).fillna(0).astype(int))
    multi = df.groupby("doc_id").size()
    print(f"articles with >1 chunk: {(multi > 1).sum()} (max {multi.max()} chunks)")
    print(f"chunk_index min/max: {df.chunk_index.min()} / {df.chunk_index.max()}")


def versions(df: pd.DataFrame) -> None:
    section("Historical versions")
    for num in CHECKED_NUMBERS:
        sub = df[df.number == num].drop_duplicates("doc_id").sort_values("start_date")
        print(f"\n{num}: {len(sub)} version(s)")
        if len(sub):
            print(sub[["doc_id", "status", "start_date", "end_date"]].to_string(index=False))


def sample_livre_ii(df: pd.DataFrame, n: int = 20) -> None:
    section(f"{n} random articles (partie 6 livre II, regex {LIVRE_II_REGEX})")
    arts = df.drop_duplicates("doc_id")
    sub = arts[arts.number.str.match(LIVRE_II_REGEX, na=False)]
    print(f"{len(sub)} article versions in this area; "
          f"{sub[sub.status == 'VIGUEUR'].shape[0]} in VIGUEUR")
    for _, r in sub.sample(n, random_state=42).iterrows():
        print(f"\n[{r.number}] {r.doc_id} ({r.status}, {r.start_date} -> {r.end_date})")
        print(f"  subtitles: {r.subtitles}")
        print(f"  text: {r.text[:120].replace(chr(10), ' ')}...")


def main() -> None:
    df = load_chunks(COLUMNS)
    counts(df)
    versions(df)
    sample_livre_ii(df)


if __name__ == "__main__":
    main()
