"""Retrieval: BM25 + dense (+ DINUM baseline) -> RRF fusion -> reranker.

Every stage returns a ranking: a list of (row_index, score), best first.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import unicodedata

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ARTICLES_PATH = "data/processed/articles.parquet"
BASELINE_PATH = "data/processed/dinum_baseline_chunks.parquet"
CACHE_DIR = Path("data/index")
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
POOL = 30                # candidates kept per stage before fusion / reranking
RRF_K = 60               # standard RRF constant

NUM_BODY = r"\d{4}(?:-\d+)?"
NUMBER = rf"[lrd]{NUM_BODY}"                  # l6222-35, d6222-26, r6227-7
NUMBER_RAW = rf"\b([lrd])(?!')\s*[-.]?\s*({NUM_BODY})"  # L. 6222-35, L 6222-35, L6222-35


@dataclass
class Hit:
    doc_id: str
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


# ============ 1. data ============
@lru_cache(maxsize=1)
def load_articles():
    df = pd.read_parquet(ARTICLES_PATH).reset_index(drop=True)
    return df, df["chunk_text"].tolist()


def make_mask(df, filters):
    """True = row allowed. Only the filters we support; unknown keys are ignored."""
    mask = np.ones(len(df), dtype=bool)
    if filters and "etat" in filters:
        mask &= (df["etat"] == filters["etat"]).to_numpy()
    if filters and "hierarchy_contains" in filters:
        mask &= df["hierarchy"].str.contains(filters["hierarchy_contains"], case=False, regex=False).to_numpy()
    return mask


def top_n(scores, mask, n):
    """Indices of the n best scores among allowed rows -> [(index, score), ...]."""
    scores = np.where(mask, scores, -np.inf)
    return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:n] if np.isfinite(scores[i])]


# ============ 2. BM25 ============
def clean(text):
    """lowercase, remove accents, write 'L. 6222-35' as 'l6222-35'."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(NUMBER_RAW, r"\1\2", text)


def tokenize(text):
    return re.findall(NUMBER + r"|\w+", clean(text))   # article number stays ONE token


@lru_cache(maxsize=1)
def get_bm25():
    _, texts = load_articles()
    return BM25Okapi([tokenize(t) for t in texts])


def bm25_ranking(query, mask, n=POOL):
    scores = np.asarray(get_bm25().get_scores(tokenize(query)), dtype=float)
    return [(i, s) for i, s in top_n(scores, mask, n) if s > 0]   # drop zero-score noise


# ============ 3. dense (our embeddings) ============
@lru_cache(maxsize=1)
def get_embedder():
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 2048          # the longest articles are ~1000 tokens
    return model


def embed_query(query):
    return get_embedder().encode([query], normalize_embeddings=True)[0]


@lru_cache(maxsize=1)
def get_article_embeddings():
    """Embed every article once, cache on disk (key = model + texts, so edits invalidate it)."""
    _, texts = load_articles()
    key = hashlib.md5((EMBED_MODEL + "\n".join(texts)).encode()).hexdigest()[:10]
    path = CACHE_DIR / f"emb_{key}.npy"
    if path.exists():
        return np.load(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    emb = get_embedder().encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    np.save(path, emb)
    return emb


def dense_ranking(query, mask, n=POOL):
    scores = get_article_embeddings() @ embed_query(query)
    return top_n(scores, mask, n)


# ============ 4. dense (DINUM baseline embeddings) ============
@lru_cache(maxsize=1)
def get_baseline():
    """DINUM chunks + their stored bge-m3 embeddings; rows map to articles via doc_id."""
    df, _ = load_articles()
    b = pd.read_parquet(BASELINE_PATH)
    emb = np.asarray(b["embedding"].tolist(), dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    row_of_doc = {d: i for i, d in enumerate(df["doc_id"])}
    return emb, b["doc_id"].map(row_of_doc).to_numpy()   # chunk -> article row index


def baseline_ranking(query, mask, n=POOL):
    emb, chunk_row = get_baseline()
    chunk_scores = emb @ embed_query(query)
    # several chunks can belong to one article: keep the best chunk per article
    article_scores = np.full(len(mask), -np.inf)
    np.maximum.at(article_scores, chunk_row, chunk_scores)
    return top_n(article_scores, mask, n)


# ============ 5. fusion + reranker ============
def rrf(rankings, k=RRF_K):
    """Reciprocal Rank Fusion: score(doc) = sum over rankings of 1 / (k + rank)."""
    fused = {}
    for ranking in rankings:
        for rank, (i, _) in enumerate(ranking, start=1):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANK_MODEL, max_length=2048)


def rerank(query, ranking):
    if not ranking:
        return []
    _, texts = load_articles()
    scores = get_reranker().predict([(query, texts[i]) for i, _ in ranking])
    return sorted(((i, float(s)) for (i, _), s in zip(ranking, scores)), key=lambda x: -x[1])


# ============ 6. exact article number ============
def exact_number_first(query, df, mask, ranking):
    """If the question names an article (e.g. L6222-35), put that article first."""
    wanted = set(re.findall(NUMBER, clean(query)))
    if not wanted:
        return ranking
    exact = [int(i) for i in np.flatnonzero(df["num"].str.lower().isin(wanted).to_numpy() & mask)]
    top_score = ranking[0][1] if ranking else 0.0
    return [(i, top_score + 1.0) for i in exact] + [(i, s) for i, s in ranking if i not in exact]


# ============ 7. the public function ============
def search(
    query: str,
    k: int = 10,
    filters: dict | None = None,
    mode: str = "hybrid_rerank",
    exact_match: bool = True,
) -> list[Hit]:
    """mode: bm25 | dense | baseline | hybrid | hybrid_rerank"""
    df, _ = load_articles()
    mask = make_mask(df, filters)

    if mode == "bm25":
        ranking = bm25_ranking(query, mask)
    elif mode == "dense":
        ranking = dense_ranking(query, mask)
    elif mode == "baseline":
        ranking = baseline_ranking(query, mask)
    elif mode in ("hybrid", "hybrid_rerank"):
        ranking = rrf([bm25_ranking(query, mask), dense_ranking(query, mask)])[:POOL]
        if mode == "hybrid_rerank":
            ranking = rerank(query, ranking)
    else:
        raise ValueError(f"unknown mode: {mode}")

    if exact_match:
        ranking = exact_number_first(query, df, mask, ranking)

    hits = []
    for i, score in ranking[:k]:
        row = df.iloc[i]
        hits.append(Hit(
            doc_id=row["doc_id"],
            chunk_id=row["doc_id"],   # 1 article = 1 chunk
            score=score,
            text=row["text"],
            metadata={
                "num": row["num"],
                "hierarchy": row["hierarchy"],
                "etat": row["etat"],
                "date_debut": str(row["date_debut"]),
                "date_fin": str(row["date_fin"]),
                "url": row["url"],
                "nota": row["nota"],
            },
        ))
    return hits


def main() -> None:
    print("1. tokenizer keeps the article number as ONE word")
    print("  ", tokenize("Que dit l'article L. 6222-35 ?"))
    # expected: [..., 'article', 'l6222-35']

    print("\n2. article number query -> that article must be first")
    for q in ["L6222-35", "l. 6222-35", "article D6222-26"]:
        hits = search(q, k=3)
        print(f"  {q!r:25} ->", [h.metadata["num"] for h in hits])
    # expected: first number in each list matches the query

    print("\n3. natural language query -> read the results yourself")
    for h in search("durée maximale du contrat d'apprentissage", k=5):
        print(f"   {h.metadata['num']:<10} {h.score:5.1f}  {h.text[:80]}")
    # expected: plausible articles about contract duration (judge by eye)

    print("\n4. nothing should come back for nonsense or an unknown number for bm25")
    print("  ", search("zzzxqv", k=5, mode="bm25"))       # expected: []
    print("  ", search("L9999-99", k=5, mode="bm25"))     # expected: []

    print("\n5. k is respected and scores are sorted")
    hits = search("apprenti", k=4)
    scores = [h.score for h in hits]
    print("  ", len(hits), "hits, sorted:", scores == sorted(scores, reverse=True))
    # expected: 4 hits, sorted: True


if __name__ == "__main__":
    main()
