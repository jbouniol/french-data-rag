"""Sanity checks and unit tests for the retrieval pipeline."""
import numpy as np
import pandas as pd

from src.retrieval import search as S


def test_baseline_structure():
    df, _ = S.load_articles()
    b = pd.read_parquet(S.BASELINE_PATH)
    emb, chunk_row = S.get_baseline()

    assert len(b) == 542
    assert len(df) == 540
    assert (~b.doc_id.isin(df.doc_id)).sum() == 0
    assert (~df.doc_id.isin(b.doc_id)).sum() == 0
    assert emb.shape == (len(b), 1024)
    assert not np.isnan(chunk_row.astype(float)).any()


def test_baseline_cosine_consistency():
    b = pd.read_parquet(S.BASELINE_PATH)
    sample = b.sample(10, random_state=0)
    ours = S.get_embedder().encode(sample.chunk_text.tolist(), normalize_embeddings=True)
    stored = np.asarray(sample.embedding.tolist(), dtype=np.float32)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True)
    cos = (ours * stored).sum(axis=1)
    assert float(cos.min()) >= 0.99


def test_self_retrieval():
    df, _ = S.load_articles()
    rng = np.random.default_rng(0)
    picks = rng.choice(len(df), 20, replace=False)
    mask = S.make_mask(df, None)

    baseline_hits = 0
    dense_hits = 0
    for i in picks:
        query = df.text[i].split(".")[0][:200]
        baseline_hits += i in [j for j, _ in S.baseline_ranking(query, mask, n=3)]
        dense_hits += i in [j for j, _ in S.dense_ranking(query, mask, n=3)]

    assert baseline_hits >= 14
    assert dense_hits >= 14


def test_tokenizer_and_exact_article_number():
    assert S.tokenize("Que dit l'article L. 6222-35 ?") == ["que", "dit", "l", "article", "l6222-35"]
    for q, expected in [("L6222-35", "L6222-35"), ("l.  6222-35", "L6222-35"), ("article D6222-26", "D6222-26")]:
        hits = S.search(q, k=3, mode="bm25")
        assert hits and hits[0].metadata["num"] == expected


def test_search_modes_and_metadata():
    for mode in ("bm25", "dense", "baseline", "hybrid", "hybrid_rerank"):
        hits = S.search("durée maximale du contrat d'apprentissage", k=4, mode=mode)
        assert len(hits) == 4
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        assert {"num", "hierarchy", "etat", "date_debut", "date_fin", "url", "nota"} <= set(hits[0].metadata)

    # Empty / nonsense queries and empty filter must not crash rerank
    assert S.search("zzzxqv", k=5, mode="bm25") == []
    assert S.search("L9999-99", k=5, mode="bm25") == []
    assert S.search("apprenti", k=5, filters={"etat": "ABROGE"}, mode="hybrid_rerank") == []


def main() -> None:
    df, _ = S.load_articles()
    mask = S.make_mask(df, None)

    print("1. AGREEMENT: baseline vs our contextualized dense retrieval")
    queries = [
        "durée maximale du contrat d'apprentissage",
        "rémunération de l'apprenti",
        "rupture du contrat pendant la période d'essai",
        "travail de nuit des jeunes travailleurs",
    ]
    for q in queries:
        base = S.baseline_ranking(q, mask, n=3)
        own = S.dense_ranking(q, mask, n=3)
        print(f"   {q[:45]:<46} baseline={df.num[base[0][0]]:<10} ours={df.num[own[0][0]]}")

    q = "durée maximale du contrat d'apprentissage"
    for mode in ("bm25", "dense", "baseline", "hybrid", "hybrid_rerank"):
        print(f"\n--- mode={mode} ---")
        for h in S.search(q, k=5, mode=mode):
            print(f"   {h.metadata['num']:<10} {h.score:8.3f}  {h.text[:70]}")


if __name__ == "__main__":
    main()