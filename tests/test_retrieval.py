import numpy as np
import pandas as pd

from src.retrieval import search as S

df, texts = S.load_articles()
b = pd.read_parquet(S.BASELINE_PATH)

# ---------- 1. structure: does every baseline chunk map to a corpus article? ----------
print("1. STRUCTURE")
print("   chunks:", len(b), "| articles:", len(df))
print("   chunks with unknown doc_id:", (~b.doc_id.isin(df.doc_id)).sum(), "   (expected 0)")
print("   articles with no baseline chunk:", (~df.doc_id.isin(b.doc_id)).sum(), "   (expected 0)")
emb, chunk_row = S.get_baseline()
print("   embedding matrix:", emb.shape, "  (expected (n_chunks, 1024))")
print("   NaN in mapping:", bool(np.isnan(chunk_row.astype(float)).any()), "   (expected False)")

# ---------- 2. are the stored embeddings really bge-m3 of chunk_text? ----------
print("\n2. COSINE: stored embedding vs our bge-m3 on the same chunk_text")
sample = b.sample(10, random_state=0)
ours = S.get_embedder().encode(sample.chunk_text.tolist(), normalize_embeddings=True)
stored = np.asarray(sample.embedding.tolist(), dtype=np.float32)
stored /= np.linalg.norm(stored, axis=1, keepdims=True)
cos = (ours * stored).sum(axis=1)
print("   min / mean:", round(float(cos.min()), 4), "/", round(float(cos.mean()), 4))
print("   -> >= 0.99 : baseline is valid, same model and same text")
print("   -> 0.90-0.99: same model, but they embedded slightly different text; say so in the README")
print("   -> < 0.90  : not comparable, re-embed their chunk_text yourself")

# ---------- 3. does a baseline chunk retrieve its own article? ----------
print("\n3. SELF-RETRIEVAL: query = first sentence of an article, expect that article in the top 3")
rng = np.random.default_rng(0)
hits = 0
picks = rng.choice(len(df), 20, replace=False)
mask = S.make_mask(df, None)
for i in picks:
    query = df.text[i].split(".")[0][:200]
    top = [j for j, _ in S.baseline_ranking(query, mask, n=3)]
    hits += i in top
print(f"   {hits}/20 found in the top 3   (expected: most of them, say >= 16)")

# ---------- 4. baseline vs our dense on the same queries ----------
print("\n4. AGREEMENT: do baseline and our dense retrieval return the same top article?")
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

# ---------- 5. test results on all rankings ----------
def show(ranking):
    for i, s in ranking[:5]:
        print(f"{df.num[i]:<10}{s:8.3f}  {df.text[i][:70]}")

mask = S.make_mask(df, None)
q = "durée maximale du contrat d'apprentissage"

show(S.bm25_ranking(q, mask))        # 1. keywords
show(S.dense_ranking(q, mask))       # 2. our embeddings
show(S.baseline_ranking(q, mask))    # 3. DINUM embeddings
fused = S.rrf([S.bm25_ranking(q, mask), S.dense_ranking(q, mask)])
show(fused)                          # 4. fusion
show(S.rerank(q, fused[:30]))        # 5. reranker
print([h.metadata["num"] for h in S.search("L. 6222-35", k=3)])   # 6. L6222-35 first