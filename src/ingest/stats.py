"""Corpus statistics for the data card (BGE-M3 tokens, breakdown per book).

Usage: uv run python -m src.ingest.stats
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def count_tokens(texts: list[str]) -> list[int]:
    tok = Tokenizer.from_file(hf_hub_download("BAAI/bge-m3", "tokenizer.json"))
    return [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]


def main() -> None:
    arts = pd.read_parquet("data/processed/articles.parquet")
    arts["tokens"] = count_tokens(arts.text.tolist())
    arts["livre"] = arts.hierarchy.str.extract(r"(?:^|> )(Livre [^:]+: [^>]+?)(?: >|$)")[0]
    arts["partie"] = arts.hierarchy.str.split(" > ").str[:2].str.join(" > ")
    print(f"articles: {len(arts)}")
    print(arts.tokens.describe(percentiles=[.5, .9, .99]).round(1))
    print(f"total tokens: {arts.tokens.sum()}")
    print(f"articles > 512 tokens: {(arts.tokens > 512).sum()}")
    print("\nPer book:")
    g = arts.groupby(["partie", "livre"]).tokens.agg(["count", "mean"]).round(0)
    print(g.to_string())
    print(f"\nWith nota: {(arts.nota != '').sum()}")
    print(f"Législative: {arts.hierarchy.str.startswith('Partie législative').sum()}, "
          f"réglementaire: {arts.hierarchy.str.startswith('Partie réglementaire').sum()}")


if __name__ == "__main__":
    main()
