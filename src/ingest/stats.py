"""Statistiques du corpus pour la data card (tokens BGE-M3, répartition par livre).

Usage: uv run python -m src.ingest.stats
"""
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
    print("\nPar livre:")
    g = arts.groupby(["partie", "livre"]).tokens.agg(["count", "mean"]).round(0)
    print(g.to_string())
    print(f"\nAvec nota: {(arts.nota != '').sum()}")
    print(f"Partie législative: {arts.hierarchy.str.startswith('Partie législative').sum()}, "
          f"réglementaire: {arts.hierarchy.str.startswith('Partie réglementaire').sum()}")


if __name__ == "__main__":
    main()
