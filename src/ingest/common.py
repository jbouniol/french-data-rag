"""Chargement du parquet brut du Code du travail."""
import pandas as pd
import pyarrow.parquet as pq

RAW_PATH = "data/raw/data/legi-latest/legi_code_du_travail/legi_code_du_travail_part_0.parquet"


def load_chunks(columns: list[str]) -> pd.DataFrame:
    """Lit uniquement les colonnes demandées (les embeddings sont lourds)."""
    return pq.read_table(RAW_PATH, columns=columns).to_pandas()
