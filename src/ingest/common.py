"""Loading of the raw Code du travail parquet."""
import pandas as pd
import pyarrow.parquet as pq

RAW_PATH = "data/raw/data/legi-latest/legi_code_du_travail/legi_code_du_travail_part_0.parquet"


def load_chunks(columns: list[str]) -> pd.DataFrame:
    """Read only the requested columns (embeddings are heavy)."""
    return pq.read_table(RAW_PATH, columns=columns).to_pandas()
