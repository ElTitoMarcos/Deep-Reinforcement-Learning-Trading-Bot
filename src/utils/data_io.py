from __future__ import annotations
import os
from typing import Optional
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_table(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
