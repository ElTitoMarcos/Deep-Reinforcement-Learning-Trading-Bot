from __future__ import annotations
import os
from typing import Optional
import pandas as pd

REQUIRED_OHLCV_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "exchange",
    "symbol",
    "timeframe",
    "source",
]

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


# ---------------------------------------------------------------------------
# universe helpers

UNIVERSE_COLUMNS = [
    "symbol",
    "quote",
    "base",
    "avg_volume_24h",
    "first_seen",
    "exchange",
]


def save_universe(df: pd.DataFrame, path: str) -> str:
    """Persist a universe table in CSV or Parquet format."""

    df = df[UNIVERSE_COLUMNS].copy()
    save_table(df, path)
    return path


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Basic validation and UTC normalization for OHLCV tables."""
    missing = set(REQUIRED_OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, unit="ms")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    # store timestamps as integer milliseconds
    df["ts"] = (df["ts"].astype("int64") // 1_000_000).astype("int64")
    return df


def save_ohlcv(df: pd.DataFrame, root: str, exchange: str, symbol: str, timeframe: str) -> str:
    df = validate_ohlcv(df)
    sym_fs = symbol.replace("/", "_")
    path = os.path.join(root, exchange, sym_fs, f"{timeframe}.parquet")
    save_table(df, path)
    return path
