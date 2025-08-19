"""Incremental OHLCV dataset updater."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import paths
from .ccxt_loader import fetch_ohlcv

_EXCHANGE = "binance"


def _manifest_path(symbol: str, timeframe: str) -> Path:
    return paths.raw_parquet_path(_EXCHANGE, symbol, timeframe).with_suffix(".manifest.json")


def last_watermark(symbol: str, timeframe: str) -> int | None:
    """Return last persisted timestamp in ms for symbol/timeframe."""
    mpath = _manifest_path(symbol, timeframe)
    if mpath.exists():
        with open(mpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("watermark")
    return None


def fetch_ohlcv_incremental(
    exchange: Any, symbol: str, timeframe: str, since_ms: int | None = None
) -> pd.DataFrame:
    """Fetch OHLCV rows newer than ``since_ms``."""
    dfs: list[pd.DataFrame] = []
    since = since_ms
    while True:
        df = fetch_ohlcv(exchange, symbol, timeframe, since=since)
        if df.empty:
            break
        dfs.append(df)
        last_ts = int(df["ts"].iloc[-1])
        if since is not None and last_ts <= since:
            break
        if len(df) < 1000:
            break
        since = last_ts + 1
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])


def upsert_parquet(df: pd.DataFrame, path: Path) -> None:
    """Merge *df* into Parquet *path* without duplicates."""
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(paths.posix(tmp), index=False)
    tmp.replace(path)
