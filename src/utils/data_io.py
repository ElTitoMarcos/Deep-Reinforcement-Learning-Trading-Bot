from __future__ import annotations

import os
import numpy as np
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

def ensure_dir(path: str) -> None:
    """Create ``path`` if it does not already exist."""
    os.makedirs(path, exist_ok=True)

def save_table(df: pd.DataFrame, path: str) -> None:
    """Save ``df`` to ``path`` inferring the format from the extension."""
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


def load_universe(path: str) -> pd.DataFrame:
    """Load a previously saved universe table ensuring column order."""
    df = load_table(path)
    missing = set(UNIVERSE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df[UNIVERSE_COLUMNS].copy()


# ---------------------------------------------------------------------------
# OHLCV utilities


def resample_to(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample *df* to a canonical OHLCV table for ``timeframe``.

    Parameters
    ----------
    df: pd.DataFrame
        Table with at least ``ts`` (ms) and OHLCV columns.
    timeframe: str
        ``"1s"`` or ``"1m"``.
    """

    if timeframe not in {"1s", "1m"}:
        raise ValueError("timeframe must be '1s' or '1m'")

    df = df.copy()
    dt_index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index(dt_index, inplace=True)

    rule = "1S" if timeframe == "1s" else "1T"
    ohlcv = df.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    ohlcv.reset_index(inplace=True)
    ohlcv.rename(columns={"index": "ts"}, inplace=True)
    ohlcv["ts"] = (ohlcv["ts"].astype("int64") // 1_000_000).astype("int64")
    return ohlcv[["ts", "open", "high", "low", "close", "volume"]]


def fill_small_gaps(df: pd.DataFrame, max_ticks: int = 3) -> tuple[pd.DataFrame, int]:
    """Fill gaps of up to ``max_ticks`` in *df*.

    Returns the filled DataFrame and the number of ticks inserted."""

    df = df.copy()
    dt_index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.set_index(dt_index, inplace=True)

    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        df.reset_index(inplace=True)
        df.rename(columns={"index": "ts"}, inplace=True)
        df["ts"] = (df["ts"].astype("int64") // 1_000_000).astype("int64")
        return df[["ts", "open", "high", "low", "close", "volume"]], 0
    freq = diffs.mode().iloc[0]

    full_index = pd.date_range(df.index[0], df.index[-1], freq=freq)
    df = df.reindex(full_index)

    mask = df["open"].isna()
    filled = 0
    i = 0
    while i < len(df):
        if mask.iloc[i]:
            j = i
            while j < len(df) and mask.iloc[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_ticks and i > 0:
                prev_close = df["close"].iloc[i - 1]
                df.iloc[i:j, [df.columns.get_loc(c) for c in ["open", "high", "low", "close"]]] = prev_close
                df.iloc[i:j, df.columns.get_loc("volume")] = 0.0
                filled += gap_len
            i = j
        else:
            i += 1

    df.reset_index(inplace=True)
    df.rename(columns={"index": "ts"}, inplace=True)
    df["ts"] = (df["ts"].astype("int64") // 1_000_000).astype("int64")
    return df[["ts", "open", "high", "low", "close", "volume"]], filled


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV data and ensure canonical ordering."""

    missing = set(REQUIRED_OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, unit="ms")
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df[REQUIRED_OHLCV_COLUMNS].isna().any().any():
        raise ValueError("NaN values found in OHLCV table")

    if df["close"].std(ddof=0) > 0:
        z = np.abs((df["close"] - df["close"].mean()) / df["close"].std(ddof=0))
        if (z > 6).any():
            raise ValueError("outlier detected in close prices")

    tf = df["timeframe"].iloc[0]
    try:
        tf_ms = int(pd.to_timedelta(tf).total_seconds() * 1000)
    except Exception as e:  # pragma: no cover - invalid timeframe
        raise ValueError(f"invalid timeframe: {tf}") from e

    diffs = df["ts"].diff().dropna()
    if not diffs.empty and not (diffs == tf_ms).all():
        raise ValueError("time gaps detected in OHLCV table")

    df["ts"] = (df["ts"].astype("int64") // 1_000_000).astype("int64")
    return df


def save_ohlcv(df: pd.DataFrame, root: str, exchange: str, symbol: str, timeframe: str) -> str:
    df = validate_ohlcv(df)
    sym_fs = symbol.replace("/", "_")
    path = os.path.join(root, exchange, sym_fs, f"{timeframe}.parquet")
    save_table(df, path)
    return path
