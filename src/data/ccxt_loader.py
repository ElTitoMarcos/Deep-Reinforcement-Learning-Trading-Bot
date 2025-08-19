"""CCXT data helpers.

Currently only provides a tiny utility used in the tests to generate
synthetic high frequency data from minute bars.  The function is kept
lightweight to avoid heavy dependencies during testing.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import pandas as pd

__all__ = [
    "simulate_1s_from_1m",
    "get_exchange",
    "fetch_ohlcv",
    "save_history",
]


def simulate_1s_from_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Expand 1 minute OHLCV data into 1 second bars.

    The implementation performs a simple linear interpolation between the
    open and close of each minute bar and divides the volume evenly.  It is
    intended purely for testing and does not attempt to model real market
    micro-structure.
    """

    rows = []
    for _, row in df_1m.iterrows():
        start_ts = int(row["ts"])
        open_ = float(row["open"])
        close = float(row["close"])
        volume = float(row.get("volume", 0.0)) / 60.0
        for i in range(60):
            ts = start_ts + i * 1000
            # simple linear interpolation for the price
            if 59 > 0:
                frac = i / 59.0
            else:
                frac = 0.0
            price = open_ + (close - open_) * frac
            rows.append(
                {
                    "ts": ts,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                }
            )

    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def get_exchange(name: str | None = None, *, use_testnet: bool | None = None):
    """Return a ``ccxt`` Binance client.

    ``name`` is kept for backwards compatibility and must be ``\"binance\"`` if
    provided.  Testnet mode is enabled when ``use_testnet`` is ``True`` or when
    the ``BINANCE_USE_TESTNET`` environment variable is truthy.
    """

    try:  # pragma: no cover - exercised only when ccxt is available
        import ccxt  # type: ignore
    except Exception as exc:  # pragma: no cover - missing optional dep
        raise ImportError("ccxt is required to create exchange clients") from exc

    if name and name.lower() != "binance":
        raise ValueError("only binance exchange is supported")

    if use_testnet is None:
        use_testnet = os.getenv("BINANCE_USE_TESTNET", "").lower() in ("1", "true", "yes")

    ex = ccxt.binance({"enableRateLimit": True})
    if use_testnet:  # pragma: no cover - network/ccxt quirks
        try:
            ex.set_sandbox_mode(True)
        except Exception:
            ex.urls["api"] = ex.urls.get("test", ex.urls.get("api"))

    try:  # pragma: no cover - network/ccxt quirks
        ex.load_markets()
    except Exception:
        pass
    return ex


def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: str = "1m",
    since: Optional[int] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV data from *exchange* and return a DataFrame."""

    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    return pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])


def save_history(
    df: pd.DataFrame, root_dir: str, exchange: str, symbol: str, timeframe: str
) -> str:
    """Persist OHLCV *df* as CSV and return the output path."""

    os.makedirs(root_dir, exist_ok=True)
    fname = f"{exchange}_{symbol.replace('/', '-')}_{timeframe}.csv"
    path = os.path.join(root_dir, fname)
    df.to_csv(path, index=False)
    return path

