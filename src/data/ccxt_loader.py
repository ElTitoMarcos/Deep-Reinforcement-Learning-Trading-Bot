"""CCXT data helpers.

Currently only provides a tiny utility used in the tests to generate
synthetic high frequency data from minute bars.  The function is kept
lightweight to avoid heavy dependencies during testing.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional
from pathlib import Path

import pandas as pd

from ..utils import paths
from ..utils.credentials import load_binance_creds, compute_rate_limit_ms, mask

__all__ = [
    "simulate_1s_from_1m",
    "get_exchange",
    "fetch_ohlcv",
    "save_history",
]


logger = logging.getLogger(__name__)


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

    key, sec, env_testnet = load_binance_creds()
    use_testnet = env_testnet if use_testnet is None else use_testnet

    ex = ccxt.binance({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.rateLimit = compute_rate_limit_ms()

    if use_testnet:
        ex.urls["api"]["public"] = "https://testnet.binance.vision/api"
        ex.urls["api"]["private"] = "https://testnet.binance.vision/api"

    logger.info(
        "[binance] Credenciales cargadas (key=%s), testnet=%s, rateLimit=%sms",
        mask(key),
        use_testnet,
        ex.rateLimit,
    )

    try:  # pragma: no cover - network/ccxt quirks
        ex.load_markets()
    except Exception:
        pass
    return ex


def fetch_ohlcv(
    exchange: Any,
    symbol: str,
    timeframe: Optional[str] = None,
    since: Optional[int] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV data from *exchange* and return a DataFrame.

    When *timeframe* is ``None`` the function attempts to pick the smallest
    available timeframe from ``exchange.timeframes`` following the order
    ``["1s","3s","5s","15s","30s","1m","3m","5m"]``.  The chosen
    timeframe is stored in ``df.attrs['timeframe']`` and logged via the module
    logger.
    """

    tf = timeframe
    if tf is None:
        candidates = ["1s", "3s", "5s", "15s", "30s", "1m", "3m", "5m"]
        available = getattr(exchange, "timeframes", {}) or {}
        for cand in candidates:
            if cand in available:
                tf = cand
                break
        if tf is None:  # pragma: no cover - unlikely for Binance
            raise ValueError("no supported timeframe found")
        logger.info("selected_timeframe=%s", tf)

    data = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=limit)
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
    df.attrs["timeframe"] = tf
    return df


def save_history(
    df: pd.DataFrame, root_dir: os.PathLike[str] | str, exchange: str, symbol: str, timeframe: str
) -> str:
    """Persist OHLCV *df* as CSV and return the output path."""

    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    fname = f"{exchange}_{paths.symbol_to_dir(symbol)}_{timeframe}.csv"
    out = root / fname
    df.to_csv(paths.posix(out), index=False)
    return paths.posix(out)

