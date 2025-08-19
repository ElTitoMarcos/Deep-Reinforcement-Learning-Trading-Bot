from __future__ import annotations

"""Helpers to ensure OHLCV data availability."""

from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.paths import raw_parquet_path


def ensure_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    *,
    hours: int = 24,
    root: Optional[Path | str] = None,
) -> Path:
    """Ensure a parquet file with OHLCV data exists and return its path.

    If the file is missing the function attempts to download the minimal
    required range using ``ccxt``.  A simple exponential backoff handles
    rate limits.  When downloading fails a ``RuntimeError`` is raised.
    """

    path = raw_parquet_path(exchange, symbol, timeframe, root)
    if path.exists():
        return path

    try:  # pragma: no cover - optional dependency
        import ccxt  # type: ignore
    except Exception as exc:  # pragma: no cover - missing optional dep
        raise RuntimeError("ccxt is required to download data") from exc

    ex_class = getattr(ccxt, exchange)
    ex = ex_class({"enableRateLimit": True})

    since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp() * 1000)
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now_ms = int(datetime.utcnow().timestamp() * 1000)

    rows: list[list[float]] = []
    while since < now_ms:
        for attempt in range(5):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
                break
            except ccxt.RateLimitExceeded:  # pragma: no cover - network
                time.sleep(2 ** attempt)
        else:  # pragma: no cover - network
            raise RuntimeError("rate limit exceeded fetching OHLCV")

        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms

    if not rows:
        raise RuntimeError("no OHLCV data downloaded")

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["source"] = "ccxt"

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path
