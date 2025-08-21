from __future__ import annotations

"""Helpers to ensure OHLCV data availability."""

from datetime import datetime, timedelta, UTC
import time
from pathlib import Path
import logging

import pandas as pd

from ..utils.paths import raw_parquet_path


logger = logging.getLogger(__name__)


def ensure_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    *,
    hours: int = 24,
) -> Path:
    """Ensure a parquet file with OHLCV data exists and return its path.

    If the file is missing the function attempts to download the minimal
    required range using ``ccxt``.  A simple exponential backoff handles
    rate limits.  When downloading fails a ``RuntimeError`` is raised.
    """

    path = raw_parquet_path(exchange, symbol, timeframe)
    logger.info("ensure_ohlcv path=%s", path)
    if path.exists():
        logger.info("existing file found, skipping download")
        return path

    try:  # pragma: no cover - optional dependency
        import ccxt  # type: ignore
    except Exception as exc:  # pragma: no cover - missing optional dep
        raise RuntimeError("ccxt is required to download data") from exc

    ex_class = getattr(ccxt, exchange)
    ex = ex_class({"enableRateLimit": True})
    logger.info(
        "downloading OHLCV exchange=%s symbol=%s timeframe=%s hours=%d",
        exchange,
        symbol,
        timeframe,
        hours,
    )

    since = int((datetime.now(UTC) - timedelta(hours=hours)).timestamp() * 1000)
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    now_ms = int(datetime.now(UTC).timestamp() * 1000)

    rows: list[list[float]] = []
    while since < now_ms:
        for attempt in range(5):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
                logger.info(
                    "fetched batch rows=%d since=%d attempt=%d",
                    len(batch),
                    since,
                    attempt + 1,
                )
                break
            except ccxt.RateLimitExceeded:  # pragma: no cover - network
                logger.warning("rate limit exceeded, retrying...", exc_info=True)
                time.sleep(2 ** attempt)
        else:  # pragma: no cover - network
            logger.error("rate limit exceeded fetching OHLCV")
            raise RuntimeError("rate limit exceeded fetching OHLCV")

        if not batch:
            logger.warning("empty batch received, stopping download")
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms

    if not rows:
        logger.error("no OHLCV data downloaded for %s %s", exchange, symbol)
        raise RuntimeError("no OHLCV data downloaded")
    logger.info("download complete rows=%d", len(rows))

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["source"] = "ccxt"

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("saving parquet to %s", path)
    df.to_parquet(path)
    return path
