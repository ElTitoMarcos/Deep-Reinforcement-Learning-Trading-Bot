from __future__ import annotations

import logging
import time
from typing import Iterable, List, Tuple

import pandas as pd

from .ccxt_loader import get_exchange, fetch_ohlcv

logger = logging.getLogger(__name__)


__all__ = ["find_high_activity_windows"]


def _hour_floor(ts: int) -> int:
    return (ts // 3600000) * 3600000


def find_high_activity_windows(
    symbols: Iterable[str],
    timeframe_min: int,
    lookback_years: int = 5,
    target_hours: int = 2000,
) -> List[Tuple[int, int]]:
    """Return high activity windows for *symbols*.

    The function performs a lightweight scan over recent historical data,
    scoring each hour by the standard deviation of returns plus the traded
    volume.  The top ``target_hours`` are selected and merged into
    contiguous windows.  The chosen windows and total hours are logged.
    """

    exchange = get_exchange()
    tf_str = f"{timeframe_min}m"
    end = int(time.time() * 1000)
    lookback_ms = lookback_years * 365 * 24 * 3600 * 1000
    since = end - lookback_ms

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        try:
            df = fetch_ohlcv(exchange, sym, timeframe=tf_str, since=since)
            df["symbol"] = sym
            frames.append(df)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("ohlcv_fetch_failed symbol=%s error=%s", sym, exc)

    if not frames:
        logger.warning("no_data_for_volatility_scan")
        return []

    df_all = pd.concat(frames)
    df_all.sort_values("ts", inplace=True)
    df_all["ret"] = df_all.groupby("symbol")["close"].pct_change().fillna(0.0)
    df_all["hour"] = df_all["ts"].apply(_hour_floor)

    grouped = df_all.groupby("hour").agg({"ret": "std", "volume": "sum"})
    grouped["score"] = grouped["ret"].fillna(0) + grouped["volume"].fillna(0)
    top = grouped.sort_values("score", ascending=False).head(target_hours)

    hours = sorted(top.index)
    windows: List[Tuple[int, int]] = []
    start = prev = None
    for h in hours:
        if start is None:
            start = prev = int(h)
        elif int(h) == prev + 3600000:
            prev = int(h)
        else:
            windows.append((int(start), int(prev + 3600000)))
            start = prev = int(h)
    if start is not None:
        windows.append((int(start), int(prev + 3600000)))

    total_hours = sum((e - s) // 3600000 for s, e in windows)
    logger.info("selected_vol_windows=%s total_hours=%s", windows, total_hours)
    return windows
