"""Helpers to select high-activity data windows."""

from __future__ import annotations

import logging
import time
from typing import Iterable, List, Tuple, Optional

import pandas as pd

from .ccxt_loader import get_exchange
from .ensure import ensure_ohlcv

logger = logging.getLogger(__name__)

__all__ = ["find_high_activity_windows"]


def _hour_floor(ts: int) -> int:
    return (ts // 3600000) * 3600000


def _fetch_trade_counts(exchange, symbol: str, since: int, end: int) -> pd.DataFrame:
    """Fetch trades and return hourly counts.

    Network errors are ignored and result in an empty DataFrame so callers can
    proceed using only OHLCV-based signals.
    """

    try:  # pragma: no cover - network dependent
        trades: list[dict] = []
        cursor = since
        while cursor < end:
            batch = exchange.fetch_trades(symbol, since=cursor, limit=1000)
            if not batch:
                break
            trades.extend(batch)
            cursor = batch[-1]["timestamp"] + 1
            if len(trades) >= 10000:  # avoid huge downloads
                break
        if not trades:
            return pd.DataFrame(columns=["hour", "n_trades"])
        df = pd.DataFrame(trades)
        df["hour"] = df["timestamp"].apply(_hour_floor)
        return df.groupby("hour").size().reset_index(name="n_trades")
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("trade_fetch_failed symbol=%s error=%s", symbol, exc)
        return pd.DataFrame(columns=["hour", "n_trades"])


def find_high_activity_windows(
    symbols: Iterable[str],
    timeframe_min: int,
    *,
    target_hours: int = 24,
    lookback_hours: Optional[int] = None,
    exchange: Optional[object] = None,
) -> Tuple[List[Tuple[int, int]], int]:
    """Return high activity windows and the lookback span used.

    Each hour is scored by ``std(returns) + volume + number_of_trades``.  The
    ``target_hours`` top-scoring hours are merged into contiguous windows.  If
    ``lookback_hours`` is omitted it defaults to six times ``target_hours`` to
    provide a pool of candidates.
    """

    ex = exchange or get_exchange()
    tf_str = f"{timeframe_min}m"
    if lookback_hours is None:
        lookback_hours = target_hours * 6
    end = int(time.time() * 1000)
    since = end - lookback_hours * 3600000

    frames: List[pd.DataFrame] = []
    trade_frames: List[pd.DataFrame] = []
    for sym in symbols:
        try:
            path = ensure_ohlcv(
                ex.id if hasattr(ex, "id") else "binance", sym, tf_str, hours=lookback_hours
            )
            df = pd.read_parquet(path)
            df = df[df["ts"] >= since]
            df["symbol"] = sym
            frames.append(df)

            trade_counts = _fetch_trade_counts(ex, sym, since, end)
            if not trade_counts.empty:
                trade_frames.append(trade_counts)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("ohlcv_fetch_failed symbol=%s error=%s", sym, exc)

    if not frames:
        logger.warning("no_data_for_volatility_scan")
        return [], lookback_hours

    df_all = pd.concat(frames)
    df_all.sort_values("ts", inplace=True)
    df_all["ret"] = df_all.groupby("symbol")["close"].pct_change().fillna(0.0)
    df_all["hour"] = df_all["ts"].apply(_hour_floor)

    grouped = df_all.groupby("hour").agg(ret=("ret", "std"), volume=("volume", "sum"))
    if trade_frames:
        trades_all = pd.concat(trade_frames)
        trades_group = trades_all.groupby("hour")["n_trades"].sum()
        grouped = grouped.join(trades_group, how="left")
    grouped["n_trades"] = grouped.get(
        "n_trades", pd.Series(0, index=grouped.index)
    ).fillna(0)

    grouped["score"] = (
        grouped["ret"].fillna(0)
        + grouped["volume"].fillna(0)
        + grouped["n_trades"].fillna(0)
    )

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
    logger.info(
        "selected_vol_windows=%s total_hours=%s lookback_hours=%s", windows, total_hours, lookback_hours
    )
    return windows, lookback_hours

