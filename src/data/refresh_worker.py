from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import Iterable

from .ccxt_loader import get_exchange
from .incremental import fetch_ohlcv_incremental, last_watermark, upsert_parquet
from ..utils import paths

# Event set when new data appended by the worker
# Consumers should clear after handling

dataset_updated = threading.Event()
_stop_event = threading.Event()
_thread: threading.Thread | None = None


def _parse_interval(timeframe_min: str) -> float:
    """Return minutes represented by timeframe string like ``"1m"``."""
    try:
        return float(timeframe_min.rstrip("m"))
    except Exception:
        return 1.0


def _run(symbols: Iterable[str], timeframe: str, interval_min: float) -> None:
    ex = get_exchange()
    while not _stop_event.is_set():
        updated = False
        for sym in symbols:
            try:
                since = last_watermark(sym, timeframe)
                if since is None:
                    since = int((datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000)
                df_new = fetch_ohlcv_incremental(ex, sym, timeframe, since_ms=since)
                if df_new.empty:
                    continue
                path = paths.raw_parquet_path(
                    ex.id if hasattr(ex, "id") else "binance", sym, timeframe
                )
                upsert_parquet(df_new, path)
                manifest = {
                    "symbol": sym,
                    "timeframe": timeframe,
                    "watermark": int(df_new["ts"].max()),
                    "obtained_at": datetime.now(UTC).isoformat(),
                }
                with open(path.with_suffix(".manifest.json"), "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                print(f"Dataset actualizado +{len(df_new)} velas")
                updated = True
            except Exception:
                continue
        if updated:
            dataset_updated.set()
        # wait but allow stop_event to break early
        _stop_event.wait(interval_min * 60)


def start_refresh_worker(
    symbols: Iterable[str], timeframe_min: str, every: float | None = None
) -> None:
    """Start background thread to periodically refresh OHLCV data."""
    global _thread
    if _thread and _thread.is_alive():
        return
    interval = every if every is not None else _parse_interval(timeframe_min)
    dataset_updated.clear()
    _stop_event.clear()
    _thread = threading.Thread(
        target=_run, args=(list(symbols), timeframe_min, interval), daemon=True
    )
    _thread.start()


def stop_refresh_worker() -> None:
    """Stop the background refresh thread."""
    _stop_event.set()
    if _thread:
        _thread.join()
