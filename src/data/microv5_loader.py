"""Microstructure V5 data collector.

This module provides a light‑weight implementation of the ticker/order book
collector used in the original private ``BOT_v5`` project.  It is purposely
simplified but keeps the public API requested in the project description.  The
collector periodically fetches a price tick and a shallow order book, derives a
number of features and stores them into daily parquet files.

Network access is not exercised during the tests; ``fetch_*`` helpers can be
monkey‑patched with stubbed versions.  The collector itself uses only standard
Python libraries plus ``pandas`` for the persistence layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import time

import pandas as pd

from .microv5_features import (
    derivados_basicos,
    detectar_resistencia_asks,
    umbral_entrada_dinamico,
    umbral_salida_dinamico,
)

# ---------------------------------------------------------------------------
#  Low level fetch helpers
# ---------------------------------------------------------------------------


def fetch_price_tick(ex: Any, symbol_ccxt: str) -> float:
    """Return last price using ``ccxt.fetch_ticker`` with REST fallback."""

    try:
        ticker = ex.fetch_ticker(symbol_ccxt)
        return float(ticker["last"])
    except Exception:
        # Simplified REST fallback for Binance compatible exchanges.
        import requests

        base = getattr(ex, "urls", {}).get("api", {}).get("public", "https://api.binance.com")
        r = requests.get(f"{base}/api/v3/ticker/price", params={"symbol": symbol_ccxt.replace("/", "")}, timeout=10)
        r.raise_for_status()
        return float(r.json()["price"])


def fetch_order_book(ex: Any, symbol_ccxt: str, limit: int = 20) -> Dict[str, Any]:
    """Fetch the order book using ``ccxt``."""

    return ex.fetch_order_book(symbol_ccxt, limit=limit)


def fetch_lot_size_meta(ex: Any, symbol_raw: str) -> Tuple[float, float, float]:
    """Return ``(minQty, maxQty, stepSize)`` from exchangeInfo.

    The function expects an exchange compatible with Binance.  For other
    exchanges callers may monkey‑patch this function during tests.
    """

    try:
        info = ex.public_get_exchangeinfo({"symbol": symbol_raw})
        filt = next(f for f in info["symbols"][0]["filters"] if f["filterType"] == "LOT_SIZE")
        return float(filt["minQty"]), float(filt["maxQty"]), float(filt["stepSize"])
    except Exception:
        return 0.0, 0.0, 0.0


def seed_from_klines_if_needed(ex: Any, symbol_ccxt: str, minutes: int = 120) -> List[float]:
    """Optionally pre-seed the price window with historical 1m candles."""

    try:
        klines = ex.fetch_ohlcv(symbol_ccxt, timeframe="1m", limit=minutes)
    except Exception:
        return []
    return [float(o) for o, _, _, _, c, _ in klines for o in (o, c)]


# ---------------------------------------------------------------------------
#  Collector class
# ---------------------------------------------------------------------------


@dataclass
class MicroV5Collector:
    ex_public: Any
    symbol_ccxt: str
    win_secs: int = 7200
    ob_limit: int = 20
    interval_s: float = 1.0
    out_dir: str | Path = "data/processed"
    flush_every_n: int = 60

    def __post_init__(self) -> None:
        self.symbol_raw = self.symbol_ccxt.replace("/", "")
        self.prices: deque[float] = deque(maxlen=self.win_secs)
        self.buffer: List[Dict[str, Any]] = []
        self.out_path = Path(self.out_dir) / self.symbol_raw
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.minQty, self.maxQty, self.stepSize = fetch_lot_size_meta(self.ex_public, self.symbol_raw)

    # -- internal utilities -------------------------------------------------
    def _flush(self) -> None:
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        day = datetime.fromtimestamp(df["ts"].iloc[0], tz=UTC).strftime("%Y%m%d")
        file = self.out_path / f"micro_v5_{day}.parquet"
        if file.exists():
            prev = pd.read_parquet(file)
            df = pd.concat([prev, df], ignore_index=True)
        df.to_parquet(file, index=False)
        # update index file
        idx_path = self.out_path / "_index.json"
        stats = {
            "min_ts": float(df["ts"].min()),
            "max_ts": float(df["ts"].max()),
            "rows": len(df),
        }
        with open(idx_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh)
        self.buffer.clear()

    # -- public API ---------------------------------------------------------
    def step(self) -> None:
        ts = time.time()
        last = fetch_price_tick(self.ex_public, self.symbol_ccxt)
        ob = fetch_order_book(self.ex_public, self.symbol_ccxt, self.ob_limit)
        derivs = derivados_basicos(ob)

        self.prices.append(last)
        min_price = min(self.prices) if self.prices else last
        max_price = max(self.prices) if self.prices else last
        pct_up_from_min = (last - min_price) / min_price if min_price else 0.0
        pct_down_from_max = (max_price - last) / max_price if max_price else 0.0

        ask_peak_price = detectar_resistencia_asks(ob.get("asks", []))
        ask_mean_qty = (sum(q for _p, q in ob.get("asks", [])) / len(ob.get("asks", []))) if ob.get("asks") else 0.0
        ask_peak_qty = 0.0
        if ask_peak_price is not None:
            for price, qty in ob.get("asks", []):
                if price == ask_peak_price:
                    ask_peak_qty = qty
                    break

        thr_entry_up = umbral_entrada_dinamico(list(self.prices), 0.002, None)
        thr_exit_down = umbral_salida_dinamico(list(self.prices), 0.002, None)

        row = {
            "ts": ts,
            "last": last,
            "best_bid": derivs["best_bid"],
            "best_ask": derivs["best_ask"],
            "spread": derivs["spread"],
            "mid": derivs["mid"],
            "ask_peak_price": ask_peak_price,
            "ask_peak_qty": ask_peak_qty,
            "ask_mean_qty": ask_mean_qty,
            "pct_up_from_min": pct_up_from_min,
            "pct_down_from_max": pct_down_from_max,
            "thr_entry_up": thr_entry_up,
            "thr_exit_down": thr_exit_down,
            "minQty": self.minQty,
            "stepSize": self.stepSize,
            "symbol_raw": self.symbol_raw,
            "symbol_ccxt": self.symbol_ccxt,
        }
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_every_n:
            self._flush()

    def run_for(self, seconds: int | None = None, until_stop_flag: Any | None = None) -> None:
        start = time.time()
        while True:
            if seconds is not None and time.time() - start >= seconds:
                break
            if until_stop_flag and until_stop_flag():
                break
            try:
                self.step()
            except Exception:
                # Simple backoff on any error
                time.sleep(self.interval_s)
                continue
            time.sleep(self.interval_s)
        # Flush remaining
        self._flush()

