from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, UTC
from typing import List

import ccxt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils.logging import ensure_logger
from src.utils.data_io import save_ohlcv, validate_ohlcv, resample_to, fill_small_gaps
from src.utils import paths


def parse_since(s: str | None) -> int | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def fetch_with_retries(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int | None,
    rate_limit: float,
    retries: int,
    logger,
) -> List[List[float]]:
    delay = rate_limit
    for attempt in range(retries):
        try:
            time.sleep(rate_limit)
            data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
            return data
        except Exception as e:  # ccxt.BaseError
            logger.warning("fetch_retry", symbol=symbol, attempt=attempt + 1, error=str(e), wait=delay)
            time.sleep(delay)
            delay *= 2
    logger.error("fetch_failed", symbol=symbol, timeframe=timeframe)
    return []


def download_ohlcv(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since: int | None,
    rate_limit: float,
    retries: int,
    logger,
) -> pd.DataFrame:
    all_rows: List[List[float]] = []
    ms_per_tf = int(ex.parse_timeframe(timeframe) * 1000)
    fetch_since = since
    while True:
        chunk = fetch_with_retries(ex, symbol, timeframe, fetch_since, rate_limit, retries, logger)
        if not chunk:
            break
        all_rows.extend(chunk)
        fetch_since = chunk[-1][0] + ms_per_tf
        if len(chunk) < 1000:
            break
    columns = ["ts", "open", "high", "low", "close", "volume"]
    return pd.DataFrame(all_rows, columns=columns)


def synthesize_1s_from_1m(df_1m: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _, row in df_1m.iterrows():
        base_ts = int(row["ts"])
        base_open = row["open"]
        base_close = row["close"]
        base_volume = row["volume"] / 60.0
        for i in range(60):
            ts = base_ts + i * 1000
            price = base_open + (base_close - base_open) * (i / 59 if 59 else 0)
            noise = rng.normal(scale=0.0001 * price)
            price = price + noise
            rows.append([ts, price, price, price, price, base_volume])
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframe", type=str, choices=["1m", "1s"], default="1m")
    ap.add_argument("--since", type=str, default=None, help="ISO date (UTC) p.ej. 2024-01-01")
    ap.add_argument("--root", type=str, default=paths.posix(paths.RAW_DIR))
    ap.add_argument("--rate-limit", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="seed for synthetic data")
    args = ap.parse_args()

    logger = ensure_logger(None)

    since_ms = parse_since(args.since)

    paths.RAW_DIR = Path(args.root)
    paths.ensure_dirs_exist()

    for sym in args.symbols:
        ex_name = args.exchange
        ex_class = getattr(ccxt, ex_name)
        ex = ex_class()
        ex.load_markets()

        if ex_name == "binance" and sym not in ex.symbols:
            logger.warning("symbol_not_found", exchange=ex_name, symbol=sym, fallback="kraken")
            ex_name = "kraken"
            ex = ccxt.kraken()
            ex.load_markets()
            if sym not in ex.symbols:
                logger.error("symbol_not_found_any", symbol=sym)
                continue

        try:
            df = download_ohlcv(ex, sym, args.timeframe, since_ms, args.rate_limit, args.retries, logger)
        except Exception as e:
            logger.error("download_failed", exchange=ex_name, symbol=sym, error=str(e))
            continue

        source = "exchange"
        if args.timeframe == "1s" and df.empty:
            logger.warning("timeframe_unavailable", exchange=ex_name, symbol=sym, timeframe="1s")
            df_1m = download_ohlcv(ex, sym, "1m", since_ms, args.rate_limit, args.retries, logger)
            df = synthesize_1s_from_1m(df_1m, seed=args.seed)
            source = "synthetic"

        df = resample_to(df, args.timeframe)
        df, filled = fill_small_gaps(df)
        if filled > 0:
            logger.warning("filled_small_gaps", symbol=sym, ticks=filled)

        df["exchange"] = ex_name
        df["symbol"] = sym
        df["timeframe"] = args.timeframe
        df["source"] = source

        try:
            df = validate_ohlcv(df)
        except ValueError as e:
            logger.error("validation_failed", exchange=ex_name, symbol=sym, error=str(e))
            continue

        path = save_ohlcv(df, ex_name, sym, args.timeframe)
        logger.info("saved", path=path, rows=len(df))


if __name__ == "__main__":
    main()

