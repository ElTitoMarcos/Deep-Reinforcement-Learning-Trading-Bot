"""Build a liquid trading universe from exchange spot markets.

This script queries an exchange using the `ccxt` library and builds a
universe of spot trading pairs that satisfy basic liquidity
requirements.  The resulting table is written to a CSV file and
contains the following columns:

`symbol`, `quote`, `base`, `avg_volume_24h`, `first_seen`, `exchange`.

Filters can be applied based on 24h volume in USD, market age and the
available OHLCV history.  Basic JSON logging is emitted to stdout with
the number of markets remaining after each filtering step.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

try:  # pragma: no cover - optional import
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - makes intent explicit
    ccxt = None

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils.logging import ensure_logger
from src.utils.data_io import save_universe


# ---------------------------------------------------------------------------
# helpers


def _parse_first_seen(info: Dict) -> Optional[pd.Timestamp]:
    """Attempt to extract a listing timestamp from the raw exchange info."""

    candidates = [
        info.get("onboardDate"),
        info.get("listingTime"),
        info.get("listed"),
        info.get("timestamp"),
        info.get("created"),
        info.get("launchTime"),
        info.get("openDate"),
    ]
    for val in candidates:
        if val in (None, ""):
            continue
        try:
            if isinstance(val, (int, float)):
                unit = "ms" if int(val) > 1e10 else "s"
                return pd.to_datetime(int(val), unit=unit, utc=True)
            return pd.to_datetime(val, utc=True)
        except Exception:
            continue
    return None


def _history_days(ex: "ccxt.Exchange", symbol: str) -> Optional[int]:
    """Return number of days of OHLCV history available or ``None`` if
    unavailable."""

    try:
        data = ex.fetch_ohlcv(symbol, timeframe="1d", since=0, limit=1)
        if data:
            first_ts = data[0][0]
            first_dt = pd.to_datetime(first_ts, unit="ms", utc=True)
            return (pd.Timestamp.utcnow() - first_dt).days
    except Exception:
        return None
    return None


def _compute_volume_usd(ticker: Dict) -> float:
    """Best effort at computing 24h quote volume in USD."""

    vol = ticker.get("quoteVolume")
    if vol is None:
        base_vol = ticker.get("baseVolume")
        price = ticker.get("last") or ticker.get("close")
        if base_vol is not None and price is not None:
            vol = base_vol * price
    return float(vol) if vol is not None else 0.0


# ---------------------------------------------------------------------------
# main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--quote", type=str, default="USDT")
    ap.add_argument("--min-vol-usd", type=float, default=1_000_000.0)
    ap.add_argument("--min-age-days", type=float, default=30.0)
    ap.add_argument("--min-history-days", type=float, default=30.0)
    ap.add_argument(
        "--out",
        type=str,
        default="src/data/universe/liquid_universe.csv",
    )
    args = ap.parse_args()

    logger = ensure_logger(None)

    # normalize the quote currency to uppercase so that comparisons are
    # case-insensitive regardless of how the user provides the argument.
    args.quote = args.quote.upper()

    if ccxt is None:
        raise RuntimeError("ccxt no instalado. `pip install ccxt`.")

    ex_class = getattr(ccxt, args.exchange)
    ex = ex_class({"enableRateLimit": True})
    markets = ex.load_markets()
    logger.info("loaded_markets", count=len(markets))

    try:
        tickers = ex.fetch_tickers()
    except Exception as e:  # pragma: no cover - exchange dependent
        logger.warning("fetch_tickers_failed", error=str(e))
        tickers = {}

    rows: List[Dict] = []
    now = pd.Timestamp.utcnow()
    for sym, m in markets.items():
        if m.get("spot") is not True:
            continue
        # some exchanges may return the quote in lowercase; guard against it
        quote = (m.get("quote") or "").upper()
        if quote != args.quote:
            continue
        if not m.get("active", True):
            continue

        ticker = tickers.get(sym, {})
        vol_usd = _compute_volume_usd(ticker)
        first_seen = _parse_first_seen(m.get("info", {}))
        age_days = (now - first_seen).days if first_seen is not None else None

        rows.append(
            {
                "symbol": sym,
                "quote": m.get("quote"),
                "base": m.get("base"),
                "avg_volume_24h": vol_usd,
                "first_seen": first_seen,
                "age_days": age_days,
            }
        )

    logger.info("after_quote_filter", count=len(rows))

    df = pd.DataFrame(rows)

    before = len(df)
    df = df[df["avg_volume_24h"] >= args.min_vol_usd]
    logger.info("after_vol_filter", before=before, after=len(df))

    before = len(df)
    df = df[(df["age_days"].isna()) | (df["age_days"] >= args.min_age_days)]
    logger.info("after_age_filter", before=before, after=len(df))

    if args.min_history_days > 0 and not df.empty:
        history: List[Optional[int]] = []
        for sym in df["symbol"]:
            history.append(_history_days(ex, sym))
        df["history_days"] = history
        before = len(df)
        df = df[(df["history_days"].isna()) | (df["history_days"] >= args.min_history_days)]
        logger.info("after_history_filter", before=before, after=len(df))
    else:
        logger.info("after_history_filter", before=len(df), after=len(df))

    if df.empty:
        logger.warning("empty_universe")

    df.drop(columns=["age_days", "history_days"], errors="ignore", inplace=True)
    if "first_seen" in df.columns:
        df["first_seen"] = df["first_seen"].dt.strftime("%Y-%m-%d")
    df["exchange"] = args.exchange
    df = df[["symbol", "quote", "base", "avg_volume_24h", "first_seen", "exchange"]]

    save_universe(df, args.out)
    logger.info("saved_universe", path=args.out, symbols=len(df))

    print(f"Universo guardado en {args.out} con {len(df)} símbolos.")


if __name__ == "__main__":
    main()

