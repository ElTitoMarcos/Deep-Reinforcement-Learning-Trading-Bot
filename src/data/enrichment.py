from __future__ import annotations

"""Utilities for enriching market data.

The functions in this module provide best-effort access to public market
information.  Binance via ``ccxt`` is attempted first; hooks for alternative
providers are left as extension points.  The goal is to normalise the returned
information so downstream components can rely on a minimal schema.
"""

from typing import Any, Dict, List

import pandas as pd

from .ccxt_loader import get_exchange


def _safe_get(lst, idx, default=None):
    try:
        return lst[idx]
    except Exception:
        return default


def fetch_symbol_metadata(symbols: List[str], prefer: str = "binance") -> Dict[str, Any]:
    """Return normalised metadata for *symbols*.

    Parameters
    ----------
    symbols:
        List of symbols like ``"BTC/USDT"``.
    prefer:
        The preferred data source.  Currently only ``"binance"`` is supported
        but the argument is kept as an extension point for future providers.
    """

    meta: Dict[str, Any] = {}
    exchange = None
    if prefer == "binance":
        try:
            exchange = get_exchange()
        except Exception as exc:  # pragma: no cover - network/ccxt quirks
            exchange = None
            meta["__error"] = f"binance_unavailable: {exc}"

    for sym in symbols:
        info: Dict[str, Any] = {"source": prefer}
        if exchange is not None:
            try:  # pragma: no cover - network/ccxt quirks
                market = exchange.market(sym)
                ticker = exchange.fetch_ticker(sym)
                orderbook = exchange.fetch_order_book(sym, limit=5)

                filters = {
                    "PRICE_FILTER": _safe_get(market.get("precision", {}), "price"),
                    "LOT_SIZE": _safe_get(market.get("precision", {}), "amount"),
                    "MIN_NOTIONAL": _safe_get(
                        market.get("limits", {}).get("cost", {}), "min"
                    ),
                }

                best_bid = _safe_get(orderbook.get("bids", []), 0, [None, None])[0]
                best_ask = _safe_get(orderbook.get("asks", []), 0, [None, None])[0]

                info.update(
                    {
                        "status": "TRADING" if market.get("active") else "OFF",
                        "filters": filters,
                        "ticker": {
                            "last": ticker.get("last"),
                            "volume": ticker.get("baseVolume"),
                            "quoteVolume": ticker.get("quoteVolume"),
                            "count": _safe_get(ticker.get("info", {}), "count"),
                        },
                        "book": {
                            "bid": best_bid,
                            "ask": best_ask,
                            "depth": {
                                "bids": orderbook.get("bids", [])[:5],
                                "asks": orderbook.get("asks", [])[:5],
                            },
                        },
                    }
                )
            except Exception as exc:  # pragma: no cover - network/ccxt quirks
                info["source"] = "unavailable"
                info["error"] = str(exc)
        meta[sym] = info
    return meta


def fetch_extra_series(
    symbol: str, timeframe: str = "1m", hours: int = 720
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV and auxiliary series for *symbol*.

    Parameters
    ----------
    symbol:
        Market symbol like ``"BTC/USDT"``.
    timeframe:
        Candle timeframe supported by the exchange.
    hours:
        Number of hours of history to request.
    """

    exchange = get_exchange()
    end = exchange.milliseconds()
    since = end - hours * 3600 * 1000

    data: Dict[str, pd.DataFrame] = {}
    try:  # pragma: no cover - network/ccxt quirks
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
        df = pd.DataFrame(
            ohlcv, columns=["ts", "open", "high", "low", "close", "volume"]
        )
        df.set_index("ts", inplace=True)
        returns = df["close"].pct_change()
        df["volatility"] = returns.rolling(30).std()
        df["vol_norm"] = df["volume"] / df["volume"].rolling(30).mean()

        book = exchange.fetch_order_book(symbol, limit=5)
        bid = _safe_get(book.get("bids", []), 0, [None, None])[0]
        ask = _safe_get(book.get("asks", []), 0, [None, None])[0]
        if bid and ask:
            mid = (bid + ask) / 2.0
            df["spread"] = ask - bid
            df["mid_price"] = mid
            bid_vol = _safe_get(book.get("bids", []), 0, [0, 0])[1]
            ask_vol = _safe_get(book.get("asks", []), 0, [0, 0])[1]
            denom = bid_vol + ask_vol
            if denom > 0:
                df["ob_imbalance"] = (bid_vol - ask_vol) / denom
        data["ohlcv"] = df
    except Exception:  # pragma: no cover - network/ccxt quirks
        data["ohlcv"] = pd.DataFrame()

    try:  # pragma: no cover - network/ccxt quirks
        trades = exchange.fetch_trades(symbol, since=since)
        df_trades = pd.DataFrame(trades)
        data["trades"] = df_trades
    except Exception:
        data["trades"] = pd.DataFrame()

    return data

