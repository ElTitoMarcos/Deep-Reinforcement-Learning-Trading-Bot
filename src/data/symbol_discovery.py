from __future__ import annotations

from typing import List
from datetime import datetime, UTC

STABLES = {
    "USDT",
    "USDC",
    "BUSD",
    "DAI",
    "TUSD",
    "PAX",
    "EUR",
    "GBP",
    "JPY",
    "RUB",
}

EXOTIC_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def discover_symbols(exchange, quote: str = "USDT", top_n: int = 20) -> List[str]:
    """Return the most active symbols for *exchange* sorted by quote volume.

    Parameters
    ----------
    exchange:
        A ccxt exchange instance providing :meth:`fetch_tickers`.
    quote:
        The quote currency to filter for, defaults to ``"USDT"``.
    top_n:
        Maximum number of symbols to return.

    Returns
    -------
    list[str]
        Symbols like ``"BTC/USDT"`` ordered by descending ``quoteVolume``.
    """

    try:  # pragma: no cover - network/ccxt quirks
        tickers = exchange.fetch_tickers()
    except Exception as exc:  # pragma: no cover - network/ccxt quirks
        raise RuntimeError("failed to fetch tickers for discovery") from exc

    ranked = []
    for symbol, info in tickers.items():
        if not symbol.endswith(f"/{quote}"):
            continue
        base, _ = symbol.split("/")
        if base in STABLES:
            continue
        if any(base.endswith(suf) for suf in EXOTIC_SUFFIXES):
            continue
        qv = float(info.get("quoteVolume") or 0.0)
        ranked.append((qv, symbol))

    ranked.sort(reverse=True)
    return [sym for _, sym in ranked[:top_n]]


def discover_summary(symbols: List[str]) -> str:
    """Return a human readable summary for *symbols*.

    Example::

        "Top 15 por volumen USDT, excluidos stablecoins, actualizado 10:32 UTC"

    Parameters
    ----------
    symbols:
        List of symbol strings like ``"BTC/USDT"``.

    Returns
    -------
    str
        Summary describing the discovery outcome.
    """

    quote = symbols[0].split("/")[1] if symbols else "USDT"
    now = datetime.now(UTC).strftime("%H:%M")
    return (
        f"Top {len(symbols)} por volumen {quote}, excluidos stablecoins, "
        f"actualizado {now} UTC"
    )
