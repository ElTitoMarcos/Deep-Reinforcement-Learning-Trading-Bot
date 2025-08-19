from __future__ import annotations

"""Validation helpers for symbol availability and state."""

from typing import Dict, List, Tuple
from difflib import get_close_matches


SUGGEST_CUTOFF = 0.85


def validate_symbols(exchange, symbols: List[str]) -> Tuple[List[str], List[Dict]]:
    """Return valid symbols and diagnostics for invalid ones.

    Parameters
    ----------
    exchange:
        ccxt-like exchange instance exposing ``load_markets``.
    symbols:
        List of market symbols in ``BASE/QUOTE`` notation.

    Returns
    -------
    (valid, invalid):
        ``valid`` is a list of symbols that are tradable spot markets.
        ``invalid`` contains dictionaries with ``symbol`` and ``reason`` keys;
        a ``suggest`` field is included when a close match is found.
    """

    try:
        markets = exchange.load_markets()
    except Exception:  # pragma: no cover - network or exchange errors
        markets = {}

    valid: List[str] = []
    invalid: List[Dict] = []
    market_names = list(markets.keys())

    for sym in symbols:
        reasons = []
        market = markets.get(sym)
        if market is None:
            reasons.append("no existe")
            suggestion = get_close_matches(sym, market_names, n=1, cutoff=SUGGEST_CUTOFF)
            entry: Dict = {"symbol": sym, "reason": "; ".join(reasons)}
            if suggestion:
                entry["suggest"] = suggestion[0]
            invalid.append(entry)
            continue

        status = market.get("info", {}).get("status")
        if status != "TRADING":
            reasons.append(f"status={status}")
        if not market.get("spot", False):
            reasons.append("no en spot")

        if reasons:
            invalid.append({"symbol": sym, "reason": "; ".join(reasons)})
        else:
            valid.append(sym)

    return valid, invalid

