"""Feature helpers for Microstructure V5 pipeline.

This module implements a tiny subset of the logic used by the original
``BOT_v5`` project.  The real bot exposes a fairly involved set of signal and
threshold calculations; for the purposes of the open source project we only
implement the pieces that are required by the unit tests.  The goal is to
provide deterministic behaviour that mimics the public description of the
original functions.

All functions operate on plain Python types making them easy to unit test and
mock in isolation.  Percentages are expressed as fractions (``0.01`` equals
``1%``), matching the conventions of ``BOT_v5``.
"""

from __future__ import annotations

from statistics import mean
from typing import Iterable, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
#  Basic order book helpers
# ---------------------------------------------------------------------------

def derivados_basicos(ob: Dict[str, Any]) -> Dict[str, float]:
    """Return standard microstructure metrics from an order book.

    Parameters
    ----------
    ob: dict
        A dictionary with ``"bids"`` and ``"asks"`` lists.  Each list contains
        price/quantity pairs ``[price, qty]`` ordered best first.

    Returns
    -------
    dict
        ``best_bid``, ``best_ask``, ``spread``, ``mid``, ``imbalance`` and
        ``microprice``.  Missing data results in ``None`` values.
    """

    bids = ob.get("bids") or []
    asks = ob.get("asks") or []

    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    spread = (best_ask - best_bid) if best_bid is not None and best_ask is not None else None
    mid = ((best_ask + best_bid) / 2) if best_bid is not None and best_ask is not None else None

    bid_vol = bids[0][1] if bids else 0.0
    ask_vol = asks[0][1] if asks else 0.0
    total_vol = bid_vol + ask_vol
    imbalance = ((bid_vol - ask_vol) / total_vol) if total_vol else 0.0
    microprice = (
        (best_ask * bid_vol + best_bid * ask_vol) / total_vol
        if total_vol and best_bid is not None and best_ask is not None
        else None
    )

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid": mid,
        "imbalance": imbalance,
        "microprice": microprice,
    }


# ---------------------------------------------------------------------------
#  Resistance detection
# ---------------------------------------------------------------------------

def detectar_resistencia_asks(asks: Iterable[Tuple[float, float]]) -> float | None:
    """Return the price level that acts as ask-side "resistance".

    The original BOT_v5 defined a resistance when a single ask level presented
    a quantity at least five times larger than the average quantity across the
    observed levels.  The function returns the price of the first level that
    satisfies the condition or ``None`` if none do.
    """

    asks = list(asks)
    if not asks:
        return None

    quantities = [qty for _price, qty in asks]
    if len(quantities) <= 1:
        return None
    max_qty = max(quantities)
    others = [q for q in quantities if q != max_qty]
    avg = mean(others) if others else 0
    if max_qty >= 5 * avg and avg > 0:
        for price, qty in asks:
            if qty == max_qty:
                return price
    return None


# ---------------------------------------------------------------------------
#  Dynamic thresholds
# ---------------------------------------------------------------------------

def _pct_drop(prices: List[float]) -> float:
    max_p = max(prices)
    last = prices[-1]
    return (max_p - last) / max_p if max_p else 0.0


def _pct_rise(prices: List[float]) -> float:
    min_p = min(prices)
    last = prices[-1]
    return (last - min_p) / min_p if min_p else 0.0


def umbral_entrada_dinamico(precios_window: List[float], thr_actual: float, hist_bajada: List[float] | None = None) -> float:
    """Adjust the entry threshold based on recent price drops.

    ``precios_window`` is a list of recent last prices.  ``thr_actual`` is the
    current threshold expressed as a fraction.  Whenever the price has dropped
    more than ``0.40%`` from the local maximum the threshold is reduced in the
    same proportion (but never below zero).  This loosely mimics the behaviour
    of ``modificar_precio_subida`` from BOT_v5.
    """

    drop = _pct_drop(precios_window)
    if drop >= 0.004:  # 0.40%
        thr_actual = max(thr_actual - drop, 0.0)
    return thr_actual


def umbral_salida_dinamico(precios_window: List[float], thr_actual: float, hist_subida: List[float] | None = None) -> float:
    """Adjust the exit threshold based on recent price rises.

    When the price has risen more than ``1.2%`` from the local minimum, the
    threshold is reduced (making exits easier).  This approximates
    ``modificar_precio_bajada`` from BOT_v5.
    """

    rise = _pct_rise(precios_window)
    if rise > 0.012:  # 1.2%
        thr_actual = max(thr_actual - rise, 0.0)
    return thr_actual

