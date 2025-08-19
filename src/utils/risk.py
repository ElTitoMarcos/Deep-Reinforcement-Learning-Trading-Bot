from __future__ import annotations
from typing import Tuple

def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step

def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick

def passes_min_notional(price: float, qty: float, min_notional_usd: float, quote_to_usd: float = 1.0) -> bool:
    return (price * qty * quote_to_usd) >= min_notional_usd
