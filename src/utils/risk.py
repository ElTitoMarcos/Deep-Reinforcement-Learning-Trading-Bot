from __future__ import annotations

from decimal import Decimal


def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(price / tick) * tick


def apply_price_tick(price: float, tick_size: float) -> float:
    """Floor ``price`` to the nearest multiple of ``tick_size``.

    Binance requires that order prices are multiples of the tick size. Any
    excess precision is dropped rather than rounded to the nearest tick.
    """
    if tick_size <= 0:
        return price
    d_price = Decimal(str(price))
    d_tick = Decimal(str(tick_size))
    return float((d_price // d_tick) * d_tick)


def apply_qty_step(qty: float, step_size: float) -> float:
    """Floor ``qty`` to the nearest multiple of ``step_size``."""
    if step_size <= 0:
        return qty
    d_qty = Decimal(str(qty))
    d_step = Decimal(str(step_size))
    return float((d_qty // d_step) * d_step)


def respects_min_notional(price: float, qty: float, min_notional: float) -> bool:
    """Return ``True`` if ``price * qty`` meets ``min_notional``."""
    return price * qty >= min_notional


def passes_min_notional(price: float, qty: float, min_notional_usd: float, quote_to_usd: float = 1.0) -> bool:
    return (price * qty * quote_to_usd) >= min_notional_usd
