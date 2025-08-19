"""Slippage estimation utilities."""
from __future__ import annotations

from typing import Sequence, Any
import logging
import numpy as np

from ..data.ccxt_loader import get_exchange

logger = logging.getLogger(__name__)


def _volatility_fallback(prices: Sequence[float] | None) -> float:
    """Estimate slippage via recent volatility when order book depth is insufficient."""
    if prices is not None and len(prices) > 1:
        returns = np.diff(np.log(np.asarray(prices, dtype=float)))
        vol = float(np.std(returns))
        if np.isfinite(vol) and vol > 0:
            return vol
    return 0.001  # default fallback


def estimate_slippage(
    symbol: str,
    notional_usd: float,
    side: str,
    *,
    depth: int = 50,
    exchange: Any | None = None,
    prices: Sequence[float] | None = None,
) -> float:
    """Return estimated proportional slippage for a trade.

    Parameters
    ----------
    symbol: str
        Trading pair such as ``"BTC/USDT"``.
    notional_usd: float
        Quote currency value of the intended trade.
    side: str
        ``"buy"`` or ``"sell"``.
    depth: int, optional
        Order book depth levels to request.
    exchange: ccxt-like client, optional
        Used mainly for testing. When ``None`` a Binance client is created.
    prices: sequence of floats, optional
        Recent prices used to compute a volatility fallback when the order
        book does not provide enough depth (e.g. on testnet).
    """
    ex = exchange
    if ex is None:  # pragma: no cover - exercised when ccxt is installed
        try:
            ex = get_exchange()
        except Exception as exc:  # pragma: no cover - ccxt missing
            logger.warning("could not create exchange: %s", exc)
            return _volatility_fallback(prices)

    try:  # pragma: no cover - network call
        ob = ex.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            raise ValueError("empty order book")
        mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
        book = asks if side.lower() == "buy" else bids
        remaining = float(notional_usd)
        cost = 0.0
        qty_acc = 0.0
        for price, amount in book:
            px = float(price)
            amt = float(amount)
            vol = px * amt
            take = min(remaining, vol)
            qty = take / px if px > 0 else 0.0
            cost += px * qty
            qty_acc += qty
            remaining -= take
            if remaining <= 0:
                break
        if remaining > 0:
            raise ValueError("insufficient depth")
        avg_px = cost / qty_acc if qty_acc > 0 else mid
        impact = (avg_px - mid) / mid if side.lower() == "buy" else (mid - avg_px) / mid
        return max(impact, 0.0)
    except Exception as exc:  # pragma: no cover - network issues, shallow book
        logger.info("slippage_fallback reason=%s", exc)
        return _volatility_fallback(prices)
