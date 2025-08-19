from __future__ import annotations
from typing import List, Tuple, Dict

def wall_signal(orderbook: Dict, pct_threshold: float = 0.02) -> float:
    """Placeholder: detecta 'murallas' de liquidez en bids/asks.
    Devuelve un escalar en [-1, 1]: positivo si hay pared en bids (soporte),
    negativo si hay pared en asks (resistencia). Cero si neutro.
    """
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    if not bids or not asks:
        return 0.0
    best_bid_qty = float(bids[0][1])
    best_ask_qty = float(asks[0][1])
    total_bid = sum(float(q) for _, q in bids[:10])
    total_ask = sum(float(q) for _, q in asks[:10])
    if total_bid == 0 or total_ask == 0:
        return 0.0
    bias = (total_bid - total_ask) / max(total_bid + total_ask, 1e-12)
    # Clip to threshold range
    if abs(bias) < pct_threshold:
        return 0.0
    return max(-1.0, min(1.0, bias))
