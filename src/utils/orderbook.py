from __future__ import annotations
from typing import List, Tuple, Dict

import numpy as np


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


def compute_walls(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    z_thr: float = 3.0,
) -> List[float]:
    """Detecta niveles de precio con volumen anómalo.

    Calcula el *z-score* del tamaño de cada nivel en ``bids`` y ``asks`` y
    devuelve los precios cuyo valor supera ``z_thr``.  El cálculo se hace de
    forma independiente por cada lado del libro.
    """

    walls: List[float] = []
    for side in (bids, asks):
        if not side:
            continue
        sizes = np.asarray([float(q) for _, q in side], dtype=float)
        mean = float(sizes.mean())
        std = float(sizes.std())
        if std == 0:
            continue
        for price, qty in side:
            z = (float(qty) - mean) / std
            if z > z_thr:
                walls.append(float(price))
    return walls


def distancia_a_muralla(mid: float, walls: List[float]) -> float:
    """Distancia normalizada desde ``mid`` a la muralla más cercana."""
    if mid <= 0 or not walls:
        return 0.0
    nearest = min(walls, key=lambda w: abs(w - mid))
    return abs(nearest - mid) / mid
