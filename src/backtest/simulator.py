from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import logging

from ..utils.risk import (
    passes_min_notional,
    apply_qty_step,
    apply_price_tick,
)
from ..risk.slippage import estimate_slippage
from ..exchange.binance_meta import BinanceMeta

logger = logging.getLogger(__name__)


def simulate(
    price_df: pd.DataFrame,
    policy,
    *,
    fees: float = 0.001,
    slippage_multiplier: float = 1.0,
    min_notional_usd: float = 10.0,
    tick_size: float = 0.01,
    step_size: float = 0.0001,
    symbol: str = "BTC/USDT",
    slippage_depth: int = 50,
    slippage_static: float = 0.0,
    meta: BinanceMeta | None = None,
) -> Dict[str, Any]:
    """Run a minimalistic trading simulation.

    The simulator supports basic execution frictions such as commissions,
    slippage, minimum notionals and exchange rounding rules.
    """

    equity = 1.0
    position = 0  # 0 -> flat, 1 -> long
    entry = 0.0
    peak = 0.0
    trades: list[Dict[str, Any]] = []

    if meta is None:
        try:
            meta = BinanceMeta()
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("meta_fallback reason=%s", exc)
            meta = None
    if meta is not None:
        try:
            filt = meta.get_symbol_filters(symbol)
            tick_size = float(filt.get("tickSize", tick_size))
            step_size = float(filt.get("stepSize", step_size))
            min_notional_usd = float(filt.get("minNotional", min_notional_usd))
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("filter_fallback reason=%s", exc)

    qty = apply_qty_step(1.0, step_size)

    for i in range(len(price_df)):
        row = price_df.iloc[i]
        px = float(row.close)

        trailing_norm = 0.0 if position == 0 else (px - peak) / (peak + 1e-12)
        obs = np.array(
            [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, float(position), trailing_norm],
            dtype=np.float32,
        )
        action = policy.act(obs) if hasattr(policy, "act") else 0

        if action == 1 and position == 0:
            notional = px * qty
            recent = price_df["close"].iloc[max(0, i - 60) : i + 1]
            try:
                slip = estimate_slippage(
                    symbol, notional, "buy", depth=slippage_depth, prices=recent
                ) * slippage_multiplier
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("slippage_static reason=%s", exc)
                slip = slippage_static * slippage_multiplier
            exec_px = apply_price_tick(px * (1 + slip), tick_size)
            value = exec_px * qty
            logger.info(
                "sim_open i=%s price=%.8f qty=%.8f value=%.8f slippage=%.6f tick=%.6f step=%.6f minNotional=%.6f",
                i,
                exec_px,
                qty,
                value,
                slip,
                tick_size,
                step_size,
                min_notional_usd,
            )
            if passes_min_notional(exec_px, qty, min_notional_usd):
                entry = exec_px
                peak = entry
                position = 1
            else:
                logger.info(
                    "Trade blocked: value %.8f < minNotional %.8f",
                    value,
                    min_notional_usd,
                )
        elif action == 2 and position == 1:
            notional = px * qty
            recent = price_df["close"].iloc[max(0, i - 60) : i + 1]
            try:
                slip = estimate_slippage(
                    symbol, notional, "sell", depth=slippage_depth, prices=recent
                ) * slippage_multiplier
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("slippage_static reason=%s", exc)
                slip = slippage_static * slippage_multiplier
            exit_px = apply_price_tick(px * (1 - slip), tick_size)
            value = exit_px * qty
            logger.info(
                "sim_close i=%s price=%.8f qty=%.8f value=%.8f slippage=%.6f tick=%.6f step=%.6f minNotional=%.6f",
                i,
                exit_px,
                qty,
                value,
                slip,
                tick_size,
                step_size,
                min_notional_usd,
            )
            if passes_min_notional(exit_px, qty, min_notional_usd):
                cost = entry * qty * (1 + fees)
                proceeds = exit_px * qty * (1 - fees)
                pnl = (proceeds - cost) / cost
                equity *= 1.0 + pnl
                trades.append(
                    {"i": i, "entry": entry, "exit": exit_px, "pnl": pnl, "equity": equity}
                )
                position = 0
                entry = 0.0
                peak = 0.0
            else:
                logger.info(
                    "Trade blocked: value %.8f < minNotional %.8f",
                    value,
                    min_notional_usd,
                )

        if position == 1:
            peak = max(peak, px)

    returns = pd.Series([t["pnl"] for t in trades]) if trades else pd.Series(dtype=float)
    return {"equity": equity, "trades": trades, "returns": returns}

