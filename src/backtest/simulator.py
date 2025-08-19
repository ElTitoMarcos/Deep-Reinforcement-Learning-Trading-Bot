from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from ..utils.risk import passes_min_notional, round_to_step, round_to_tick

def simulate(price_df: pd.DataFrame, policy, fees: float=0.001, slippage: float=0.0005, min_notional_usd: float=10.0, tick_size: float=0.01, step_size: float=0.0001) -> Dict[str, Any]:
    equity = 1.0
    max_equity = equity
    position = 0
    entry = 0.0
    trail = 0.0
    trades = []
    for i in range(len(price_df)):
        row = price_df.iloc[i]
        obs = np.array([1.0, 0.0, 0.0, 0.0, 0.0, float(position), trail/(row.close+1e-12)], dtype=np.float32)
        action = policy.act(obs) if hasattr(policy, "act") else 0
        px = float(row.close)
        if action == 1 and position == 0:
            position = 1
            entry = px * (1 + slippage)
        elif action == 2 and position == 1:
            exit_px = px * (1 - slippage)
            pnl = (exit_px - entry)/entry - fees
            equity *= (1.0 + pnl)
            max_equity = max(max_equity, equity)
            trades.append({"i": i, "entry": entry, "exit": exit_px, "pnl": pnl, "equity": equity})
            position = 0
            entry = 0.0
            trail = 0.0
        if position == 1:
            trail = max(trail, px - entry)
    returns = pd.Series([t["pnl"] for t in trades]) if trades else pd.Series(dtype=float)
    return {"equity": equity, "trades": trades, "returns": returns}
