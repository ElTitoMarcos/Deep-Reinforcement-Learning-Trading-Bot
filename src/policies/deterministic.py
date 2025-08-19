from __future__ import annotations
from typing import Dict, Any, Tuple

import numpy as np


class DeterministicPolicy:
    """Rule-based policy with rebound entry, trailing exit and cooldown.

    Parameters are expressed as fractions (``0.01`` → ``1%``).
    """

    def __init__(
        self,
        base_threshold: float = 0.001,
        bounce_coef: float = 0.5,
        trailing_pct: float = 0.01,
        cooldown_ticks: int = 5,
        threshold: float | None = None,
    ) -> None:
        # ``threshold`` kept for backwards compatibility with older configs
        if threshold is not None:
            base_threshold = threshold

        self.base_threshold = float(base_threshold)
        self.bounce_coef = float(bounce_coef)
        self.trailing_pct = float(trailing_pct)
        self.cooldown_ticks = int(cooldown_ticks)

        self.last_trade_tick = -np.inf  # enforce cooldown
        self.tick = 0

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, return_trace: bool = False) -> Any:
        """Return an action and optionally a decision trace.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector. Expected order:
            ``[ret_5, ret_15, ret_60, vol_60, drawdown, dist_wall, in_pos, trailing_norm]``.
        return_trace : bool, optional
            Whether to return a trace dict along with the action.
        """

        ret_5 = float(obs[0])
        drawdown = abs(float(obs[4]))
        in_pos = bool(obs[6] > 0.5)
        trailing_norm = float(obs[7])

        # dynamic entry threshold: bigger prior drop → smaller required rise
        dyn_threshold = max(0.0, self.base_threshold - self.bounce_coef * drawdown)

        can_trade = (self.tick - self.last_trade_tick) >= self.cooldown_ticks
        action = 0  # default hold

        if not in_pos and can_trade and ret_5 > dyn_threshold:
            action = 1  # open long
            self.last_trade_tick = self.tick
        elif in_pos and trailing_norm <= -self.trailing_pct:
            action = 2  # close position
            self.last_trade_tick = self.tick

        trace: Dict[str, Any] = {
            "ret_5": ret_5,
            "drawdown": drawdown,
            "dyn_threshold": dyn_threshold,
            "trailing_norm": trailing_norm,
            "in_pos": in_pos,
            "can_trade": can_trade,
            "action": action,
        }

        self.tick += 1

        if return_trace:
            return action, trace
        return action
