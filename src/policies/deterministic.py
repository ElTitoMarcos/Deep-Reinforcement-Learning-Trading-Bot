from __future__ import annotations
from typing import Dict, Any, Tuple
from collections import deque

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
        cooldown_seconds: float | None = None,
        cooldown_ticks: int | None = 5,
        max_trades: int | None = None,
        window_seconds: float | None = None,
        step_seconds: float = 60.0,
        threshold: float | None = None,
    ) -> None:
        """Create policy instance.

        ``threshold`` and ``cooldown_ticks`` kept for backwards compatibility.
        ``cooldown_seconds`` takes precedence over ``cooldown_ticks`` if both are
        provided.
        """

        if threshold is not None:
            base_threshold = threshold

        self.base_threshold = float(base_threshold)
        self.bounce_coef = float(bounce_coef)
        self.trailing_pct = float(trailing_pct)
        self.step_seconds = float(step_seconds)

        if cooldown_seconds is not None:
            self.cooldown_ticks = int(np.ceil(cooldown_seconds / self.step_seconds))
        else:
            self.cooldown_ticks = int(cooldown_ticks or 0)

        self.max_trades = int(max_trades) if max_trades is not None else None
        if self.max_trades is not None and window_seconds is not None:
            self.window_ticks = int(np.ceil(window_seconds / self.step_seconds))
        else:
            self.window_ticks = None
        self._trade_ticks = deque()

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

        def window_ok() -> bool:
            if self.max_trades is None or self.window_ticks is None:
                return True
            while self._trade_ticks and (self.tick - self._trade_ticks[0]) >= self.window_ticks:
                self._trade_ticks.popleft()
            return len(self._trade_ticks) < self.max_trades

        can_trade = (self.tick - self.last_trade_tick) >= self.cooldown_ticks and window_ok()
        action = 0  # default hold

        if not in_pos and can_trade and ret_5 > dyn_threshold:
            action = 1  # open long
            self.last_trade_tick = self.tick
            if self.max_trades is not None and self.window_ticks is not None:
                self._trade_ticks.append(self.tick)
        elif in_pos and can_trade and trailing_norm <= -self.trailing_pct:
            action = 2  # close position
            self.last_trade_tick = self.tick
            if self.max_trades is not None and self.window_ticks is not None:
                self._trade_ticks.append(self.tick)

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
