"""Minimal trading environment.

This is intentionally lightweight and only provides the tiny subset of
functionality required for the unit tests.  It is **not** intended to be a
full featured trading simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class _Space:
    """Simple stand-in for a gym ``Space`` object."""

    shape: Tuple[int, ...]
    dtype: Any = np.float32

@dataclass
class _Discrete:
    """Minimal discrete space (``n`` possible integer actions)."""

    n: int
    dtype: Any = np.int64


class TradingEnv:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.current_step = 0

        # caches ---------------------------------------------------------
        self._close = self.df["close"].to_numpy(dtype=float)
        self._low = self.df["low"].to_numpy(dtype=float)
        self._high = self.df["high"].to_numpy(dtype=float)

        # state ----------------------------------------------------------
        self.in_position = False
        self.trailing_stop: float | None = None
        self.entry_price: float | None = None
        self.equity = 0.0
        self.equity_peak = 0.0

        # history for robust scaling (one list per feature)
        self._feature_histories: List[List[float]] = [[] for _ in range(8)]

        # observation space: 8 engineered features, float32
        self.observation_space = _Space((8,), np.float32)
        # discrete action space: 0=hold, 1=open_long, 2=close
        self.action_space = _Discrete(3, np.int64)
        # in the future this could include a continuous component (0..1)
        # to express position sizing alongside the discrete action
        # config ---------------------------------------------------------
        with open("configs/default.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        fees = cfg.get("fees", {})
        self.fee_rate = float(fees.get("taker", 0.0))
        rw = cfg.get("reward_weights", {})
        self.w_pnl = float(rw.get("pnl", 1.0))
        self.w_turn = float(rw.get("turn", 0.0))
        self.w_dd = float(rw.get("dd", 0.0))
        self.w_vol = float(rw.get("vol", 0.0))

    # ------------------------------------------------------------------
    def _make_observation(self, step: int) -> np.ndarray:
        """Create the observation vector for ``step``.

        Features (pre-normalisation):
            - log returns over 5/15/60 ticks
            - rolling volatility over the past 60 returns
            - local drawdown over the past 300 ticks
            - distance to local minimum ("wall") over the past 300 ticks
            - in-position flag (0/1)
            - normalised trailing stop distance

        Each feature is normalised using a simple online robust scaler
        (median/IQR) that only looks at past values of the respective
        feature.
        """

        price = self._close[step]

        # price based ----------------------------------------------------
        def safe_log_return(n: int) -> float:
            if step >= n:
                return float(np.log(price / self._close[step - n]))
            return 0.0

        ret_5 = safe_log_return(5)
        ret_15 = safe_log_return(15)
        ret_60 = safe_log_return(60)

        # rolling volatility of 1-step log returns
        start_idx = max(1, step - 59)
        window_returns = np.diff(np.log(self._close[start_idx: step + 1]))
        vol_60 = float(np.std(window_returns)) if len(window_returns) > 0 else 0.0

        # drawdown relative to local max over last 300 ticks
        max_price = float(np.max(self._close[max(0, step - 299): step + 1]))
        drawdown_300 = float(price / max_price - 1.0) if max_price > 0 else 0.0

        # distance to local minimum ("wall") over last 300 ticks
        min_price = float(np.min(self._low[max(0, step - 299): step + 1]))
        dist_wall = float(price - min_price)

        # position based -------------------------------------------------
        en_posicion = 1.0 if self.in_position else 0.0

        if self.in_position and self.trailing_stop is not None and self.trailing_stop > 0:
            trailing_normalizado = float((price - self.trailing_stop) / self.trailing_stop)
        else:
            trailing_normalizado = 0.0

        raw_features = [
            ret_5,
            ret_15,
            ret_60,
            vol_60,
            drawdown_300,
            dist_wall,
            en_posicion,
            trailing_normalizado,
        ]

        # robust scaling -------------------------------------------------
        scaled_features = []
        for i, val in enumerate(raw_features):
            hist = self._feature_histories[i]
            if hist:
                median = float(np.median(hist))
                q75 = float(np.percentile(hist, 75))
                q25 = float(np.percentile(hist, 25))
                iqr = q75 - q25
                if iqr == 0:
                    iqr = 1.0
                scaled = (val - median) / iqr
            else:
                scaled = 0.0
            hist.append(val)
            scaled_features.append(scaled)

        return np.asarray(scaled_features, dtype=np.float32)

    # public API -------------------------------------------------------
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.current_step = 0
        self.in_position = False
        self.trailing_stop = None
        self.entry_price = None
        self.equity = 0.0
        self.equity_peak = 0.0
        self._feature_histories = [[] for _ in range(8)]
        obs = self._make_observation(self.current_step)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_price = self._close[self.current_step]
        prev_equity = self.equity
        prev_drawdown = self.equity_peak - self.equity
        trade = False

        # action: 0=hold, 1=open_long, 2=close
        if action == 1 and not self.in_position:  # open long
            self.in_position = True
            self.trailing_stop = prev_price
            self.entry_price = prev_price
            fee = prev_price * self.fee_rate
            self.equity -= fee
            trade = True
        elif action == 2 and self.in_position:  # close position
            fee = prev_price * self.fee_rate
            self.equity += (prev_price - self.entry_price) - fee
            self.in_position = False
            self.trailing_stop = None
            self.entry_price = None
            trade = True

        # TODO: support a continuous size component (0..1) alongside the
        # discrete action for finer trade management

        self.current_step += 1
        self.current_step = min(self.current_step, len(self._close) - 1)
        done = self.current_step >= len(self._close) - 1
        price = self._close[self.current_step]

        if self.in_position:
            self.equity += price - prev_price
            if self.trailing_stop is not None:
                self.trailing_stop = max(self.trailing_stop, price)

        pnl_step = self.equity - prev_equity
        self.equity_peak = max(self.equity_peak, self.equity)
        new_drawdown = self.equity_peak - self.equity
        dd_step = max(0.0, new_drawdown - prev_drawdown)
        trades_step = 1.0 if trade else 0.0

        start_idx = max(1, self.current_step - 4)
        window_returns = np.diff(np.log(self._close[start_idx: self.current_step + 1]))
        vol_step = float(np.std(window_returns)) if len(window_returns) > 0 else 0.0

        reward = (
            self.w_pnl * pnl_step
            - self.w_turn * trades_step
            - self.w_dd * dd_step
            - self.w_vol * vol_step
        )

        logger.debug(
            "step=%s pnl=%.6f trades=%.2f dd=%.6f vol=%.6f reward=%.6f",
            self.current_step,
            pnl_step,
            trades_step,
            dd_step,
            vol_step,
            reward,
        )

        obs = self._make_observation(self.current_step)
        info = {
            "reward_terms": {
                "pnl": pnl_step,
                "turnover": trades_step,
                "drawdown": dd_step,
                "volatility": vol_step,
            }
        }
        return obs, reward, done, False, info

