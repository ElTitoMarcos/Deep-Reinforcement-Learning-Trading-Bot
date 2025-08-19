"""Minimal trading environment.

This is intentionally lightweight and only provides the tiny subset of
functionality required for the unit tests.  It is **not** intended to be a
full featured trading simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class _Space:
    """Simple stand-in for a gym ``Space`` object."""

    shape: Tuple[int, ...]


class TradingEnv:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        # observation consists of OHLCV values
        self.observation_space = _Space((5,))

    # ------------------------------------------------------------------
    def _get_obs(self, step: int) -> np.ndarray:
        row = self.df.loc[step, ["open", "high", "low", "close", "volume"]]
        return row.to_numpy(dtype=float)

    # public API -------------------------------------------------------
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.current_step = 0
        return self._get_obs(self.current_step), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_close = float(self.df.loc[self.current_step, "close"])
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        curr_close = float(self.df.loc[self.current_step, "close"])
        reward = float(curr_close - prev_close)
        obs = self._get_obs(self.current_step)
        return obs, reward, done, False, {}

