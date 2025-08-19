"""Minimal trading environment.

This is intentionally lightweight and only provides the tiny subset of
functionality required for the unit tests.  It is **not** intended to be a
full featured trading simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Callable
from collections import deque

import numpy as np
import pandas as pd
import yaml
import logging

from ..exchange.binance_meta import BinanceMeta

try:  # pragma: no cover - optional gym dependency
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - optional gym dependency
    try:
        import gym
        from gym import spaces  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - fallback when gym is unavailable
        gym = None  # type: ignore[assignment]

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

        def _box(*, low: Any, high: Any, shape: Tuple[int, ...], dtype: Any = np.float32) -> _Space:
            return _Space(shape, dtype)

        def _discrete(n: int, dtype: Any = np.int64) -> _Discrete:
            return _Discrete(n, dtype)

        class _Spaces:  # minimal module-like container
            Box = staticmethod(_box)
            Discrete = staticmethod(_discrete)

        spaces = _Spaces()  # type: ignore[assignment]

from ..utils.orderbook import compute_walls, distancia_a_muralla
from ..risk.slippage import estimate_slippage

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env if 'gym' in globals() and gym is not None else object):
    def __init__(
        self,
        df: pd.DataFrame,
        orderbook_hook: Callable[[int], Dict[str, Any]] | None = None,
        *,
        cfg: dict | None = None,
        symbol: str | None = None,
        meta: BinanceMeta | None = None,
        max_trades_per_window: int | None = None,
        trade_window_seconds: float = 0.0,
        trade_cooldown_seconds: float = 0.0,
    ):
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self._orderbook_hook = orderbook_hook

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
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        # discrete action space: 0=hold, 1=open_long, 2=close
        self.action_space = spaces.Discrete(3)
        # in the future this could include a continuous component (0..1)
        # to express position sizing alongside the discrete action
        # config ---------------------------------------------------------
        if cfg is None:
            with open("configs/default.yaml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        self.cfg = cfg
        fees = cfg.get("fees", {})
        self.fee_rate = float(fees.get("taker", 0.0))
        rw = cfg.get("reward_weights", {})
        self.w_pnl = float(rw.get("pnl", 1.0))
        self.w_turn = float(rw.get("turn", 0.0))
        self.w_dd = float(rw.get("dd", 0.0))
        self.w_vol = float(rw.get("vol", 0.0))
        self.slippage_mult = float(cfg.get("slippage_multiplier", 1.0))
        self.slippage_depth = int(cfg.get("slippage_depth", 50))

        self.meta = meta
        self.symbol = symbol
        f_cfg = cfg.get("filters", {})
        self._tick_size = float(f_cfg.get("tickSize", 0.0))
        self._step_size = float(f_cfg.get("stepSize", 1.0))
        self._min_notional = float(cfg.get("min_notional_usd", 0.0))
        self.position_size = 0.0
        if self.meta and self.symbol:
            try:
                filt = self.meta.get_symbol_filters(self.symbol)
                self._tick_size = float(filt.get("tickSize", self._tick_size))
                self._step_size = float(filt.get("stepSize", self._step_size))
                self._min_notional = float(filt.get("minNotional", self._min_notional))
            except Exception:  # pragma: no cover - network issues
                pass

        self.meta = meta
        self.symbol = symbol
        f_cfg = cfg.get("filters", {})
        self._tick_size = float(f_cfg.get("tickSize", 0.0))
        self._step_size = float(f_cfg.get("stepSize", 1.0))
        self._min_notional = float(cfg.get("min_notional_usd", 0.0))
        self.position_size = 0.0
        if self.meta and self.symbol:
            try:
                filt = self.meta.get_symbol_filters(self.symbol)
                self._tick_size = float(filt.get("tickSize", self._tick_size))
                self._step_size = float(filt.get("stepSize", self._step_size))
                self._min_notional = float(filt.get("minNotional", self._min_notional))
            except Exception:  # pragma: no cover - network issues
                pass

        # environment timing and trade limits ----------------------------
        self.step_seconds = float(cfg.get("step_seconds", 60))
        self.max_trades_per_window = max_trades_per_window
        self.trade_window_seconds = float(trade_window_seconds)
        self.trade_cooldown_seconds = float(trade_cooldown_seconds)
        self._trade_times: deque[float] = deque()
        self._last_trade_time: float | None = None

    # ------------------------------------------------------------------
    def set_symbol(self, symbol: str) -> None:
        """Update the active *symbol* and reload exchange filters."""

        self.symbol = symbol
        if self.meta:
            try:
                filt = self.meta.get_symbol_filters(symbol)
                self._tick_size = float(filt.get("tickSize", self._tick_size))
                self._step_size = float(filt.get("stepSize", self._step_size))
                self._min_notional = float(filt.get("minNotional", self._min_notional))
            except Exception:  # pragma: no cover - network issues
                pass

    # ------------------------------------------------------------------
    def _make_observation(self, step: int) -> np.ndarray:
        """Create the observation vector for ``step``.

        Features (pre-normalisation):
            - log returns over 5/15/60 ticks
            - rolling volatility over the past 60 returns
            - local drawdown over the past 300 ticks
            - distance to liquidity wall (normalised), or to recent low if no orderbook
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

        # distance to liquidity wall -----------------------------------
        if self._orderbook_hook is not None:
            ob = self._orderbook_hook(step)
            bids = ob.get("bids") if ob else None
            asks = ob.get("asks") if ob else None
            if bids and asks:
                mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                walls = compute_walls(bids, asks)
                dist_wall = distancia_a_muralla(mid, walls)
            else:
                dist_wall = 0.0
        else:
            min_price = float(np.min(self._low[max(0, step - 299): step + 1]))
            dist_wall = (price - min_price) / price if price > 0 else 0.0

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
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment state.

        Parameters
        ----------
        seed: int | None
            Optional random seed for compatibility with Gym/Gymnasium.
        options: Dict[str, Any] | None
            Currently ignored, present for API completeness.
        """

        if seed is not None:
            try:  # pragma: no cover - seeding not critical for logic
                if hasattr(super(), "reset"):
                    super().reset(seed=seed)  # type: ignore[misc]
                else:
                    np.random.seed(seed)
            except TypeError:
                np.random.seed(seed)

        self.current_step = 0
        self.in_position = False
        self.trailing_stop = None
        self.entry_price = None
        self.position_size = 0.0
        self.equity = 0.0
        self.equity_peak = 0.0
        self._feature_histories = [[] for _ in range(8)]
        self._trade_times.clear()
        self._last_trade_time = None
        obs = self._make_observation(self.current_step)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        prev_price = self._close[self.current_step]
        prev_equity = self.equity
        prev_drawdown = self.equity_peak - self.equity
        trade = False
        attempted_trade = False

        now = self.current_step * self.step_seconds

        def can_trade() -> bool:
            if self.trade_cooldown_seconds > 0 and self._last_trade_time is not None:
                if now - self._last_trade_time < self.trade_cooldown_seconds:
                    logger.info(
                        "Trade blocked: cooldown %.1fs", self.trade_cooldown_seconds - (now - self._last_trade_time)
                    )
                    return False
            if (
                self.max_trades_per_window is not None
                and self.trade_window_seconds > 0
            ):
                while self._trade_times and now - self._trade_times[0] >= self.trade_window_seconds:
                    self._trade_times.popleft()
                if len(self._trade_times) >= self.max_trades_per_window:
                    logger.info(
                        "Trade blocked: max %s trades in %ss window",
                        self.max_trades_per_window,
                        self.trade_window_seconds,
                    )
                    return False
            return True

        # action: 0=hold, 1=open_long, 2=close
        if action == 1 and not self.in_position:
            attempted_trade = True
            if can_trade():
                price = prev_price
                qty = 1.0
                if self._step_size > 0:
                    qty = round(qty / self._step_size) * self._step_size
                notional = price * qty
                recent = self._close[max(0, self.current_step - 60) : self.current_step + 1]
                slip = estimate_slippage(
                    self.symbol or "BTC/USDT",
                    notional,
                    "buy",
                    depth=self.slippage_depth,
                    prices=recent,
                ) * self.slippage_mult
                price *= 1.0 + slip
                if self._tick_size > 0:
                    price = round(price / self._tick_size) * self._tick_size
                value = price * qty
                logger.info(
                    "open_order price=%.8f qty=%.8f value=%.8f slippage=%.6f tick=%.8f step=%.8f",
                    price,
                    qty,
                    value,
                    slip,
                    self._tick_size,
                    self._step_size,
                )
                if value < self._min_notional:
                    logger.info(
                        "Trade blocked: value %.8f < minNotional %.8f",
                        value,
                        self._min_notional,
                    )
                else:
                    self.in_position = True
                    self.position_size = qty
                    self.trailing_stop = price
                    self.entry_price = price
                    fee = value * self.fee_rate
                    self.equity -= fee
                    trade = True
        elif action == 2 and self.in_position:
            attempted_trade = True
            if can_trade():
                price = prev_price
                qty = self.position_size if self.position_size > 0 else 1.0
                if self._step_size > 0:
                    qty = round(qty / self._step_size) * self._step_size
                notional = price * qty
                recent = self._close[max(0, self.current_step - 60) : self.current_step + 1]
                slip = estimate_slippage(
                    self.symbol or "BTC/USDT",
                    notional,
                    "sell",
                    depth=self.slippage_depth,
                    prices=recent,
                ) * self.slippage_mult
                price *= 1.0 - slip
                if self._tick_size > 0:
                    price = round(price / self._tick_size) * self._tick_size
                value = price * qty
                logger.info(
                    "close_order price=%.8f qty=%.8f value=%.8f slippage=%.6f tick=%.8f step=%.8f",
                    price,
                    qty,
                    value,
                    slip,
                    self._tick_size,
                    self._step_size,
                )
                if value < self._min_notional:
                    logger.info(
                        "Trade blocked: value %.8f < minNotional %.8f",
                        value,
                        self._min_notional,
                    )
                else:
                    fee = value * self.fee_rate
                    self.equity += (price - self.entry_price) * qty - fee
                    self.in_position = False
                    self.position_size = 0.0
                    self.trailing_stop = None
                    self.entry_price = None
                    trade = True

        if trade:
            self._last_trade_time = now
            if self.max_trades_per_window is not None:
                self._trade_times.append(now)

        # TODO: support a continuous size component (0..1) alongside the
        # discrete action for finer trade management

        self.current_step += 1
        self.current_step = min(self.current_step, len(self._close) - 1)
        done = self.current_step >= len(self._close) - 1
        price = self._close[self.current_step]

        if self.in_position:
            self.equity += (price - prev_price) * self.position_size
            if self.trailing_stop is not None:
                self.trailing_stop = max(self.trailing_stop, price)

        pnl_step = self.equity - prev_equity
        self.equity_peak = max(self.equity_peak, self.equity)
        new_drawdown = self.equity_peak - self.equity
        dd_step = max(0.0, new_drawdown - prev_drawdown)
        trades_step = 1.0 if attempted_trade else 0.0
        if attempted_trade and not trade:
            trades_step += 1.0

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

