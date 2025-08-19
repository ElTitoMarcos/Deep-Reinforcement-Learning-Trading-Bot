import pandas as pd
import logging

from src.env.trading_env import TradingEnv


def _make_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [i * 60000 for i in range(n)],
            "open": [100 + i for i in range(n)],
            "high": [101 + i for i in range(n)],
            "low": [99 + i for i in range(n)],
            "close": [100 + i for i in range(n)],
            "volume": [1 for _ in range(n)],
        }
    )


def test_cooldown_blocks(caplog):
    env = TradingEnv(_make_df(), trade_cooldown_seconds=120)
    env.reset()
    env.step(1)  # open trade at t=0
    with caplog.at_level(logging.INFO):
        _, _, _, _, info = env.step(2)  # attempt close at t=60s
    assert info["reward_terms"]["turnover"] == 2.0
    assert env.in_position  # still open due to block
    assert "cooldown" in caplog.text.lower()


def test_max_trades_window(caplog):
    env = TradingEnv(
        _make_df(),
        max_trades_per_window=1,
        trade_window_seconds=180,
        trade_cooldown_seconds=0,
    )
    env.reset()
    env.step(1)  # first trade
    with caplog.at_level(logging.INFO):
        _, _, _, _, info = env.step(2)  # second trade within window
    assert info["reward_terms"]["turnover"] == 2.0
    assert env.in_position
    assert "max" in caplog.text.lower()
