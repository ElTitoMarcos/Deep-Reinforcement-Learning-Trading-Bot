import pandas as pd
import pytest

from src.env.trading_env import TradingEnv


def test_reward_is_weighted_sum_and_info_terms_present():
    df = pd.DataFrame(
        {
            "ts": [0, 1000, 2000, 3000, 4000],
            "open": [100, 101, 102, 101, 103],
            "high": [101, 102, 103, 102, 104],
            "low": [99, 100, 101, 100, 102],
            "close": [100, 101, 102, 101, 103],
            "volume": [1, 1, 1, 1, 1],
        }
    )
    env = TradingEnv(df)
    env.reset()
    _, reward, *_ , info = env.step(0)

    terms = info["reward_terms"]
    assert set(terms) == {"pnl", "turnover", "drawdown", "volatility"}

    expected = (
        env.w_pnl * terms["pnl"]
        - env.w_turn * terms["turnover"]
        - env.w_dd * terms["drawdown"]
        - env.w_vol * terms["volatility"]
    )
    assert reward == pytest.approx(expected)
