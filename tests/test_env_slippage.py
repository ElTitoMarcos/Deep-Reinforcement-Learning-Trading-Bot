import logging
import pandas as pd

from src.env.trading_env import TradingEnv


def test_env_logs_slippage(monkeypatch, caplog):
    df = pd.DataFrame({
        "open": [100, 101, 101],
        "high": [100, 101, 101],
        "low": [100, 101, 101],
        "close": [100, 101, 101],
    })
    cfg = {
        "fees": {"taker": 0.0},
        "filters": {"tickSize": 0.01, "stepSize": 1.0},
        "slippage_multiplier": 2.0,
    }

    def fake_slippage(symbol, notional, side, depth=50, prices=None, exchange=None):
        return 0.01

    monkeypatch.setattr("src.env.trading_env.estimate_slippage", fake_slippage)
    env = TradingEnv(df, cfg=cfg, symbol="BTC/USDT")

    with caplog.at_level(logging.INFO):
        env.step(1)  # open
    assert any(
        "slippage=0.020000" in r.message
        and "tick=0.010000" in r.message
        and "step=1.000000" in r.message
        and "minNotional=0.000000" in r.message
        for r in caplog.records
    )


def test_env_slippage_fallback(monkeypatch, caplog):
    df = pd.DataFrame({
        "open": [100, 101],
        "high": [100, 101],
        "low": [100, 101],
        "close": [100, 101],
    })
    cfg = {
        "fees": {"taker": 0.0},
        "filters": {"tickSize": 0.01, "stepSize": 1.0},
        "slippage_multiplier": 1.0,
        "slippage_static": 0.05,
    }

    def raise_slip(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.env.trading_env.estimate_slippage", raise_slip)
    env = TradingEnv(df, cfg=cfg, symbol="BTC/USDT")

    with caplog.at_level(logging.INFO):
        env.step(1)
    assert any("slippage=0.050000" in r.message for r in caplog.records)
