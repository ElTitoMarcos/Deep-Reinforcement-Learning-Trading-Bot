import pandas as pd
from src.env.trading_env import TradingEnv


class DummyMeta:
    def __init__(self):
        self.calls = []

    def get_symbol_filters(self, symbol: str):
        self.calls.append(symbol)
        if symbol == "BTC/USDT":
            return {"tickSize": 0.1, "stepSize": 0.5, "minNotional": 100.0}
        return {"tickSize": 1.0, "stepSize": 1.0, "minNotional": 1000.0}


def make_df(prices):
    return pd.DataFrame(
        {
            "ts": range(len(prices)),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1.0] * len(prices),
        }
    )


def test_order_respects_filters(monkeypatch):
    df = make_df([101.23, 110.55])
    cfg = {
        "fees": {"taker": 0.0},
        "min_notional_usd": 0.0,
        "filters": {"tickSize": 0.1, "stepSize": 0.5},
    }
    meta = DummyMeta()
    monkeypatch.setattr("src.env.trading_env.estimate_slippage", lambda *a, **k: 0.0)
    env = TradingEnv(df, cfg=cfg, symbol="BTC/USDT", meta=meta)
    env.reset()
    env.step(1)
    assert env.in_position
    assert env.entry_price == 101.2
    env.step(2)
    assert not env.in_position
    assert meta.calls == ["BTC/USDT"]


def test_min_notional_blocks_order():
    df = make_df([1.0, 1.0])
    cfg = {
        "fees": {"taker": 0.0},
        "min_notional_usd": 0.0,
        "filters": {"tickSize": 0.1, "stepSize": 0.1},
    }
    meta = DummyMeta()
    meta.get_symbol_filters = lambda s: {"tickSize": 0.1, "stepSize": 0.1, "minNotional": 10.0}
    env = TradingEnv(df, cfg=cfg, symbol="BTC/USDT", meta=meta)
    env.reset()
    env.step(1)
    assert not env.in_position


def test_set_symbol_loads_new_filters():
    df = make_df([100.0, 100.0])
    cfg = {
        "fees": {"taker": 0.0},
        "min_notional_usd": 0.0,
        "filters": {"tickSize": 0.1, "stepSize": 0.1},
    }
    meta = DummyMeta()
    env = TradingEnv(df, cfg=cfg, symbol="BTC/USDT", meta=meta)
    env.set_symbol("ETH/USDT")
    assert meta.calls == ["BTC/USDT", "ETH/USDT"]
