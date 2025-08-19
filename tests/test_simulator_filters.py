import logging
import pandas as pd

from src.backtest.simulator import simulate


class DummyPolicy:
    def __init__(self):
        self.calls = 0
    def act(self, obs):
        self.calls += 1
        if self.calls == 1:
            return 1  # open
        if self.calls == 2:
            return 2  # close
        return 0


class DummyMeta:
    def get_symbol_filters(self, symbol: str):
        return {"tickSize": 0.1, "stepSize": 0.5, "minNotional": 50.0}


def test_simulator_logs_filters(monkeypatch, caplog):
    df = pd.DataFrame({"close": [100.0, 101.0]})
    monkeypatch.setattr(
        "src.backtest.simulator.estimate_slippage", lambda *a, **k: 0.01
    )
    pol = DummyPolicy()
    meta = DummyMeta()
    with caplog.at_level(logging.INFO):
        sim = simulate(df, pol, fees=0.0, symbol="BTC/USDT", meta=meta)
    assert len(sim["trades"]) == 1
    assert any(
        "slippage=0.010000" in r.message
        and "tick=0.100000" in r.message
        and "step=0.500000" in r.message
        and "minNotional=50.000000" in r.message
        for r in caplog.records
    )


def test_simulator_blocks_min_notional(monkeypatch):
    df = pd.DataFrame({"close": [1.0, 1.0]})
    class MetaHigh:
        def get_symbol_filters(self, symbol: str):
            return {"tickSize": 0.1, "stepSize": 0.1, "minNotional": 10.0}
    pol = DummyPolicy()
    meta = MetaHigh()
    monkeypatch.setattr(
        "src.backtest.simulator.estimate_slippage", lambda *a, **k: 0.0
    )
    sim = simulate(df, pol, fees=0.0, symbol="BTC/USDT", meta=meta)
    assert sim["trades"] == []
