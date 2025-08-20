import pytest

from src.policy import HybridRuntime
from src.auto import AlgoController


class DummyDQN:
    def act(self, obs):
        return {"side": "buy", "qty": 0.05, "price": 100.0}


class DummyPPO:
    def __init__(self, min_notional, slippage):
        self.min_notional = min_notional
        self.slippage = slippage
        self.act_called = False

    def act(self, obs):
        self.act_called = True
        return {"side": "sell"}

    def filter(self, signal, obs):
        qty = max(signal["qty"], self.min_notional / signal["price"])
        price = signal["price"] * (1 + self.slippage)
        return {**signal, "qty": qty, "price": price}


def test_hybrid_runtime_applies_control():
    controller = AlgoController()
    ppo = DummyPPO(min_notional=10.0, slippage=0.01)
    runtime = HybridRuntime(DummyDQN(), ppo, controller)
    action = runtime.act(
        {"price": 100.0},
        stage_info={"intraminute": True},
        data_profile={"volatility": 0.05},
        stability={"td_error": 2.0},
    )
    assert action["qty"] == pytest.approx(0.1)
    assert action["price"] == pytest.approx(101.0)
    assert action["side"] == "buy"
    assert not ppo.act_called
