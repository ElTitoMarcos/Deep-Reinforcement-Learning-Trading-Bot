from src.policies.hybrid import HybridPolicy


class Dummy:
    def __init__(self, val: int):
        self.val = val

    def act(self, obs):
        return self.val


def test_record_trade_updates_weights():
    pol = HybridPolicy({"a": Dummy(0), "b": Dummy(1)}, {"a": 0.5, "b": 0.5}, block_size=2)
    m = {
        "a": {"pnl": 1.0, "max_drawdown": 0.1},
        "b": {"pnl": 0.0, "max_drawdown": 0.2},
    }
    pol.record_trade(m)
    before = pol.weights.copy()
    pol.record_trade(m)
    assert pol.weights["a"] > pol.weights["b"]
    assert pol.weights != before
