from src.auto import AlgoController


def test_high_vol_and_td_error_mapping():
    ctrl = AlgoController()
    mapping = ctrl.decide(
        {"intraminute": True},
        {"volatility": 0.05},
        {"td_error": 2.0},
    )
    assert mapping["entries_exits"] == "dqn"
    assert mapping["risk_limits"] == "ppo"
    assert mapping["position_sizing"] == "ppo"
