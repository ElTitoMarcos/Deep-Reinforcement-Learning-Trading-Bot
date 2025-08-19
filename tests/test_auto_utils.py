from src.auto import choose_algo, tune


def test_choose_algo_hybrid():
    stats = {"ppo_sharpe": 1.6, "dqn_sharpe": 1.0, "ppo_maxdd": 0.2, "dqn_maxdd": 0.1}
    env_caps = {"obs_type": "continuous", "action_type": "discrete", "state_space": 100}
    res = choose_algo(stats, env_caps)
    assert res["algo"] == "hybrid"
    assert "PPO" in res["reason"]


def test_tune_basic_and_hybrid():
    cfg = tune("ppo", None, None)
    assert {"learning_rate", "batch_size", "n_steps"} <= set(cfg)
    cfg_h = tune("hybrid", None, None)
    assert "ppo" in cfg_h and "dqn" in cfg_h
