import pandas as pd
import numpy as np
from src.env.trading_env import TradingEnv

def test_env_smoke():
    # tiny synthetic data
    df = pd.DataFrame({
        "ts": [0,1000,2000,3000,4000],
        "open": [100,101,102,101,103],
        "high": [101,102,103,102,104],
        "low": [99,100,101,100,102],
        "close": [100,101,102,101,103],
        "volume": [1,1,1,1,1],
    })
    env = TradingEnv(df)
    obs, info = env.reset()
    assert env.observation_space.shape[0] == len(obs)
    obs2, r, done, trunc, info = env.step(0)
    assert isinstance(r, float)
