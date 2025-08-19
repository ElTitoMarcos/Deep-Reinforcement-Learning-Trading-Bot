import pandas as pd
from src.data.ccxt_loader import simulate_1s_from_1m

def test_simulate_1s_from_1m():
    df_1m = pd.DataFrame({
        "ts":[0,60000],
        "open":[100,101],
        "high":[101,102],
        "low":[99,100],
        "close":[101,102],
        "volume":[60,60],
    })
    df_1s = simulate_1s_from_1m(df_1m)
    assert len(df_1s) == 120
    assert set(df_1s.columns) == set(["ts","open","high","low","close","volume"])
