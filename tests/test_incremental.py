import json
import json
import pandas as pd

from src.data import incremental
from src.utils import paths


def test_last_watermark_and_upsert(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(paths, "RAW_DIR", raw_dir)

    path = paths.raw_parquet_path("binance", "ETH/USDT", "1m")
    df1 = pd.DataFrame({
        "ts": [1, 2],
        "open": [1.0, 2.0],
        "high": [1.0, 2.0],
        "low": [1.0, 2.0],
        "close": [1.0, 2.0],
        "volume": [0.0, 0.0],
    })
    incremental.upsert_parquet(df1, path)
    assert path.exists()
    assert incremental.last_watermark("ETH/USDT", "1m") is None

    mpath = path.with_suffix(".manifest.json")
    mpath.write_text(json.dumps({"watermark": 2}))
    assert incremental.last_watermark("ETH/USDT", "1m") == 2

    df2 = pd.DataFrame({
        "ts": [2, 3],
        "open": [2.0, 3.0],
        "high": [2.0, 3.0],
        "low": [2.0, 3.0],
        "close": [2.0, 3.0],
        "volume": [0.0, 0.0],
    })
    incremental.upsert_parquet(df2, path)
    out = pd.read_parquet(path)
    assert out["ts"].tolist() == [1, 2, 3]


def test_fetch_ohlcv_incremental(monkeypatch):
    calls = []

    def fake_fetch(_ex, _sym, _tf, since=None):
        calls.append(since)
        cols = ["ts", "open", "high", "low", "close", "volume"]
        if since in (None, 0):
            data = [[i, 0, 0, 0, 0, 0] for i in range(1000)]
        elif since == 1000:
            data = [[1000, 0, 0, 0, 0, 0], [1001, 0, 0, 0, 0, 0]]
        else:
            data = []
        return pd.DataFrame(data, columns=cols)

    monkeypatch.setattr(incremental, "fetch_ohlcv", fake_fetch)
    df = incremental.fetch_ohlcv_incremental(None, "ETH/USDT", "1m", since_ms=0)
    assert df["ts"].tolist() == list(range(1002))
    assert calls == [0, 1000]
