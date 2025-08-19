import time

import pandas as pd

from src.data import refresh_worker


def test_refresh_worker_triggers_update(monkeypatch, tmp_path):
    class DummyEx:
        id = "binance"

    monkeypatch.setattr(refresh_worker, "get_exchange", lambda: DummyEx())
    monkeypatch.setattr(refresh_worker, "last_watermark", lambda s, t: None)
    def fake_fetch(ex, sym, tf, since_ms=None):
        refresh_worker.dataset_updated.set()
        return pd.DataFrame(
            {"ts": [1], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
        )

    monkeypatch.setattr(refresh_worker, "fetch_ohlcv_incremental", fake_fetch)
    monkeypatch.setattr(refresh_worker, "upsert_parquet", lambda df, p: None)
    monkeypatch.setattr(
        refresh_worker.paths,
        "raw_parquet_path",
        lambda exch, sym, tf: tmp_path / f"{sym}-{tf}.parquet",
    )

    refresh_worker.start_refresh_worker(["BTC/USDT"], "1m")
    assert refresh_worker.dataset_updated.wait(2)
    refresh_worker.stop_refresh_worker()
