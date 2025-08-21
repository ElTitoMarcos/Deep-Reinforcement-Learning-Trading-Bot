import os
from pathlib import Path

import pandas as pd

from src.data import microv5_loader
from src.data.microv5_loader import MicroV5Collector


class DummyEx:
    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def fetch_order_book(self, symbol, limit=20):
        return {"bids": [[99, 1]], "asks": [[101, 1]]}


def test_collector_step_and_flush(tmp_path, monkeypatch):
    # Avoid lot size lookups
    monkeypatch.setattr(microv5_loader, "fetch_lot_size_meta", lambda ex, sym: (0.0, 0.0, 0.0))

    ex = DummyEx()
    collector = MicroV5Collector(ex, "BTC/USDT", out_dir=tmp_path, flush_every_n=1)
    collector.step()

    files = list(tmp_path.glob("BTCUSDT/micro_v5_*.parquet"))
    assert files, "parquet not created"
    df = pd.read_parquet(files[0])
    assert df.loc[0, "best_bid"] == 99
    assert df.loc[0, "best_ask"] == 101
    assert df.loc[0, "spread"] == 2
    assert df.loc[0, "mid"] == 100
