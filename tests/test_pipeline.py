import pandas as pd

from src.data import pipeline


def test_prepare_data_runs_all(monkeypatch):
    calls = []

    def fake_discover(ex, top_n=20):
        calls.append("discover")
        return ["BTC/USDT"]

    def fake_meta(symbols):
        calls.append("meta")
        return {"BTC/USDT": {}}

    def fake_validate_meta(meta):
        calls.append("v_meta")

    def fake_fetch_ohlcv(ex, sym, tf):
        calls.append("fetch")
        return pd.DataFrame()

    def fake_validate_ohlcv(df):
        calls.append("v_ohlcv")

    def fake_ensure(exch, sym, tf):
        calls.append("ensure")

    def fake_update(symbols, timeframe):
        calls.append("update")

    def fake_start(symbols, timeframe, every=5):
        calls.append("start")

    monkeypatch.setattr(pipeline, "discover_symbols", fake_discover)
    monkeypatch.setattr(pipeline, "fetch_symbol_metadata", fake_meta)
    monkeypatch.setattr(pipeline, "validate_metadata", fake_validate_meta)
    monkeypatch.setattr(pipeline, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr(pipeline, "validate_ohlcv", fake_validate_ohlcv)
    monkeypatch.setattr(pipeline, "ensure_ohlcv", fake_ensure)
    monkeypatch.setattr(pipeline, "update_all", fake_update)
    monkeypatch.setattr(pipeline, "start_refresh_worker", fake_start)
    monkeypatch.setattr(pipeline, "get_exchange", lambda: None)

    pipeline.prepare_data()

    assert calls == [
        "discover",
        "meta",
        "v_meta",
        "fetch",
        "v_ohlcv",
        "ensure",
        "update",
        "start",
    ]
