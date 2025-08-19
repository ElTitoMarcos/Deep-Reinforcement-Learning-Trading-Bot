from src.data.ccxt_loader import fetch_ohlcv


class DummyExchange:
    timeframes = {"1m": 60, "5m": 300}

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        assert timeframe == "1m"
        return [[0, 1, 1, 1, 1, 1]]


def test_fetch_ohlcv_auto_selects_smallest(caplog):
    ex = DummyExchange()
    with caplog.at_level("INFO"):
        df = fetch_ohlcv(ex, "BTC/USDT")
    assert df.attrs["timeframe"] == "1m"
    assert "selected_timeframe=1m" in caplog.text
