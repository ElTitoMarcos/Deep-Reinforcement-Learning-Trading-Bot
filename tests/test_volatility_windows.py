import time
import pandas as pd

from src.data.volatility_windows import find_high_activity_windows


class DummyExchange:
    id = "binance"

    def __init__(self):
        self.trades: list[dict] = []

    def seed(self, start: int, counts):
        for h, count in counts.items():
            base = start + h * 3600000
            for i in range(count):
                self.trades.append({"timestamp": base + i})

    def fetch_trades(self, symbol, since=None, limit=1000):
        since = since or 0
        res = [t for t in self.trades if t["timestamp"] >= since]
        return res[:limit]


def test_selects_top_volatility_windows(tmp_path, monkeypatch):
    end = int(time.time() * 1000)
    start = end - 3 * 3600000
    rows = []
    for i in range(60):
        ts = start + i * 60000
        rows.append([ts, 100, 100, 100, 100, 1])
    for i in range(60, 120):
        ts = start + i * 60000
        price = 100 + (i - 60)
        rows.append([ts, price, price, price, price, 10])
    for i in range(120, 180):
        ts = start + i * 60000
        rows.append([ts, 100, 100, 100, 100, 5])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    path = tmp_path / "binance_BTC-USDT_1m.parquet"
    df.to_parquet(path)

    def fake_ensure(exchange, symbol, timeframe, hours, root=None):
        return path

    monkeypatch.setattr(
        "src.data.volatility_windows.ensure_ohlcv", fake_ensure
    )

    ex = DummyExchange()
    ex.seed(start, {0: 10, 1: 100, 2: 30})

    windows, lookback = find_high_activity_windows(
        ["BTC/USDT"], 1, target_hours=2, lookback_hours=3, exchange=ex
    )

    assert lookback == 3
    hour_start = (start // 3600000) * 3600000
    expected = (hour_start + 3600000, hour_start + 3 * 3600000)
    assert windows == [expected]

