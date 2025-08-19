import types
from src.exchange.binance_meta import BinanceMeta


class DummyResp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_binance_meta_parses_response(monkeypatch):
    sample = [{"symbol": "BTCUSDT", "makerCommission": 0.001, "takerCommission": 0.002}]
    def fake_get(url, params=None, headers=None, timeout=10):
        return DummyResp(sample)
    monkeypatch.setattr('src.exchange.binance_meta.requests.get', fake_get)
    meta = BinanceMeta("k", "s")
    fees = meta.get_account_trade_fees()
    assert fees["BTCUSDT"]["taker"] == 0.002
    assert fees["BTCUSDT"]["maker"] == 0.001


def test_binance_meta_fallback(monkeypatch):
    def raise_exc(*args, **kwargs):
        raise Exception("boom")
    monkeypatch.setattr('src.exchange.binance_meta.requests.get', raise_exc)
    class DummyEx:
        fees = {"trading": {"maker": 0.005, "taker": 0.006}}
    monkeypatch.setattr('src.exchange.binance_meta.get_exchange', lambda **kwargs: DummyEx())
    meta = BinanceMeta("k", "s")
    fees = meta.get_account_trade_fees()
    assert fees["SPOT"]["maker"] == 0.005
    assert fees["SPOT"]["taker"] == 0.006
