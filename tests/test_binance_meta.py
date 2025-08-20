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

def test_symbol_filters_cached(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=10):
        calls.append(1)

        return DummyResp(
            {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [
                            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr('src.exchange.binance_meta.requests.get', fake_get)
    meta = BinanceMeta("k", "s")
    f1 = meta.get_symbol_filters("BTCUSDT")
    f2 = meta.get_symbol_filters("BTCUSDT")
    assert calls.count(1) == 1
    assert f1 == f2 == {"tickSize": 0.01, "stepSize": 0.001, "minNotional": 10.0}

def test_binance_meta_fallback(monkeypatch):
    def raise_exc(*args, **kwargs):
        raise Exception("boom")
    monkeypatch.setattr('src.exchange.binance_meta.requests.get', raise_exc)
    meta = BinanceMeta("k", "s")
    fees = meta.get_account_trade_fees()
    assert fees["SPOT"]["maker"] == 0.001
    assert fees["SPOT"]["taker"] == 0.001
    assert meta.last_fee_origin == "Fuente: Fallback"


def test_binance_meta_testnet_404(monkeypatch, caplog):
    class Dummy404:
        status_code = 404

        def json(self):
            return {}

    def fake_get(*args, **kwargs):
        return Dummy404()

    monkeypatch.setattr('src.exchange.binance_meta.requests.get', fake_get)
    meta = BinanceMeta("k", "s", use_testnet=True)
    with caplog.at_level("INFO"):
        fees = meta.get_account_trade_fees()
    assert fees["SPOT"]["maker"] == 0.001
    assert meta.last_fee_origin == "Fuente: Fallback testnet"
    assert "Testnet sin endpoint tradeFee" in caplog.text
