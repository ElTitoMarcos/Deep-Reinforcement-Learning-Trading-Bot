from src.data.symbol_guard import validate_symbols


class DummyExchange:
    def __init__(self):
        self._markets = {
            "BTC/USDT": {"info": {"status": "TRADING"}, "spot": True},
            "ABC/USDT": {"info": {"status": "BREAK"}, "spot": True},
            "NONSPOT/USDT": {"info": {"status": "TRADING"}, "spot": False},
        }

    def load_markets(self):
        return self._markets


def test_validate_symbols_basic():
    ex = DummyExchange()
    symbols = ["BTC/USDT", "ABC/USDT", "NONSPOT/USDT", "BTC/USTD"]
    valid, invalid = validate_symbols(ex, symbols)
    assert valid == ["BTC/USDT"]
    reasons = {entry["symbol"]: entry for entry in invalid}
    assert reasons["ABC/USDT"]["reason"] == "status=BREAK"
    assert reasons["NONSPOT/USDT"]["reason"] == "no en spot"
    assert reasons["BTC/USTD"]["reason"].startswith("no existe")
    assert reasons["BTC/USTD"].get("suggest") == "BTC/USDT"
