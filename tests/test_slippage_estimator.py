import numpy as np
from types import SimpleNamespace

from src.risk.slippage import estimate_slippage


class DummyEx:
    def fetch_order_book(self, symbol, depth):
        # simple two-level book
        return {
            "bids": [[99.0, 5.0], [98.5, 5.0]],
            "asks": [[101.0, 5.0], [102.0, 5.0]],
        }


def test_slippage_varies_with_notional():
    ex = DummyEx()
    s1 = estimate_slippage("BTC/USDT", 50.0, "buy", exchange=ex)
    s2 = estimate_slippage("BTC/USDT", 800.0, "buy", exchange=ex)
    assert s2 > s1 > 0


def test_slippage_fallback_uses_volatility():
    class ShallowEx(DummyEx):
        def fetch_order_book(self, symbol, depth):
            # only 1 USD depth so it is insufficient
            return {"bids": [[99.0, 0.01]], "asks": [[101.0, 0.01]]}

    prices = [100, 101, 99, 102]
    vol = float(np.std(np.diff(np.log(prices))))
    s = estimate_slippage("BTC/USDT", 1000.0, "buy", exchange=ShallowEx(), prices=prices)
    assert abs(s - vol) < 1e-12
