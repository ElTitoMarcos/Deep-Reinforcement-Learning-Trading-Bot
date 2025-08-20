from src.data.symbol_discovery import discover_symbols, discover_summary


class DummyEx:
    def fetch_tickers(self):
        return {
            "BTC/USDT": {"quoteVolume": 100000},
            "ETH/USDT": {"quoteVolume": 90000},
            "USDC/USDT": {"quoteVolume": 80000},
            "LTC/USDT": {"quoteVolume": 50000},
            "BTC/USDC": {"quoteVolume": 120},
            "XRP/USDT": {"quoteVolume": 70000},
            "ETHUP/USDT": {"quoteVolume": 60000},
            "BNB/USDT": {"quoteVolume": 65000},
            "FOO/USDT": {"quoteVolume": 10},
        }


def test_discover_symbols_filters_and_sorts():
    ex = DummyEx()
    syms = discover_symbols(ex, top_n=5)
    assert syms[0] == "BTC/USDT"
    assert "USDC/USDT" not in syms  # filtered stablecoin
    assert "ETHUP/USDT" not in syms  # filtered exotic suffix
    assert len(syms) == 5
    assert syms == [
        "BTC/USDT",
        "ETH/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "LTC/USDT",
    ]


def test_discover_summary_format():
    syms = ["BTC/USDT", "ETH/USDT"]
    summary = discover_summary(syms)
    assert summary.startswith(
        "Top 2 por volumen USDT, excluidos stablecoins, actualizado"
    )
    assert summary.endswith("UTC")
