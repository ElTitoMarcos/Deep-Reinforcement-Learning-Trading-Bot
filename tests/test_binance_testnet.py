import os
from src.data.ccxt_loader import get_exchange


def _api_url(exchange):
    api = exchange.urls["api"]
    if isinstance(api, dict):
        return api.get("public", "")
    return api


def test_get_exchange_respects_env(monkeypatch):
    monkeypatch.setenv("BINANCE_USE_TESTNET", "true")
    monkeypatch.setenv("BINANCE_API_KEY_TESTNET", "k")
    monkeypatch.setenv("BINANCE_API_SECRET_TESTNET", "s")
    monkeypatch.setenv("BINANCE_API_KEY_MAINNET", "km")
    monkeypatch.setenv("BINANCE_API_SECRET_MAINNET", "sm")
    ex = get_exchange()
    assert "testnet" in _api_url(ex).lower()
    monkeypatch.setenv("BINANCE_USE_TESTNET", "false")
    ex2 = get_exchange()
    assert "testnet" not in _api_url(ex2).lower()
