import ccxt

from .time_sync import smart_time_sync


def make_binance_exchange(api_key: str, api_secret: str, *, use_testnet: bool, rate_limit_ms: int):
    """Create a fresh ccxt.binance instance with proper configuration."""
    ex = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {"adjustForTimeDifference": True},
        }
    )
    ex.rateLimit = rate_limit_ms

    if use_testnet:
        ex.urls["api"]["public"] = "https://testnet.binance.vision/api"
        ex.urls["api"]["private"] = "https://testnet.binance.vision/api"
    else:
        pass  # defaults already set for mainnet

    smart_time_sync(ex)
    return ex
