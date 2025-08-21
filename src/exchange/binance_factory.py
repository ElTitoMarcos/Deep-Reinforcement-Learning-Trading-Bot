import ccxt, time


def make_public_mainnet(rate_limit_ms: int):
    ex = ccxt.binance({"enableRateLimit": True})
    ex.rateLimit = rate_limit_ms
    return ex


def make_private_mainnet(api_key: str, api_secret: str, rate_limit_ms: int):
    ex = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.rateLimit = rate_limit_ms
    _smart_time_sync(ex)
    return ex


def make_private_testnet(api_key: str, api_secret: str, rate_limit_ms: int):
    ex = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.rateLimit = rate_limit_ms
    ex.urls["api"]["public"] = "https://testnet.binance.vision/api"
    ex.urls["api"]["private"] = "https://testnet.binance.vision/api"
    _smart_time_sync(ex)
    return ex


def _smart_time_sync(ex, retries=3, delay=0.5):
    for _ in range(retries):
        try:
            ex.fetch_time()
            return
        except Exception:
            time.sleep(delay)
