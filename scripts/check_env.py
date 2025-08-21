from dotenv import load_dotenv, find_dotenv
import os, ccxt, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils.credentials import load_binance_creds, load_openai_key

p = find_dotenv(usecwd=True); load_dotenv(p, override=True)
print("[.env]", p or "NO ENCONTRADO")

for k in [
    "BINANCE_USE_TESTNET",
    "BINANCE_API_KEY_MAINNET","BINANCE_API_SECRET_MAINNET",
    "BINANCE_API_KEY_TESTNET","BINANCE_API_SECRET_TESTNET",
    "API_KEY","API_SECRET",
    "OPENAI_API_KEY",
]:
    v = os.getenv(k); print(f"{k:28s}", "OK" if (v and v.strip()) else "MISSING/EMPTY")

try:
    key, sec, use_testnet = load_binance_creds()
    ex = ccxt.binance({"apiKey": key, "secret": sec, "enableRateLimit": True})
    if use_testnet:
        ex.urls["api"]["public"]  = "https://testnet.binance.vision/api"
        ex.urls["api"]["private"] = "https://testnet.binance.vision/api"
    sym = ex.fetch_ticker("BTC/USDT")["symbol"]
    print("[binance] OK,", "testnet" if use_testnet else "mainnet", "ticker:", sym)
except Exception as e:
    print("[binance] ERROR:", repr(e))

try:
    k = load_openai_key()
    print("[openai] OK (key len)", len(k))
except Exception as e:
    print("[openai] ERROR:", repr(e))
