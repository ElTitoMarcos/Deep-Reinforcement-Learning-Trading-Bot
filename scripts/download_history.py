from __future__ import annotations
import argparse, time, os
from datetime import datetime, timezone
from src.data.ccxt_loader import get_exchange, fetch_ohlcv, simulate_1s_from_1m, save_history

def parse_since(s: str | None):
    if not s:
        return None
    # Try ISO date
    try:
        dt = datetime.fromisoformat(s.replace("Z","")).replace(tzinfo=timezone.utc)
        return int(dt.timestamp()*1000)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--timeframe", type=str, default="1m")
    ap.add_argument("--since", type=str, default=None, help="ISO date (UTC) p.ej. 2024-01-01")
    ap.add_argument("--root", type=str, default="data/raw")
    args = ap.parse_args()

    ex = get_exchange(args.exchange)
    since_ms = parse_since(args.since)
    for sym in args.symbols:
        df = fetch_ohlcv(ex, sym, timeframe=args.timeframe, since=since_ms)
        if args.timeframe == "1s" and df.empty:
            print(f"[WARN] 1s no disponible; simulando desde 1m para {sym}")
            df_1m = fetch_ohlcv(ex, sym, timeframe="1m", since=since_ms)
            df = simulate_1s_from_1m(df_1m)
        path = save_history(df, args.root, args.exchange, sym, args.timeframe)
        print(f"Guardado: {path}")

if __name__ == "__main__":
    main()
