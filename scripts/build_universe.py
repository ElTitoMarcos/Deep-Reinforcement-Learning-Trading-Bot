from __future__ import annotations
import argparse, os
import pandas as pd

try:
    import ccxt  # type: ignore
except Exception as e:
    ccxt = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", type=str, default="binance")
    ap.add_argument("--quote", type=str, default="USDT")
    ap.add_argument("--min-vol-usd", type=float, default=1_000_000.0)
    ap.add_argument("--out", type=str, default="data/universe/liquid_universe.csv")
    args = ap.parse_args()

    if ccxt is None:
        raise RuntimeError("ccxt no instalado. `pip install ccxt`.")
    ex = getattr(ccxt, args.exchange)({"enableRateLimit": True})
    markets = ex.load_markets()
    rows = []
    for sym, m in markets.items():
        if not sym.endswith("/" + args.quote):
            continue
        # Placeholder liquidez: usar limits y active
        if not m.get("active", True):
            continue
        # Placeholder: si tiene info de average vol (no estándar), aquí 0
        vol_usd = 0.0
        rows.append({"symbol": sym, "active": bool(m.get("active", True)), "base": m.get("base"), "quote": m.get("quote"), "min": m.get("limits",{}).get("cost",{}).get("min")})

    df = pd.DataFrame(rows).sort_values("symbol")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Universo guardado en {args.out} con {len(df)} símbolos (placeholder).")

if __name__ == "__main__":
    main()
