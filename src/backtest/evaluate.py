from __future__ import annotations
import argparse, os
import pandas as pd
from .simulator import simulate
from ..policies.router import get_policy
from ..utils.data_io import load_table
from ..utils.config import load_config
from .metrics import sharpe, sortino, max_drawdown, turnover

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--policy", type=str, default="deterministic")
    ap.add_argument("--data", type=str, default=None, help="Ruta a parquet/csv si quieres pasar datos manualmente")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    data_root = paths.get("raw_dir", "data/raw")
    exchange = cfg.get("exchange", "binance")
    symbol = (cfg.get("symbols") or ["BTC/USDT"])[0]
    timeframe = cfg.get("timeframe", "1m")

    if args.data:
        df = load_table(args.data)
    else:
        path = os.path.join(data_root, exchange, symbol.replace("/","-"), f"{timeframe}.parquet")
        df = load_table(path)

    pol = get_policy(args.policy)
    sim = simulate(df, pol, fees=cfg.get("fees",{}).get("taker",0.001), slippage=cfg.get("slippage",0.0005),
                   min_notional_usd=cfg.get("min_notional_usd",10.0),
                   tick_size=cfg.get("filters",{}).get("tickSize",0.01),
                   step_size=cfg.get("filters",{}).get("stepSize",0.0001))
    equity = sim["equity"]
    trades = sim["trades"]
    rets = sim["returns"]
    equity_curve = (1.0 + rets).cumprod()
    print(f"Equity final: {equity:.4f}")
    print(f"Sharpe: {sharpe(rets):.3f} | Sortino: {sortino(rets):.3f} | MaxDD: {max_drawdown(equity_curve):.3%} | Turnover: {turnover(trades)}")

if __name__ == "__main__":
    main()
