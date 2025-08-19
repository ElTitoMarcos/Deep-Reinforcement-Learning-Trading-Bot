from __future__ import annotations
import argparse, os
import pandas as pd
from .simulator import simulate
from ..policies.router import get_policy
from ..utils.data_io import load_table, ensure_dir
from ..utils.config import load_config
from .metrics import pnl, sharpe, sortino, max_drawdown, hit_ratio, turnover
import json
from datetime import datetime
import matplotlib.pyplot as plt

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
    sim = simulate(
        df,
        pol,
        fees=cfg.get("fees", {}).get("taker", 0.001),
        slippage=cfg.get("slippage", 0.0005),
        min_notional_usd=cfg.get("min_notional_usd", 10.0),
        tick_size=cfg.get("filters", {}).get("tickSize", 0.01),
        step_size=cfg.get("filters", {}).get("stepSize", 0.0001),
    )
    equity = sim["equity"]
    trades = sim["trades"]
    rets = sim["returns"]
    equity_curve = (1.0 + rets).cumprod()

    metrics = {
        "pnl": pnl(rets),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": max_drawdown(equity_curve),
        "hit_ratio": hit_ratio(trades),
        "turnover": turnover(trades),
        "equity_final": equity,
    }

    print(
        f"Equity final: {equity:.4f}\n"
        f"PnL: {metrics['pnl']:.2%} | Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}\n"
        f"MaxDD: {metrics['max_drawdown']:.3%} | HitRatio: {metrics['hit_ratio']:.2%} | Turnover: {metrics['turnover']}"
    )

    reports_root = paths.get("reports_dir", "reports")
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(reports_root, run_id)
    ensure_dir(run_dir)

    # save metrics
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # save trades
    pd.DataFrame(trades).to_csv(os.path.join(run_dir, "trades.csv"), index=False)

    # save equity curve
    equity_curve.to_csv(os.path.join(run_dir, "equity.csv"), index_label="idx", header=["equity"])
    plt.figure()
    equity_curve.plot()
    plt.title("Equity Curve")
    plt.xlabel("trade")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "equity.png"))
    plt.close()

if __name__ == "__main__":
    main()
