from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .simulator import simulate
from ..policies.router import get_policy
from ..policies.hybrid import HybridPolicy
from ..utils.data_io import load_table
from ..utils.config import load_config
from ..utils.paths import get_raw_dir, get_reports_dir, ensure_dirs_exist, raw_parquet_path
from ..reports.human_friendly import write_readme
from ..utils.device import get_device, set_cpu_threads
from ..data.ensure import ensure_ohlcv

from .metrics import pnl, sharpe, sortino, max_drawdown, hit_ratio, turnover
import json
from datetime import datetime, timezone
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--policy", type=str, default="deterministic")
    ap.add_argument("--data", type=str, default=None, help="Ruta a parquet/csv si quieres pasar datos manualmente")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs_exist(cfg)
    data_root = get_raw_dir(cfg)
    exchange = cfg.get("exchange", "binance")
    symbol = (cfg.get("symbols") or ["BTC/USDT"])[0]
    timeframe = cfg.get("timeframe", "1m")

    if args.data:
        df = load_table(Path(args.data).as_posix())
    else:
        path = raw_parquet_path(exchange, symbol, timeframe, data_root)
        if not path.exists():
            try:
                ensure_ohlcv(exchange, symbol, timeframe, root=data_root)
            except Exception as exc:
                print(f"ensure_ohlcv failed: {exc}")
        if not path.exists():
            alt = path.with_suffix(".csv")
            path = alt if alt.exists() else path
        df = load_table(path.as_posix())

    fee = cfg.get("fees", {}).get("taker", 0.001)
    print(f"Using fees: {cfg.get('fees', {})}")

    device = get_device()
    if device == "cuda":
        import torch

        name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA ({name})")
    else:
        threads = set_cpu_threads()
        print(f"Using device: CPU ({threads} threads)")

    def _pk(name: str):
        return {"config": {"device": device}} if name in {"value_based", "dqn", "value", "value-based"} else {}

    if args.policy.lower() == "hybrid":
        names = ["deterministic", "stochastic", "value_based"]
        defaults = [0.5, 0.3, 0.2]
        policies = {}
        init_w = {}

        for n, w in zip(names, defaults):
            try:
                policies[n] = get_policy(n, **_pk(n))
                init_w[n] = w
            except Exception:
                pass
        pol = HybridPolicy(policies, init_w)

        block_size = int(cfg.get("hybrid_block_size", 1440))
        for start in range(0, len(df), block_size):
            block = df.iloc[start : start + block_size]
            metrics_block = {}
            for n in pol.policies:
                sim_b = simulate(
                    block,
                    get_policy(n, **_pk(n)),
                    fees=fee,
                    slippage_multiplier=cfg.get("slippage_multiplier", 1.0),
                    slippage_static=cfg.get("slippage_static", 0.0),
                    min_notional_usd=cfg.get("min_notional_usd", 10.0),
                    tick_size=cfg.get("filters", {}).get("tickSize", 0.01),
                    step_size=cfg.get("filters", {}).get("stepSize", 0.0001),
                    symbol=symbol,
                    slippage_depth=int(cfg.get("slippage_depth", 50)),
                )
                rets_b = sim_b["returns"]
                eq_b = (1.0 + rets_b).cumprod()
                metrics_block[n] = {
                    "pnl": pnl(rets_b),
                    "max_drawdown": max_drawdown(eq_b),
                }
            pol.update_weights(metrics_block)
            print(f"Weights after block {start // block_size + 1}: {pol.weights}")

        sim = simulate(
            df,
            pol,
            fees=fee,
            slippage_multiplier=cfg.get("slippage_multiplier", 1.0),
            slippage_static=cfg.get("slippage_static", 0.0),
            min_notional_usd=cfg.get("min_notional_usd", 10.0),
            tick_size=cfg.get("filters", {}).get("tickSize", 0.01),
            step_size=cfg.get("filters", {}).get("stepSize", 0.0001),
            symbol=symbol,
            slippage_depth=int(cfg.get("slippage_depth", 50)),
        )
    else:
        pol = get_policy(args.policy, **_pk(args.policy))
        sim = simulate(
            df,
            pol,
            fees=fee,
            slippage_multiplier=cfg.get("slippage_multiplier", 1.0),
            slippage_static=cfg.get("slippage_static", 0.0),
            min_notional_usd=cfg.get("min_notional_usd", 10.0),
            tick_size=cfg.get("filters", {}).get("tickSize", 0.01),
            step_size=cfg.get("filters", {}).get("stepSize", 0.0001),
            symbol=symbol,
            slippage_depth=int(cfg.get("slippage_depth", 50)),
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

    reports_root = get_reports_dir(cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = reports_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # human friendly README
    write_readme(metrics, run_dir)

    # save trades
    pd.DataFrame(trades).to_csv(run_dir / "trades.csv", index=False)

    # save equity curve
    equity_curve.to_csv(run_dir / "equity.csv", index_label="idx", header=["equity"])
    plt.figure()
    equity_curve.plot()
    plt.title("Equity Curve")
    plt.xlabel("trade")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(run_dir / "equity.png")
    plt.close()

if __name__ == "__main__":
    main()
