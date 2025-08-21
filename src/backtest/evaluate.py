from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
import os
_DOTENV = find_dotenv(usecwd=True)
load_dotenv(_DOTENV, override=True)
if __name__ == "__main__" or os.getenv("DEBUG_DOTENV") == "1":
    print(f"[.env] Cargado: {_DOTENV or 'NO ENCONTRADO'}")

import argparse
from pathlib import Path
import pandas as pd
from .simulator import simulate
from ..policies.router import get_policy
from ..policies.hybrid import HybridPolicy
from ..policy import HybridRuntime
from ..auto import AlgoController
from ..utils.data_io import load_table
from ..utils.config import load_config
from ..utils import paths
from ..reports.human_friendly import write_readme
from ..utils.device import get_device, set_cpu_threads
from ..data.ensure import ensure_ohlcv
from ..utils import exp_log

from .metrics import pnl, sharpe, sortino, max_drawdown, hit_ratio, turnover
import json
from datetime import datetime, UTC
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--policy", type=str, default="deterministic")
    ap.add_argument(
        "--hybrid",
        action="store_true",
        help="Usa HybridRuntime para combinar DQN y PPO",
    )
    ap.add_argument("--data", type=str, default=None, help="Ruta a parquet/csv si quieres pasar datos manualmente")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths.ensure_dirs_exist()
    exchange = cfg.get("exchange", "binance")
    symbol = (cfg.get("symbols") or ["BTC/USDT"])[0]
    timeframe = cfg.get("timeframe", "1m")

    if args.data:
        df = load_table(paths.posix(Path(args.data)))
    else:
        path = paths.raw_parquet_path(exchange, symbol, timeframe)
        if not path.exists():
            try:
                ensure_ohlcv(exchange, symbol, timeframe)
            except Exception as exc:
                print(f"ensure_ohlcv failed: {exc}")
        if not path.exists():
            alt = path.with_suffix(".csv")
            path = alt if alt.exists() else path
        df = load_table(paths.posix(path))

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

    algo_map = cfg.get("algo_weights") or {
        "dqn": 1.0 if (args.policy.lower() == "hybrid" or args.hybrid) else 0.0,
        "ppo": 1.0 if (args.policy.lower() == "hybrid" or args.hybrid) else 0.0,
    }
    cfg_snapshot = {
        "algo_map": algo_map,
        "reward_weights": {
            "w_pnl": cfg.get("reward_weights", {}).get("pnl"),
            "w_drawdown": cfg.get("reward_weights", {}).get("dd"),
            "w_volatility": cfg.get("reward_weights", {}).get("vol"),
            "w_turnover": cfg.get("reward_weights", {}).get("turn"),
        },
        "hparams": {k: cfg.get(k) for k in ("dqn", "ppo") if cfg.get(k)},
        "data_windows": {},
        "device": device,
        "notes_llm": cfg.get("notes_llm", ""),
    }
    run_id = exp_log.log_run_start(cfg_snapshot)

    def _pk(name: str):
        return {"config": {"device": device}} if name in {"value_based", "dqn", "value", "value-based"} else {}
    use_hybrid = False

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
        use_hybrid = args.hybrid or cfg.get("algo") == "hybrid" or cfg.get("runtime") == "hybrid"

        if use_hybrid:
            class _NoRiskControl:
                def __init__(self, base):
                    self.base = base

                def act(self, obs):
                    return self.base.act(obs)

                def filter(self, signal, obs):
                    return signal

            dqn_pol = get_policy("value_based", **_pk("value_based"))
            ppo_pol = _NoRiskControl(pol)
            controller = AlgoController()
            pol = HybridRuntime(dqn_pol, ppo_pol, controller)

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

    signal_stats = {"hit_ratio": hit_ratio(trades), "trades": len(trades)}
    control_stats = {
        "max_drawdown": max_drawdown(equity_curve),
        "turnover": turnover(trades),
    }

    metrics = {
        "pnl": pnl(rets),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": control_stats["max_drawdown"],
        "hit_ratio": signal_stats["hit_ratio"],
        "turnover": control_stats["turnover"],
        "equity_final": equity,
    }
    if use_hybrid:
        metrics["signal"] = signal_stats
        metrics["control"] = control_stats

    final_metrics = {
        "pnl": metrics["pnl"],
        "dd": metrics["max_drawdown"],
        "hit": metrics["hit_ratio"],
        "sharpe_simple": metrics["sharpe"],
        "turnover": metrics["turnover"],
    }
    exp_log.log_run_update(run_id, final_metrics)
    exp_log.log_run_end(run_id, final_metrics)

    print(
        f"Equity final: {equity:.4f}\n"
        f"PnL: {metrics['pnl']:.2%} | Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}\n"
        f"MaxDD: {control_stats['max_drawdown']:.3%} | HitRatio: {signal_stats['hit_ratio']:.2%} | Turnover: {control_stats['turnover']}"
    )
    if use_hybrid:
        hits = int(signal_stats["hit_ratio"] * signal_stats["trades"])
        print(f"DQN acierta {hits}/{signal_stats['trades']} entradas; PPO reduce DD a {control_stats['max_drawdown']:.2%}")

    reports_root = paths.reports_dir()
    run_stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = reports_root / run_stamp
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
