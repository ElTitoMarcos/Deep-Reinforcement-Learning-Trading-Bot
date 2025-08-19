#!/usr/bin/env python
"""High-level experiment runner combining training and backtesting.

This utility ties together the individual components of the project so that a
single command can execute the full workflow:

1. Ensure that price data for the configured symbol is available, generating a
   small synthetic series if necessary.
2. Construct the :class:`~src.env.trading_env.TradingEnv` according to the
   provided configuration.
3. Train the requested DRL algorithm for ``N`` timesteps.
4. Backtest the resulting policy and save a report under ``reports/{exp_id}``.

The goal is not to provide a production ready experiment manager but rather a
minimal yet complete example used in the unit tests.  Only a subset of the
project's features are supported and optional heavy dependencies are avoided
where possible.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, UTC

# Ensure project root is on ``sys.path`` when executed as a script -----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:  # pragma: no cover - defensive
    sys.path.append(ROOT)

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.data_io import ensure_dir, load_table, save_table
from src.utils import paths
from src.env.trading_env import TradingEnv
from src.backtest.simulator import simulate
from src.backtest.metrics import (
    pnl,
    sharpe,
    sortino,
    max_drawdown,
    hit_ratio,
    turnover,
)
from src.training.train_drl import (
    has_sb3,
    load_data as _load_data,
    train_value_dqn,
    train_ppo_sb3,
)
from src.policies.value_based import ValueBasedPolicy
from src.utils.device import get_device, set_cpu_threads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (optionally) PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch may be unavailable
        import torch

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass


def ensure_price_data(cfg: dict, timesteps: int) -> pd.DataFrame:
    """Ensure that a price table exists for the configured market.

    If the expected file is missing a small synthetic random walk series is
    generated and persisted so future runs are deterministic.  Downloading from
    the network is intentionally avoided to keep tests fast and hermetic.
    """

    exchange = cfg.get("exchange", "binance")
    symbol = (cfg.get("symbols") or ["BTC/USDT"])[0]
    timeframe = cfg.get("timeframe", "1m")
    fname = paths.raw_parquet_path(exchange, symbol, timeframe)

    if fname.exists():
        return load_table(paths.posix(fname))

    # Fallback to synthetic data generation used in the training smoke tests.
    df = _load_data(cfg, None, timesteps)
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    df["source"] = "synthetic"
    ensure_dir(paths.posix(fname.parent))
    try:
        save_table(df, paths.posix(fname))
    except Exception:  # parquet engine missing -> store as CSV instead
        csv_path = fname.with_suffix(".csv")
        save_table(df, paths.posix(csv_path))
    return df


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate DRL agents")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--data-mode", dest="data_mode", default=None, help="Optional feature set override")
    args = parser.parse_args()

    set_seed(args.seed)

    overrides = {"data_mode": args.data_mode, "seed": args.seed}
    cfg = load_config(args.config, overrides=overrides)

    df = ensure_price_data(cfg, args.timesteps)
    env = TradingEnv(df)

    paths.ensure_dirs_exist()
    ckpt_dir = paths.checkpoints_dir()

    device = get_device()
    if device == "cuda":
        import torch  # local import to avoid requiring torch with CUDA in tests

        name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA ({name})")
    else:
        threads = set_cpu_threads()
        print(f"Using device: CPU ({threads} threads)")
    cfg.setdefault("dqn", {})["device"] = device
    cfg.setdefault("ppo", {})["device"] = device

    if args.algo == "dqn":
        model_path = train_value_dqn(env, cfg, args.timesteps, outdir=paths.posix(ckpt_dir), checkpoint_freq=0)
        policy = ValueBasedPolicy(
            int(env.observation_space.shape[0]),
            int(env.action_space.n),
            config=cfg.get("dqn", {}),
        )
        policy.load_model(model_path)
    else:  # args.algo == "ppo"
        if not has_sb3():  # pragma: no cover - heavy optional dependency
            raise RuntimeError("stable-baselines3 is required for PPO training")
        model_path = train_ppo_sb3(env, cfg, args.timesteps, outdir=paths.posix(ckpt_dir))
        from stable_baselines3 import PPO  # pragma: no cover - optional dependency

        sb3_model = PPO.load(model_path)

        class _SB3Policy:
            def __init__(self, model):
                self.model = model

            def act(self, obs):
                action, _ = self.model.predict(obs, deterministic=True)
                return int(action)

        policy = _SB3Policy(sb3_model)
    sim = simulate(
        df,
        policy,
        fees=cfg.get("fees", {}).get("taker", 0.001),
        slippage_multiplier=cfg.get("slippage_multiplier", 1.0),
        min_notional_usd=cfg.get("min_notional_usd", 10.0),
        tick_size=cfg.get("filters", {}).get("tickSize", 0.01),
        step_size=cfg.get("filters", {}).get("stepSize", 0.0001),
        symbol=(cfg.get("symbols") or ["BTC/USDT"])[0],
        slippage_depth=int(cfg.get("slippage_depth", 50)),
    )

    reports_root = paths.reports_dir()
    exp_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = reports_root / exp_id
    ensure_dir(paths.posix(run_dir))

    equity_curve = (1.0 + sim["returns"]).cumprod()
    metrics = {
        "pnl": pnl(sim["returns"]),
        "sharpe": sharpe(sim["returns"]),
        "sortino": sortino(sim["returns"]),
        "max_drawdown": max_drawdown(equity_curve),
        "hit_ratio": hit_ratio(sim["trades"]),
        "turnover": turnover(sim["trades"]),
        "equity_final": sim["equity"],
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    pd.DataFrame(sim["trades"]).to_csv(run_dir / "trades.csv", index=False)
    equity_curve.to_csv(run_dir / "equity.csv", index_label="idx", header=["equity"])

    try:  # pragma: no cover - matplotlib not essential in tests
        import matplotlib.pyplot as plt

        plt.figure()
        equity_curve.plot()
        plt.title("Equity Curve")
        plt.xlabel("trade")
        plt.ylabel("equity")
        plt.tight_layout()
          plt.savefig(run_dir / "equity.png")
        plt.close()
    except Exception:
        pass

    print(f"Experiment artifacts saved to {paths.posix(run_dir)}")


if __name__ == "__main__":  # pragma: no cover
    main()
