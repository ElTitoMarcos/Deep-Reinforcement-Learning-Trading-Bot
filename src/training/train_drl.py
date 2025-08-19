"""Minimal training script for simple DRL agents.

The goal of this module is to provide a tiny yet complete example of how to
train a Deep Reinforcement Learning agent in the context of the project.  It
supports a small value-based agent implemented purely with :mod:`numpy` and a
fallback to :mod:`stable_baselines3` for PPO when available.

Example
-------
Running a short training session with the built-in DQN::

    python -m src.training.train_drl --config configs/default.yaml --algo dqn --timesteps 5000

The same entry point is exposed via :mod:`scripts.train` for convenience.
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from ..env.trading_env import TradingEnv
from ..utils.config import load_config
from ..utils.data_io import load_table
from ..utils.logging import ensure_logger
from ..policies.value_based import ValueBasedPolicy


# ---------------------------------------------------------------------------
# Tiny DQN -----------------------------------------------------------------


@dataclass
class DQNParams:
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 10_000
    batch_size: int = 64


class TinyDQN:
    """Very small linear DQN implemented with :mod:`numpy`.

    The network consists of a single linear layer ``obs_dim × n_actions``.  It
    is intentionally simplistic but sufficient for demonstration and unit tests
    without requiring heavy dependencies such as PyTorch.
    """

    def __init__(self, obs_dim: int, n_actions: int, params: DQNParams):
        self.W = np.zeros((obs_dim, n_actions), dtype=np.float32)
        self.gamma = params.gamma
        self.lr = params.lr
        self.buffer_size = params.buffer_size
        self.batch_size = params.batch_size
        self.memory: list[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.n_actions = n_actions

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(self.n_actions))
        q_vals = obs @ self.W
        return int(np.argmax(q_vals))

    def remember(self, s, a, r, s2, done) -> None:
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        self.memory.append((s, a, r, s2, done))

    def update(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        idxs = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in idxs]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        q_next = np.max(next_states @ self.W, axis=1)
        targets = rewards + self.gamma * (1 - dones.astype(np.float32)) * q_next
        q_vals = states @ self.W
        q_sa = q_vals[np.arange(self.batch_size), actions]
        diff = q_sa - targets
        for i in range(self.batch_size):
            self.W[:, actions[i]] -= self.lr * diff[i] * states[i]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W=self.W)


# ---------------------------------------------------------------------------
# Helper functions ----------------------------------------------------------


def has_sb3() -> bool:
    try:  # pragma: no cover - optional dependency
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:  # pragma: no cover - optional dependency
        return False


def load_data(cfg: dict, data_path: str | None, timesteps: int) -> pd.DataFrame:
    """Load price data or generate synthetic series if not available."""

    if data_path:
        return load_table(data_path)

    paths = cfg.get("paths", {})
    raw_dir = paths.get("raw_dir", "data/raw")
    exchange = cfg.get("exchange", "binance")
    symbol = (cfg.get("symbols") or ["BTC/USDT"])[0]
    timeframe = cfg.get("timeframe", "1m")
    fname = os.path.join(raw_dir, exchange, symbol.replace("/", "-"), f"{timeframe}.parquet")

    if os.path.exists(fname):  # pragma: no branch - depends on repo data
        return load_table(fname)

    # generate simple random walk prices
    n = max(timesteps + 1, 1_000)
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.normal(0, 1, size=n)) + 100.0
    df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices})
    return df


def quick_eval(env: TradingEnv, agent: TinyDQN) -> float:
    """Run a fast evaluation episode and return final equity."""

    eval_env = TradingEnv(env.df)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action = agent.act(obs, epsilon=0.0)
        obs, _, done, _, _ = eval_env.step(action)
    return float(eval_env.equity)


def train_value_dqn(
    env: TradingEnv,
    cfg: dict,
    timesteps: int,
    *,
    outdir: str = "checkpoints",
    checkpoint_freq: int = 10,
) -> str:
    """Train the PyTorch value-based policy with the flexible MLP network."""

    dqn_cfg = cfg.get("dqn", {})
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    agent = ValueBasedPolicy(obs_dim, n_actions, config=dqn_cfg)

    total_steps = 0
    episode = 0
    os.makedirs(outdir, exist_ok=True)
    while total_steps < timesteps:
        obs, _ = env.reset()
        done = False
        episode += 1
        while not done and total_steps < timesteps:
            action = agent.act(obs)
            next_obs, reward, done, trunc, _info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done or trunc)
            agent.train_step()
            obs = next_obs
            total_steps += 1

        if checkpoint_freq and episode % checkpoint_freq == 0:
            ckpt = os.path.join(outdir, f"vdqn_ep{episode}.pt")
            agent.save(ckpt)

    final_path = os.path.join(outdir, "vdqn_final.pt")
    agent.save(final_path)
    return final_path


def train_dqn(
    env: TradingEnv,
    cfg: dict,
    timesteps: int,
    *,
    outdir: str = "checkpoints",
    checkpoint_freq: int = 10,
) -> str:
    """Train the tiny DQN agent with episode/step loops."""

    dqn_cfg = cfg.get("dqn", {})
    params = DQNParams(
        gamma=dqn_cfg.get("gamma", 0.99),
        lr=dqn_cfg.get("learning_rate", 1e-3),
        buffer_size=dqn_cfg.get("buffer_size", 10_000),
        batch_size=dqn_cfg.get("batch_size", 64),
    )
    agent = TinyDQN(env.observation_space.shape[0], env.action_space.n, params)

    eps_start = dqn_cfg.get("epsilon_start", 1.0)
    eps_end = dqn_cfg.get("epsilon_end", 0.05)
    eps_decay_steps = dqn_cfg.get("epsilon_decay_steps", timesteps // 2 or 1)

    total_steps = 0
    episode = 0
    while total_steps < timesteps:
        obs, _ = env.reset()
        done = False
        episode += 1
        while not done and total_steps < timesteps:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * (total_steps / eps_decay_steps))
            action = agent.act(obs, eps)
            next_obs, reward, done, trunc, _info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done or trunc)
            agent.update()
            obs = next_obs
            total_steps += 1

        if checkpoint_freq and episode % checkpoint_freq == 0:
            ckpt = os.path.join(outdir, f"dqn_ep{episode}.npz")
            agent.save(ckpt)
            equity = quick_eval(env, agent)
            print(f"[checkpoint] episode={episode} equity={equity:.2f}")

    final_path = os.path.join(outdir, "dqn_final.npz")
    agent.save(final_path)
    meta = {
        "algo": "dqn",
        "timesteps": timesteps,
        "checkpoint_freq": checkpoint_freq,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "dqn_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return final_path


# ---------------------------------------------------------------------------
# PPO via SB3 (optional) ----------------------------------------------------


def train_ppo_sb3(env: TradingEnv, cfg: dict, timesteps: int, outdir: str) -> str:
    from stable_baselines3 import PPO  # pragma: no cover - optional dependency

    os.makedirs(outdir, exist_ok=True)
    ppo_cfg = cfg.get("ppo", {})
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=ppo_cfg.get("learning_rate", 3e-4))
    model.learn(total_timesteps=timesteps)
    path = os.path.join(outdir, "ppo_model.zip")
    model.save(path)
    return path


# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny DRL agents")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algo", default="dqn", help="dqn|tiny|ppo")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--data", type=str, default=None, help="Optional path to CSV/Parquet data")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_data(cfg, args.data, args.timesteps)
    env = TradingEnv(df)

    paths = cfg.get("paths", {})
    logs_dir = paths.get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = ensure_logger(os.path.join(logs_dir, "train.jsonl"))
    logger.log("INFO", "env_ready", obs_dim=int(env.observation_space.shape[0]), actions=int(env.action_space.n))

    if args.algo.lower() == "dqn":
        out = train_value_dqn(env, cfg, args.timesteps, outdir=paths.get("checkpoints_dir", "checkpoints"), checkpoint_freq=args.checkpoint_freq)
    elif args.algo.lower() == "tiny":
        out = train_dqn(env, cfg, args.timesteps, outdir=paths.get("checkpoints_dir", "checkpoints"), checkpoint_freq=args.checkpoint_freq)
    elif args.algo.lower() == "ppo":
        if not has_sb3():  # pragma: no cover - optional dependency
            raise RuntimeError("stable-baselines3 is required for PPO training")
        out = train_ppo_sb3(env, cfg, args.timesteps, outdir=paths.get("checkpoints_dir", "checkpoints"))
    else:  # pragma: no cover
        raise ValueError(f"Unknown algorithm: {args.algo}")

    logger.log("INFO", "training_done", algo=args.algo, artifact=out)
    print(f"Saved model/checkpoint to: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()

