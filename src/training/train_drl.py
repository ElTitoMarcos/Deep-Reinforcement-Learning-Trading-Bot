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
from datetime import datetime, timedelta, UTC
from pathlib import Path
import time
import random
import errno
import socket

import numpy as np
import pandas as pd

from ..env.trading_env import TradingEnv
from ..auto.hparam_tuner import tune
from ..auto.timeframe_adapter import propose_timeframe
from ..utils.config import load_config
from ..utils.data_io import load_table, resample_to
from ..utils.logging import ensure_logger, config_hash
from ..utils import paths
from ..utils.device import get_device, set_cpu_threads
from ..policies.value_based import ValueBasedPolicy
from ..llm import LLMClient, SYSTEM_PROMPT, build_periodic_prompt


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
    """Load price data or generate a synthetic series if none is found."""

    if data_path:
        return load_table(data_path)

    exchange = cfg.get("exchange", "binance")
    symbol = (cfg.get("symbols") or ["BTC/USDT"])[0]
    timeframe = cfg.get("timeframe", "1m")
    fname = paths.raw_parquet_path(exchange, symbol, timeframe)

    if fname.exists():  # pragma: no branch - depends on repo data
        return load_table(paths.posix(fname))

    # generate simple random walk prices
    n = max(timesteps + 1, 1_000)
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.normal(0, 1, size=n)) + 100.0
    df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices})
    return df


def quick_eval(env: TradingEnv, agent: TinyDQN) -> float:
    """Run a fast evaluation episode and return final equity."""

    eval_env = TradingEnv(env.df, cfg=env.cfg)
    obs, _ = eval_env.reset()
    done = False
    while not done:
        action = agent.act(obs, epsilon=0.0)
        obs, _, done, _, _ = eval_env.step(action)
    return float(eval_env.equity)


def _maybe_call_llm(
    cfg: dict,
    algo_key: str,
    episode: int,
    reward: float,
    env: TradingEnv,
    llm_client: LLMClient | None,
    llm_file,
    logger,
) -> None:
    """Invoke the LLM with training context if enabled."""

    if llm_client is None:
        return
    prompt = build_periodic_prompt(
        cfg,
        algo=algo_key,
        hparams=cfg.get(algo_key, {}),
        episodios=episode,
        reward=reward,
        pnl=float(getattr(env, "equity", 0.0)),
        dd=0.0,
        cons=0.0,
    )
    try:
        resp = llm_client.ask(SYSTEM_PROMPT, prompt)
        try:
            data = json.loads(resp)
        except Exception:
            data = {"raw": resp}
        if llm_file:
            llm_file.write(json.dumps(data) + "\n")
            llm_file.flush()
        logger.log("INFO", "llm_suggestion", episode=episode, suggestion=data)
    except Exception as e:  # pragma: no cover - network issues
        logger.log("ERROR", "llm_error", episode=episode, err=str(e))

# ---------------------------------------------------------------------------
# Model IO helpers ----------------------------------------------------------

def save_model(agent: ValueBasedPolicy, algo: str, symbol: str) -> str:
    """Persist a trained policy to the models directory with metadata."""
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    fname = f"{ts}_{algo}_{symbol}.pt"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    path = models_dir / fname
    agent.save_model(paths.posix(path))
    return paths.posix(path)


def load_model(path: str, cfg: dict) -> ValueBasedPolicy:
    """Load a :class:`ValueBasedPolicy` and validate config hash."""
    meta_path = os.path.splitext(path)[0] + ".json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("config_hash") != config_hash(cfg):
        raise ValueError("config hash mismatch")
    policy = ValueBasedPolicy(obs_dim=meta["obs_dim"], n_actions=meta["n_actions"], config=cfg)
    policy.load_model(path)
    return policy


def _save_checkpoint(
    path: Path,
    agent: ValueBasedPolicy,
    total_steps: int,
    episode: int,
    total_reward: float,
    start_time: float,
) -> None:
    """Persist training state for resumption."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": agent.q_net.state_dict(),
        "target": agent.target_net.state_dict(),
        "optim": agent.optim.state_dict(),
        "epsilon": agent.epsilon,
        "step_count": agent.step_count,
        "total_steps": total_steps,
        "episode": episode,
        "total_reward": total_reward,
        "elapsed": time.time() - start_time,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    torch.save(state, paths.posix(path))


def _load_checkpoint(
    path: Path, agent: ValueBasedPolicy
) -> tuple[int, int, float, float]:
    """Restore training state from ``path``."""
    import torch

    data = torch.load(paths.posix(path), map_location=agent.device)
    agent.q_net.load_state_dict(data.get("model", {}))
    tgt = data.get("target")
    if tgt:
        agent.target_net.load_state_dict(tgt)
    agent.optim.load_state_dict(data.get("optim", {}))
    agent.epsilon = data.get("epsilon", agent.epsilon)
    agent.step_count = data.get("step_count", agent.step_count)
    rng = data.get("rng_state", {})
    try:
        random.setstate(rng.get("python"))
    except Exception:
        pass
    try:
        np.random.set_state(rng.get("numpy"))
    except Exception:
        pass
    try:
        torch.set_rng_state(rng.get("torch"))
    except Exception:
        pass
    total_steps = int(data.get("total_steps", 0))
    episode = int(data.get("episode", 0))
    total_reward = float(data.get("total_reward", 0.0))
    elapsed = float(data.get("elapsed", 0.0))
    start_time = time.time() - elapsed
    return total_steps, episode, total_reward, start_time

def train_value_dqn(
    env: TradingEnv,
    cfg: dict,
    timesteps: int,
    *,
    outdir: str = "checkpoints",
    checkpoint_freq: int = 10,
    continuous: bool = False,
    checkpoint_interval_min: int = 10,
    max_hours: float | None = None,
    logger=None,
) -> str:
    """Train the PyTorch value-based policy with the flexible MLP network."""

    dqn_cfg = cfg.get("dqn", {})
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    agent = ValueBasedPolicy(obs_dim, n_actions, config=dqn_cfg)
    if logger is None:
        logger = ensure_logger(None)

    llm_cfg = cfg.get("llm", {})
    llm_client = None
    llm_file = None
    llm_every = int(llm_cfg.get("every_n") or 0)
    if llm_cfg.get("enabled") and llm_cfg.get("periodic") and llm_every > 0:
        llm_client = LLMClient(model=llm_cfg.get("model", "gpt-4o"))
        reports_dir = paths.reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        llm_file = open(reports_dir / "llm_suggestions.jsonl", "a", encoding="utf-8")

    auto_cfg = cfg.get("auto", {})
    stage_eps = int(auto_cfg.get("timeframe_every_episodes", 0))
    base_tf = cfg.get("timeframe", "1m")
    base_df = env.df.copy()
    current_tf = base_tf

    total_steps = 0
    episode = 0
    total_reward = 0.0
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = Path(outdir) / "vdqn_ckpt.pt"
    start_time = time.time()
    last_ckpt = start_time
    last_heartbeat = start_time
    if continuous and ckpt_path.exists():
        total_steps, episode, total_reward, start_time = _load_checkpoint(ckpt_path, agent)
        last_ckpt = last_heartbeat = time.time()

    backoff = 60.0
    while total_steps < timesteps:
        try:
            seed_override = None
            if stage_eps and episode % stage_eps == 0:
                returns = env.df["close"].pct_change().dropna()
                recent_vol = float(returns.rolling(20).std().iloc[-1]) if not returns.empty else 0.0
                gaps = np.diff(env.df["ts"].to_numpy())
                step = pd.Series(gaps).mode().iloc[0] if len(gaps) else 0
                gap_ratio = float((gaps > step * 1.5).mean()) if step else 0.0
                stats = {
                    "recent_volatility": recent_vol,
                    "gap_ratio": gap_ratio,
                    "device": dqn_cfg.get("device", "cpu"),
                    "batch_size": dqn_cfg.get("batch_size", 64),
                    "base_tf": base_tf,
                    "current_tf": current_tf,
                }
                vol_prof = auto_cfg.get("vol_profile", {"high": 0.02, "low": 0.005})
                lat_budget = 1.0 if dqn_cfg.get("device") == "cuda" else 0.5
                proposal = propose_timeframe(stats, vol_prof, lat_budget, llm_client if llm_cfg.get("enabled") else None)
                if proposal["resample_to"] != current_tf:
                    current_tf = proposal["resample_to"]
                    new_df = resample_to(base_df, current_tf)
                    env = TradingEnv(new_df, cfg=env.cfg)
                    seed_override = random.randint(0, 2**32 - 1)
                    logger.log(
                        "INFO",
                        "timeframe_adapted",
                        base=proposal["base_tf"],
                        resample=current_tf,
                        reason=proposal["reason"],
                        seed=seed_override,
                    )
                    print(
                        f"Timeframe adaptado: base={proposal['base_tf']} \u2192 resample={current_tf} (motivo: {proposal['reason']})"
                    )
            obs, _ = env.reset(seed=seed_override)
            done = False
            episode += 1
            ep_reward = 0.0
            while not done and total_steps < timesteps:
                action = agent.act(obs)
                next_obs, reward, done, trunc, _info = env.step(action)
                agent.remember(obs, action, reward, next_obs, done or trunc)
                agent.train_step()
                obs = next_obs
                total_steps += 1
                ep_reward += reward

                if continuous:
                    now = time.time()
                    if checkpoint_interval_min and now - last_ckpt >= checkpoint_interval_min * 60:
                        _save_checkpoint(ckpt_path, agent, total_steps, episode, total_reward + ep_reward, start_time)
                        last_ckpt = now
                    if now - last_heartbeat >= 300:
                        mean_reward = (total_reward + ep_reward) / max(1, episode)
                        hhmm = time.strftime("%H:%M", time.gmtime(now - start_time))
                        msg = f"alive t={hhmm}, steps={total_steps}, reward_mean={mean_reward:.4f}"
                        print(msg)
                        logger.log("INFO", "heartbeat", t=hhmm, steps=total_steps, reward_mean=mean_reward)
                        last_heartbeat = now
                    if max_hours and (now - start_time) >= max_hours * 3600:
                        _save_checkpoint(ckpt_path, agent, total_steps, episode, total_reward + ep_reward, start_time)
                        if llm_file:
                            llm_file.close()
                        symbol = paths.symbol_to_dir((cfg.get("symbols") or ["UNK"])[0])
                        final_path = save_model(agent, "dqn", symbol)
                        return final_path

            total_reward += ep_reward
            if llm_client and episode % llm_every == 0:
                mean_reward = total_reward / episode
                _maybe_call_llm(
                    cfg,
                    "dqn",
                    episode,
                    mean_reward,
                    env,
                    llm_client,
                    llm_file,
                    logger,
                )

            if not continuous and checkpoint_freq and episode % checkpoint_freq == 0:
                ckpt = Path(outdir) / f"vdqn_ep{episode}.pt"
                agent.save_model(paths.posix(ckpt))

        except (MemoryError, OSError) as e:
            if not continuous:
                raise
            err = getattr(e, "errno", None)
            transient = isinstance(e, MemoryError) or err in {errno.EAGAIN, getattr(socket, "EAI_AGAIN", None)}
            if not transient:
                raise
            logger.log("WARN", "transient_error", err=str(e), backoff=backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 3600)
            if ckpt_path.exists():
                total_steps, episode, total_reward, start_time = _load_checkpoint(ckpt_path, agent)
            continue

    if continuous:
        _save_checkpoint(ckpt_path, agent, total_steps, episode, total_reward, start_time)
    symbol = paths.symbol_to_dir((cfg.get("symbols") or ["UNK"])[0])
    final_path = save_model(agent, "dqn", symbol)
    if llm_file:
        llm_file.close()

    return final_path


def train_dqn(
    env: TradingEnv,
    cfg: dict,
    timesteps: int,
    *,
    outdir: str = "checkpoints",
    checkpoint_freq: int = 10,
    logger=None,
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

    if logger is None:
        logger = ensure_logger(None)

    llm_cfg = cfg.get("llm", {})
    llm_client = None
    llm_file = None
    llm_every = int(llm_cfg.get("every_n") or 0)
    if llm_cfg.get("enabled") and llm_cfg.get("periodic") and llm_every > 0:
        llm_client = LLMClient(model=llm_cfg.get("model", "gpt-4o"))
        reports_dir = paths.reports_dir()
        reports_dir.mkdir(parents=True, exist_ok=True)
        llm_file = open(reports_dir / "llm_suggestions.jsonl", "a", encoding="utf-8")

    auto_cfg = cfg.get("auto", {})
    stage_eps = int(auto_cfg.get("timeframe_every_episodes", 0))
    base_tf = cfg.get("timeframe", "1m")
    base_df = env.df.copy()
    current_tf = base_tf

    eps_start = dqn_cfg.get("epsilon_start", 1.0)
    eps_end = dqn_cfg.get("epsilon_end", 0.05)
    eps_decay_steps = dqn_cfg.get("epsilon_decay_steps", timesteps // 2 or 1)

    total_steps = 0
    episode = 0
    total_reward = 0.0
    while total_steps < timesteps:
        seed_override = None
        if stage_eps and episode % stage_eps == 0:
            returns = env.df["close"].pct_change().dropna()
            recent_vol = float(returns.rolling(20).std().iloc[-1]) if not returns.empty else 0.0
            gaps = np.diff(env.df["ts"].to_numpy())
            step = pd.Series(gaps).mode().iloc[0] if len(gaps) else 0
            gap_ratio = float((gaps > step * 1.5).mean()) if step else 0.0
            stats = {
                "recent_volatility": recent_vol,
                "gap_ratio": gap_ratio,
                "device": dqn_cfg.get("device", "cpu"),
                "batch_size": dqn_cfg.get("batch_size", 64),
                "base_tf": base_tf,
                "current_tf": current_tf,
            }
            vol_prof = auto_cfg.get("vol_profile", {"high": 0.02, "low": 0.005})
            lat_budget = 1.0 if dqn_cfg.get("device") == "cuda" else 0.5
            proposal = propose_timeframe(stats, vol_prof, lat_budget, llm_client if llm_cfg.get("enabled") else None)
            if proposal["resample_to"] != current_tf:
                current_tf = proposal["resample_to"]
                new_df = resample_to(base_df, current_tf)
                env = TradingEnv(new_df, cfg=env.cfg)
                seed_override = random.randint(0, 2**32 - 1)
                logger.log(
                    "INFO",
                    "timeframe_adapted",
                    base=proposal["base_tf"],
                    resample=current_tf,
                    reason=proposal["reason"],
                    seed=seed_override,
                )
                print(
                    f"Timeframe adaptado: base={proposal['base_tf']} \u2192 resample={current_tf} (motivo: {proposal['reason']})"
                )
        obs, _ = env.reset(seed=seed_override)
        done = False
        episode += 1
        ep_reward = 0.0
        while not done and total_steps < timesteps:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * (total_steps / eps_decay_steps))
            action = agent.act(obs, eps)
            next_obs, reward, done, trunc, _info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done or trunc)
            agent.update()
            obs = next_obs
            total_steps += 1
            ep_reward += reward

        total_reward += ep_reward
        if llm_client and episode % llm_every == 0:
            mean_reward = total_reward / episode
            _maybe_call_llm(
                cfg,
                "dqn",
                episode,
                mean_reward,
                env,
                llm_client,
                llm_file,
                logger,
            )

        if checkpoint_freq and episode % checkpoint_freq == 0:
            ckpt = Path(outdir) / f"dqn_ep{episode}.npz"
            agent.save(paths.posix(ckpt))
            equity = quick_eval(env, agent)
            print(f"[checkpoint] episode={episode} equity={equity:.2f}")

    final_path = Path(outdir) / "dqn_final.npz"
    agent.save(paths.posix(final_path))
    meta = {
        "algo": "dqn",
        "timesteps": timesteps,
        "checkpoint_freq": checkpoint_freq,
    }
    os.makedirs(outdir, exist_ok=True)
    with open(Path(outdir) / "dqn_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    if llm_file:
        llm_file.close()
    return paths.posix(final_path)


# ---------------------------------------------------------------------------
# PPO via SB3 (optional) ----------------------------------------------------


def train_ppo_sb3(env: TradingEnv, cfg: dict, timesteps: int, outdir: str, device: str) -> str:
    from stable_baselines3 import PPO  # pragma: no cover - optional dependency

    os.makedirs(outdir, exist_ok=True)
    ppo_cfg = cfg.get("ppo", {})
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=ppo_cfg.get("learning_rate", 3e-4),
        device=device,
    )
    model.learn(total_timesteps=timesteps)
    path = Path(outdir) / "ppo_model.zip"
    model.save(paths.posix(path))
    return paths.posix(path)


# ---------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny DRL agents")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--algo", default="dqn", help="dqn|tiny|ppo|hybrid")
    parser.add_argument("--algo-reason", default="", help="Descripción breve de la elección automática")
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--data", type=str, default=None, help="Optional path to CSV/Parquet data")
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--continuous", action="store_true", help="Resume training with periodic checkpoints")
    parser.add_argument(
        "--checkpoint-interval-min", type=int, default=10, help="Minutes between checkpoints"
    )
    parser.add_argument("--max-hours", type=float, default=None, help="Stop after this many hours")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths.ensure_dirs_exist()
    if args.continuous and not args.data:
        from datetime import datetime, timedelta, timezone
        from ..data.incremental import (
            last_watermark,
            fetch_ohlcv_incremental,
            upsert_parquet,
        )
        from ..data.ccxt_loader import get_exchange

        ex = get_exchange(use_testnet=cfg.get("binance_use_testnet"))
        timeframe = cfg.get("timeframe", "1m")
        for sym in cfg.get("symbols", []):
            since = last_watermark(sym, timeframe)
            if since is None:
                since = int((datetime.now(UTC) - timedelta(days=30)).timestamp() * 1000)
            df_new = fetch_ohlcv_incremental(ex, sym, timeframe, since_ms=since)
            if df_new.empty:
                continue
            path = paths.raw_parquet_path(ex.id if hasattr(ex, "id") else "binance", sym, timeframe)
            upsert_parquet(df_new, path)
            manifest = {
                "symbol": sym,
                "timeframe": timeframe,
                "watermark": int(df_new["ts"].max()),
                "obtained_at": datetime.now(UTC).isoformat(),
            }
            with open(path.with_suffix(".manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
    df = load_data(cfg, args.data, args.timesteps)
    env = TradingEnv(df, cfg=cfg)
    print(f"Using fees: {cfg.get('fees', {})}")

    paths_cfg = cfg.get("paths", {})
    logs_dir = paths_cfg.get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = ensure_logger(paths.posix(Path(logs_dir) / "train.jsonl"))
    logger.log(
        "INFO",
        "env_ready",
        obs_dim=int(env.observation_space.shape[0]),
        actions=int(env.action_space.n),
    )
    device = get_device()
    if device == "cuda":
        import torch  # local import to avoid unnecessary dependency on CPU-only

        name = torch.cuda.get_device_name(0)
        logger.log("INFO", "device", type="cuda", name=name)
        print(f"Using device: CUDA ({name})")
    else:
        threads = set_cpu_threads()
        logger.log("INFO", "device", type="cpu", threads=threads)
        print(f"Using device: CPU ({threads} threads)")
    cfg.setdefault("dqn", {})["device"] = device
    cfg.setdefault("ppo", {})["device"] = device
    if args.algo_reason:
        logger.log("INFO", "auto_algo", algo=args.algo, reason=args.algo_reason)

    algo_key = args.algo.lower()
    if args.continuous and algo_key != "dqn":
        raise ValueError("--continuous is only supported for dqn")
    data_stats = {
        "obs_dim": int(env.observation_space.shape[0]),
        "n_actions": int(env.action_space.n),
        "timesteps": args.timesteps,
    }
    tuned = tune(algo_key if algo_key in {"ppo", "dqn", "hybrid"} else "dqn", data_stats, [])
    if algo_key == "hybrid":
        ppo_params = {**tuned.get("ppo", {}), **cfg.get("ppo", {})}
        dqn_params = {**tuned.get("dqn", {}), **cfg.get("dqn", {})}
        cfg["ppo"] = ppo_params
        cfg["dqn"] = dqn_params
        logger.log("INFO", "hparams", algo="ppo", params=ppo_params)
        logger.log("INFO", "hparams", algo="dqn", params=dqn_params)
    elif algo_key == "ppo":
        params = {**tuned, **cfg.get("ppo", {})}
        cfg["ppo"] = params
        logger.log("INFO", "hparams", algo="ppo", params=params)
    else:  # dqn or other value-based variants
        params = {**tuned, **cfg.get("dqn", {})}
        cfg["dqn"] = params
        logger.log("INFO", "hparams", algo="dqn", params=params)

    ckpt_dir = paths.checkpoints_dir()
    if algo_key == "dqn":
        out = train_value_dqn(
            env,
            cfg,
            args.timesteps,
            outdir=paths.posix(ckpt_dir),
            checkpoint_freq=args.checkpoint_freq,
            continuous=args.continuous,
            checkpoint_interval_min=args.checkpoint_interval_min,
            max_hours=args.max_hours,
            logger=logger,
        )
    elif algo_key == "tiny":
        out = train_dqn(
            env,
            cfg,
            args.timesteps,
            outdir=paths.posix(ckpt_dir),
            checkpoint_freq=args.checkpoint_freq,
            logger=logger,
        )
    elif algo_key == "ppo":
        if not has_sb3():  # pragma: no cover - optional dependency
            raise RuntimeError("stable-baselines3 is required for PPO training")
        out = train_ppo_sb3(env, cfg, args.timesteps, outdir=paths.posix(ckpt_dir), device=device)
    elif algo_key == "hybrid":
        if not has_sb3():  # pragma: no cover - optional dependency
            raise RuntimeError("stable-baselines3 is required for PPO training")
        ppo_path = train_ppo_sb3(env, cfg, args.timesteps, outdir=paths.posix(ckpt_dir), device=device)
        dqn_path = train_value_dqn(
            env,
            cfg,
            args.timesteps,
            outdir=paths.posix(ckpt_dir),
            checkpoint_freq=args.checkpoint_freq,
            continuous=args.continuous,
            checkpoint_interval_min=args.checkpoint_interval_min,
            max_hours=args.max_hours,
            logger=logger,
        )
        out = json.dumps({"ppo": ppo_path, "dqn": dqn_path})
    else:  # pragma: no cover
        raise ValueError(f"Unknown algorithm: {args.algo}")

    logger.log("INFO", "training_done", algo=args.algo, artifact=out)
    print(f"Saved model/checkpoint to: {out}")


if __name__ == "__main__":  # pragma: no cover
    main()

