from __future__ import annotations
import argparse, os, time, json
import numpy as np
import pandas as pd

from ..utils.config import load_config
from ..utils.data_io import load_table
from ..utils.logging import ensure_logger
from ..env.trading_env import TradingEnv
from ..policies.router import get_policy
from ..policies.value_based import TinyDQN

def has_sb3():
    try:
        import stable_baselines3 as sb3  # noqa
        import torch  # noqa
        return True
    except Exception:
        return False

def train_with_sb3(env, cfg, timesteps: int, algo: str="ppo", outdir: str="checkpoints"):
    from stable_baselines3 import PPO, DQN
    os.makedirs(outdir, exist_ok=True)
    if algo.lower() == "ppo":
        model = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=cfg.get("ppo",{}).get("learning_rate",3e-4),
                    gamma=cfg.get("ppo",{}).get("gamma",0.99),
                    ent_coef=cfg.get("ppo",{}).get("ent_coef",0.01),
                    batch_size=cfg.get("ppo",{}).get("batch_size",64),
                    n_steps=cfg.get("ppo",{}).get("n_steps",2048))
    else:
        model = DQN("MlpPolicy", env, verbose=1,
                    learning_rate=cfg.get("dqn",{}).get("learning_rate",1e-3),
                    gamma=cfg.get("dqn",{}).get("gamma",0.99),
                    batch_size=cfg.get("dqn",{}).get("batch_size",64),
                    target_update_interval=cfg.get("dqn",{}).get("target_update",1000))
    model.learn(total_timesteps=timesteps)
    path = os.path.join(outdir, f"{algo}_model.zip")
    model.save(path)
    return path

def train_minimal_dqn(env, cfg, timesteps: int, outdir: str="checkpoints"):
    os.makedirs(outdir, exist_ok=True)
    dqn_cfg = cfg.get("dqn", {})
    eps_start = float(dqn_cfg.get("epsilon_start", 1.0))
    eps_end = float(dqn_cfg.get("epsilon_end", 0.05))
    eps_decay = int(dqn_cfg.get("epsilon_decay_steps", timesteps//2 or 1))
    agent = TinyDQN(obs_dim=env.observation_space.shape[0], n_actions=env.action_space.n,
                    gamma=float(dqn_cfg.get("gamma",0.99)),
                    lr=float(dqn_cfg.get("learning_rate",1e-3)),
                    buffer_size=int(dqn_cfg.get("buffer_size",10000)),
                    batch_size=int(dqn_cfg.get("batch_size",64)))
    obs, _ = env.reset()
    for t in range(timesteps):
        eps = max(eps_end, eps_start - (eps_start-eps_end)*(t/eps_decay))
        a = agent.select_action(obs, epsilon=eps)
        obs2, r, done, trunc, info = env.step(a)
        agent.remember(obs, a, r, obs2, done or trunc)
        agent.learn(steps=1)
        obs = obs2
        if done or trunc:
            obs, _ = env.reset()
    path = os.path.join(outdir, "dqn_minimal.npz")
    np.savez(path, W=agent.W)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--algo", type=str, default=None, help="ppo|dqn (si None usa config)")
    ap.add_argument("--timesteps", type=int, default=10000)
    ap.add_argument("--data", type=str, default=None, help="ruta parquet/csv; si no, usa paths del config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    algo = (args.algo or cfg.get("algo","ppo")).lower()
    paths = cfg.get("paths", {})
    raw_dir = paths.get("raw_dir", "data/raw")
    logs_dir = paths.get("logs_dir", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = ensure_logger(os.path.join(logs_dir, "train.jsonl"))

    exchange = cfg.get("exchange","binance")
    symbol = (cfg.get("symbols") or ["BTC/USDT"])[0]
    timeframe = cfg.get("timeframe","1m")
    if args.data:
        df = load_table(args.data)
    else:
        data_path = os.path.join(raw_dir, exchange, symbol.replace("/","-"), f"{timeframe}.parquet")
        df = load_table(data_path)

    env = TradingEnv(df, reward_weights=cfg.get("reward_weights"), fees=cfg.get("fees",{}).get("taker",0.001), slippage=cfg.get("slippage",0.0005))

    logger.log("INFO", "env_ready", obs_dim=int(env.observation_space.shape[0]), actions=int(env.action_space.n))

    if has_sb3() and algo == "ppo":
        out = train_with_sb3(env, cfg, args.timesteps, algo="ppo", outdir=paths.get("checkpoints_dir","checkpoints"))
    elif has_sb3() and algo == "dqn":
        out = train_with_sb3(env, cfg, args.timesteps, algo="dqn", outdir=paths.get("checkpoints_dir","checkpoints"))
    else:
        out = train_minimal_dqn(env, cfg, args.timesteps, outdir=paths.get("checkpoints_dir","checkpoints"))

    logger.log("INFO", "training_done", algo=algo, artifact=out)
    print(f"Saved model/checkpoint to: {out}")

if __name__ == "__main__":
    main()
