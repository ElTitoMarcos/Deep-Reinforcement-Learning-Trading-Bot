from __future__ import annotations
import argparse, os, time, json
import numpy as np
import pandas as pd

from ..utils.config import load_config
from ..utils.data_io import load_table
from ..utils.logging import ensure_logger
from ..env.trading_env import TradingEnv
from ..policies.router import exploration_scale
from ..policies.value_based import TinyDQN


def linear_schedule(start: float, end: float):
    """Returns a linear schedule callable for SB3."""
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining

    return schedule

def has_sb3():
    try:
        import stable_baselines3 as sb3  # noqa
        import torch  # noqa
        return True
    except Exception:
        return False

def train_with_sb3(env, cfg, timesteps: int, algo: str="ppo", outdir: str="checkpoints", *, data_mode: str="known", exploration: float = 1.0):
    from stable_baselines3 import PPO, DQN
    os.makedirs(outdir, exist_ok=True)
    meta: dict[str, float] = {
        "algo": algo.lower(),
        "timesteps": timesteps,
        "data_mode": data_mode,
        "exploration_scale": exploration,
    }
    if algo.lower() == "ppo":
        ppo_cfg = cfg.get("ppo", {})
        ent_start = float(ppo_cfg.get("entropy_start", ppo_cfg.get("ent_coef", 0.01)))
        ent_end = float(ppo_cfg.get("entropy_end", ent_start))
        ent_schedule = ent_start if ent_start == ent_end else linear_schedule(ent_start, ent_end)
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=ppo_cfg.get("learning_rate", 3e-4),
            gamma=ppo_cfg.get("gamma", 0.99),
            ent_coef=ent_schedule,
            batch_size=ppo_cfg.get("batch_size", 64),
            n_steps=ppo_cfg.get("n_steps", 2048),
        )
        meta.update({"entropy_start": ent_start, "entropy_end": ent_end})
    else:
        dqn_cfg = cfg.get("dqn", {})
        eps_start = float(dqn_cfg.get("epsilon_start", 1.0))
        eps_end = float(dqn_cfg.get("epsilon_end", 0.05))
        eps_steps = int(dqn_cfg.get("epsilon_decay_steps", timesteps // 2 or 1))
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=dqn_cfg.get("learning_rate", 1e-3),
            gamma=dqn_cfg.get("gamma", 0.99),
            batch_size=dqn_cfg.get("batch_size", 64),
            target_update_interval=dqn_cfg.get("target_update", 1000),
            exploration_initial_eps=eps_start,
            exploration_final_eps=eps_end,
            exploration_fraction=eps_steps / float(timesteps),
        )
        meta.update({"epsilon_start": eps_start, "epsilon_end": eps_end, "epsilon_decay_steps": eps_steps})
    model.learn(total_timesteps=timesteps)
    path = os.path.join(outdir, f"{algo}_model.zip")
    model.save(path)
    with open(os.path.splitext(path)[0] + "_config.json", "w") as fh:
        json.dump(meta, fh)
    return path

def train_minimal_dqn(env, cfg, timesteps: int, outdir: str="checkpoints", *, data_mode: str="known", exploration: float = 1.0):
    os.makedirs(outdir, exist_ok=True)
    dqn_cfg = cfg.get("dqn", {})
    eps_start = float(dqn_cfg.get("epsilon_start", 1.0))
    eps_end = float(dqn_cfg.get("epsilon_end", 0.05))
    eps_decay = int(dqn_cfg.get("epsilon_decay_steps", timesteps//2 or 1))
    agent = TinyDQN(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        gamma=float(dqn_cfg.get("gamma", 0.99)),
        lr=float(dqn_cfg.get("learning_rate", 1e-3)),
        buffer_size=int(dqn_cfg.get("buffer_size", 10000)),
        batch_size=int(dqn_cfg.get("batch_size", 64)),
    )
    obs, _ = env.reset()
    for t in range(timesteps):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * (t / eps_decay))
        a = agent.select_action(obs, epsilon=eps)
        obs2, r, done, trunc, info = env.step(a)
        agent.remember(obs, a, r, obs2, done or trunc)
        agent.learn(steps=1)
        obs = obs2
        if done or trunc:
            obs, _ = env.reset()
    path = os.path.join(outdir, "dqn_minimal.npz")
    np.savez(path, W=agent.W)
    meta = {
        "algo": "dqn_minimal",
        "epsilon_start": eps_start,
        "epsilon_end": eps_end,
        "epsilon_decay_steps": eps_decay,
        "timesteps": timesteps,
        "data_mode": data_mode,
        "exploration_scale": exploration,
    }
    with open(os.path.splitext(path)[0] + "_config.json", "w") as fh:
        json.dump(meta, fh)
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--algo", type=str, default=None, help="ppo|dqn (si None usa config)")
    ap.add_argument("--timesteps", type=int, default=10000)
    ap.add_argument("--data", type=str, default=None, help="ruta parquet/csv; si no, usa paths del config")
    ap.add_argument("--eps-start", type=float, default=None)
    ap.add_argument("--eps-end", type=float, default=None)
    ap.add_argument("--eps-steps", type=int, default=None)
    ap.add_argument("--entropy-start", type=float, default=None)
    ap.add_argument("--entropy-end", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    algo = (args.algo or cfg.get("algo","ppo")).lower()
    if args.eps_start is not None:
        cfg.setdefault("dqn", {})["epsilon_start"] = args.eps_start
    if args.eps_end is not None:
        cfg.setdefault("dqn", {})["epsilon_end"] = args.eps_end
    if args.eps_steps is not None:
        cfg.setdefault("dqn", {})["epsilon_decay_steps"] = args.eps_steps
    if args.entropy_start is not None:
        cfg.setdefault("ppo", {})["entropy_start"] = args.entropy_start
    if args.entropy_end is not None:
        cfg.setdefault("ppo", {})["entropy_end"] = args.entropy_end

    router_cfg = cfg.get("router", {})
    data_mode = router_cfg.get("data_mode", "known")
    exp_scale = exploration_scale(data_mode)
    if algo == "dqn":
        dqn_cfg = cfg.setdefault("dqn", {})
        base_start = float(dqn_cfg.get("epsilon_start", 1.0))
        base_end = float(dqn_cfg.get("epsilon_end", 0.05))
        dqn_cfg["epsilon_start"] = min(1.0, base_start * exp_scale)
        dqn_cfg["epsilon_end"] = min(1.0, base_end * exp_scale)
    elif algo == "ppo":
        ppo_cfg = cfg.setdefault("ppo", {})
        base_start = float(ppo_cfg.get("entropy_start", ppo_cfg.get("ent_coef", 0.01)))
        base_end = float(ppo_cfg.get("entropy_end", base_start))
        ppo_cfg["entropy_start"] = base_start * exp_scale
        ppo_cfg["entropy_end"] = base_end * exp_scale
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
    if algo == "dqn":
        dqn_cfg = cfg.get("dqn", {})
        logger.log(
            "INFO",
            "train_config",
            algo="dqn",
            data_mode=data_mode,
            exploration_scale=exp_scale,
            epsilon_start=dqn_cfg.get("epsilon_start"),
            epsilon_end=dqn_cfg.get("epsilon_end"),
            epsilon_decay_steps=dqn_cfg.get("epsilon_decay_steps"),
        )
    elif algo == "ppo":
        ppo_cfg = cfg.get("ppo", {})
        logger.log(
            "INFO",
            "train_config",
            algo="ppo",
            data_mode=data_mode,
            exploration_scale=exp_scale,
            entropy_start=ppo_cfg.get("entropy_start"),
            entropy_end=ppo_cfg.get("entropy_end", ppo_cfg.get("entropy_start")),
        )

    if has_sb3() and algo == "ppo":
        out = train_with_sb3(
            env,
            cfg,
            args.timesteps,
            algo="ppo",
            outdir=paths.get("checkpoints_dir", "checkpoints"),
            data_mode=data_mode,
            exploration=exp_scale,
        )
    elif has_sb3() and algo == "dqn":
        out = train_with_sb3(
            env,
            cfg,
            args.timesteps,
            algo="dqn",
            outdir=paths.get("checkpoints_dir", "checkpoints"),
            data_mode=data_mode,
            exploration=exp_scale,
        )
    else:
        out = train_minimal_dqn(
            env,
            cfg,
            args.timesteps,
            outdir=paths.get("checkpoints_dir", "checkpoints"),
            data_mode=data_mode,
            exploration=exp_scale,
        )

    logger.log("INFO", "training_done", algo=algo, artifact=out)
    print(f"Saved model/checkpoint to: {out}")

if __name__ == "__main__":
    main()
