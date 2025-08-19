"""Utilities to choose an RL algorithm given environment stats.

The selector tries to pick between PPO, DQN or a hybrid ensemble based on
simple heuristics.  It expects two dictionaries:
    stats: historical statistics such as Sharpe ratios or drawdowns.
    env_caps: description of observation/action spaces and reward sparsity.

It returns a mapping ``{"algo": ..., "reason": ...}``.
"""
from __future__ import annotations
from typing import Dict, Any


def choose_algo(stats: Dict[str, Any], env_caps: Dict[str, Any]) -> Dict[str, str]:
    """Return the recommended algorithm and a short reason.

    Parameters
    ----------
    stats : dict
        Historical performance stats. Expected keys:
        ``ppo_sharpe``, ``dqn_sharpe``, ``ppo_maxdd``, ``dqn_maxdd``,
        ``rewards_sparse``.
    env_caps : dict
        Environment capabilities such as ``obs_type`` ("continuous" or
        "discrete"), ``action_type`` and ``state_space`` size.
    """
    obs = env_caps.get("obs_type", "continuous")
    act = env_caps.get("action_type", "discrete")
    state_space = env_caps.get("state_space", float("inf"))
    sparse = stats.get("rewards_sparse", False)

    ppo_sharpe = stats.get("ppo_sharpe")
    dqn_sharpe = stats.get("dqn_sharpe")
    ppo_dd = stats.get("ppo_maxdd")
    dqn_dd = stats.get("dqn_maxdd")

    # Heuristic 3: conflicting metrics -> hybrid
    if (
        ppo_sharpe is not None
        and dqn_sharpe is not None
        and ppo_dd is not None
        and dqn_dd is not None
        and ppo_sharpe > 1.5 * dqn_sharpe
        and dqn_dd < ppo_dd
    ):
        return {
            "algo": "hybrid",
            "reason": "PPO domina en Sharpe pero DQN reduce drawdown",
        }

    # Heuristic 2: small state space and sparse rewards -> DQN
    if state_space < 50 and sparse:
        return {
            "algo": "dqn",
            "reason": "Espacio de estados pequeño con recompensas esparsas",
        }

    # Heuristic 1: continuous obs & discrete actions -> PPO
    if obs == "continuous" and act == "discrete":
        return {
            "algo": "ppo",
            "reason": "Observaciones continuas y acciones discretas",
        }

    # Fallback
    return {"algo": "ppo", "reason": "Regla por defecto"}
