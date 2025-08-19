from __future__ import annotations
from typing import Dict, Any

from .deterministic import DeterministicPolicy
from .stochastic import StochasticPolicy
from .value_based import ValueBasedPolicy

def get_policy(policy_type: str, obs_dim: int = 8, n_actions: int = 3, **kwargs):
    policy_type = (policy_type or "deterministic").lower()
    if policy_type == "deterministic":
        return DeterministicPolicy(**{k:v for k,v in kwargs.items() if k in {"threshold"}})
    elif policy_type == "stochastic":
        return StochasticPolicy(**{k:v for k,v in kwargs.items() if k in {"base_threshold","epsilon","seed"}})
    elif policy_type in {"value", "value-based", "dqn"}:
        return ValueBasedPolicy(obs_dim=obs_dim, n_actions=n_actions, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
