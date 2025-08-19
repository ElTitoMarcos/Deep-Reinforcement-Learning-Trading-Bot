from __future__ import annotations
from typing import Any

from .deterministic import DeterministicPolicy
from .stochastic import StochasticPolicy


def _load_value_based():
    """Lazy import for the torch-dependent policy."""
    from .value_based import ValueBasedPolicy  # type: ignore

    return ValueBasedPolicy

# ---------------------------------------------------------------------------
# Mapping from data modes to policy implementations and exploration levels.
POLICY_MAP = {
    "price_only": "deterministic",
    "complex_indicators": "stochastic",
    "market_cap_or_relative_value": "value_based",
}

EXPLORATION_MAP = {
    "price_only": 0.05,  # low exploration
    "complex_indicators": 0.1,  # medium exploration
    "market_cap_or_relative_value": 0.3,  # high exploration
}


def get_policy(
    data_mode: str,
    policy_override: str | None = None,
    obs_dim: int = 8,
    n_actions: int = 3,
    **kwargs: Any,
):
    """Return an appropriate policy instance for ``data_mode``.

    Parameters
    ----------
    data_mode : str
        Descriptor of the feature set.  Maps to a default policy.
    policy_override : str, optional
        Explicit policy type to use regardless of ``data_mode``.
    obs_dim : int
        Observation dimensionality for value-based methods.
    n_actions : int
        Number of discrete actions.
    **kwargs : Any
        Extra parameters forwarded to the underlying policy constructor.
    """

    mode = (data_mode or "price_only").lower()
    name = (policy_override or POLICY_MAP.get(mode, "deterministic")).lower()
    exploration = EXPLORATION_MAP.get(mode, 0.1)

    if name == "deterministic":
        return DeterministicPolicy(**kwargs)
    elif name == "stochastic":
        kwargs.setdefault("temperature", exploration)
        return StochasticPolicy(**kwargs)
    elif name in {"value", "value-based", "value_based", "dqn"}:
        ValueBasedPolicy = _load_value_based()
        cfg = kwargs.pop("config", {})
        if isinstance(cfg, dict):
            cfg = dict(cfg)
            cfg.setdefault("epsilon_start", exploration)
        else:
            # dataclass or namespace-like object
            if not hasattr(cfg, "epsilon_start"):
                setattr(cfg, "epsilon_start", exploration)
        return ValueBasedPolicy(obs_dim=obs_dim, n_actions=n_actions, config=cfg, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {name}")
