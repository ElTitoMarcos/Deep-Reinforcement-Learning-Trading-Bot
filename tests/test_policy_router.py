import pytest

from src.policies.router import get_policy
from src.policies.deterministic import DeterministicPolicy
from src.policies.stochastic import StochasticPolicy

try:  # pragma: no cover - availability is environment dependent
    import torch  # noqa: F401

    from src.policies.value_based import ValueBasedPolicy

    TORCH_AVAILABLE = True
except Exception:  # torch missing or broken
    TORCH_AVAILABLE = False


def test_router_modes_and_override():
    det = get_policy("price_only")
    sto = get_policy("complex_indicators")
    assert isinstance(det, DeterministicPolicy)
    assert isinstance(sto, StochasticPolicy)

    if TORCH_AVAILABLE:
        val = get_policy("market_cap_or_relative_value")
        assert isinstance(val, ValueBasedPolicy)
        override = get_policy("price_only", policy_override="value_based")
        assert isinstance(override, ValueBasedPolicy)
    else:
        with pytest.raises(Exception):
            get_policy("market_cap_or_relative_value")
