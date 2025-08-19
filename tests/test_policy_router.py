import pytest

from src.policies.router import get_policy, exploration_scale
from src.policies.deterministic import DeterministicPolicy
from src.policies.stochastic import StochasticPolicy

try:  # pragma: no cover - availability is environment dependent
    import torch  # noqa: F401
    from src.policies.value_based import ValueBasedPolicy
    TORCH_AVAILABLE = True
except Exception:  # torch missing or broken
    TORCH_AVAILABLE = False


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("price_only", DeterministicPolicy),
        ("complex_indicators", StochasticPolicy),
    ],
)
def test_policy_mappings(mode, expected):
    assert isinstance(get_policy(mode), expected)


def test_value_based_and_override():
    if TORCH_AVAILABLE:
        assert isinstance(get_policy("market_cap_or_relative_value"), ValueBasedPolicy)
        assert isinstance(
            get_policy("price_only", policy_override="value_based"), ValueBasedPolicy
        )
    else:
        with pytest.raises(Exception):
            get_policy("market_cap_or_relative_value")


@pytest.mark.parametrize(
    "mode,scale",
    [
        ("known", 1.0),
        ("new_feature_set", 1.2),
        ("unknown", 1.0),
    ],
)
def test_exploration_scale(mode, scale):
    assert exploration_scale(mode) == pytest.approx(scale)
