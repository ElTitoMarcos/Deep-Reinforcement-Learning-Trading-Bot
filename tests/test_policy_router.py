from src.policies.router import get_policy
from src.policies.value_based import TinyDQN
from src.policies.deterministic import DeterministicPolicy
from src.policies.stochastic import StochasticPolicy

def test_router():
    det = get_policy("deterministic")
    sto = get_policy("stochastic")
    val = get_policy("dqn")
    assert isinstance(det, DeterministicPolicy)
    assert isinstance(sto, StochasticPolicy)
    assert isinstance(val, TinyDQN)
