import numpy as np
import torch
import pytest

from src.policies.nets import MLP
from src.policies.value_based import ValueBasedPolicy, DQNConfig


def test_mlp_forward_and_dropout():
    net = MLP(3, [4, 5], out_dim=2, activation="tanh", dropout=0.1)
    x = torch.randn(7, 3)
    y = net(x)
    assert y.shape == (7, 2)
    assert any(isinstance(m, torch.nn.Dropout) for m in net.model.modules())


def test_value_policy_act_shape():
    cfg = DQNConfig(hidden_sizes=(8,), activation="relu", dropout=None)
    policy = ValueBasedPolicy(obs_dim=4, n_actions=3, config=cfg)
    obs = np.zeros(4, dtype=np.float32)
    action = policy.act(obs)
    assert 0 <= action < 3


def test_value_policy_act_raises_on_shape():
    cfg = DQNConfig(hidden_sizes=(8,), activation="relu", dropout=None)
    policy = ValueBasedPolicy(obs_dim=4, n_actions=3, config=cfg)
    with pytest.raises(ValueError):
        policy.act(np.zeros(5, dtype=np.float32))

