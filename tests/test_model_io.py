import os
import json
import pytest
import torch

from src.policies.value_based import ValueBasedPolicy, DQNConfig


def test_save_and_load_model(tmp_path):
    cfg = DQNConfig(hidden_sizes=(4,))
    policy = ValueBasedPolicy(obs_dim=3, n_actions=2, config=cfg)
    # deterministically set weights
    for p in policy.q_net.parameters():
        torch.nn.init.constant_(p, 0.5)
    path = tmp_path / "model.pt"
    policy.save_model(str(path))

    assert path.exists()
    meta_path = tmp_path / "model.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["obs_dim"] == 3
    assert "config_hash" in meta and "timestamp" in meta

    # load into new policy and verify weights
    policy2 = ValueBasedPolicy(obs_dim=3, n_actions=2, config=cfg)
    policy2.load_model(str(path))
    for p in policy2.q_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.5))

    # mismatched observation dimension should raise
    policy3 = ValueBasedPolicy(obs_dim=4, n_actions=2, config=cfg)
    with pytest.raises(ValueError):
        policy3.load_model(str(path))
