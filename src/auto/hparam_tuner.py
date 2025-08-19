"""Lightweight hyperparameter tuning utilities."""
from __future__ import annotations

import random
from typing import Dict, Sequence


def _sample_config() -> Dict[str, float | int]:
    """Sample a random hyperparameter configuration."""
    return {
        "learning_rate": 10 ** random.uniform(-5, -3),
        "batch_size": random.choice([32, 64, 128, 256]),
        "n_steps": random.choice([128, 256, 512, 1024]),
    }


def tune(algo: str, data_stats: Dict | None, prev_runs: Sequence[Dict] | None) -> Dict:
    """Return recommended hyperparameters for ``algo``.

    A tiny random search (10-30 samples) is performed with an early stop when
    no improvement is seen for five consecutive samples.  The objective is a
    stand-in for validation reward based on random numbers which is sufficient
    for lightweight suggestions in tests and examples.
    """
    if algo == "hybrid":
        # Tune each component separately
        return {
            "ppo": tune("ppo", data_stats, prev_runs),
            "dqn": tune("dqn", data_stats, prev_runs),
        }

    samples = random.randint(10, 30)
    best_cfg: Dict[str, float | int] | None = None
    best_reward = float("-inf")
    no_improve = 0
    for _ in range(samples):
        cfg = _sample_config()
        reward = random.random()
        if reward > best_reward:
            best_reward = reward
            best_cfg = cfg
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:
                break
    assert best_cfg is not None  # for type checkers
    return best_cfg
