from __future__ import annotations
import numpy as np
from .deterministic import DeterministicPolicy

class StochasticPolicy:
    def __init__(self, base_threshold: float = 0.001, epsilon: float = 0.1, seed: int = 42):
        self.det = DeterministicPolicy(threshold=base_threshold)
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 3))
        return self.det.act(obs)
