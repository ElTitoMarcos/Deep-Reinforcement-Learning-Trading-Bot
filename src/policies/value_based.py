from __future__ import annotations
from collections import deque
import numpy as np
from typing import Tuple

class TinyDQN:
    """Minimal DQN skeleton: NOT for production. For smoke tests and interface only."""
    def __init__(self, obs_dim: int, n_actions: int, gamma: float = 0.99, lr: float = 1e-3, buffer_size: int = 10000, batch_size: int = 64, seed: int = 42):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.buffer = deque(maxlen=buffer_size)
        # Tabular-ish linear weights
        self.W = self.rng.normal(scale=0.01, size=(obs_dim, n_actions))

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return obs @ self.W  # linear

    def select_action(self, obs: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q = self.q_values(obs)
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def learn(self, steps: int = 1):
        if len(self.buffer) < self.batch_size:
            return
        for _ in range(steps):
            idx = self.rng.choice(len(self.buffer), size=self.batch_size, replace=False)
            batch = [self.buffer[i] for i in idx]
            S = np.stack([b[0] for b in batch])
            A = np.array([b[1] for b in batch], dtype=int)
            R = np.array([b[2] for b in batch], dtype=float)
            S2 = np.stack([b[3] for b in batch])
            D = np.array([b[4] for b in batch], dtype=bool)
            Q = S @ self.W
            Q2 = S2 @ self.W
            target = Q.copy()
            max_next = np.max(Q2, axis=1)
            for i in range(self.batch_size):
                target[i, A[i]] = R[i] + (0.0 if D[i] else self.gamma * max_next[i])
            # simple gradient step
            grad = (Q - target) / self.batch_size
            self.W -= self.lr * (S.T @ grad)
