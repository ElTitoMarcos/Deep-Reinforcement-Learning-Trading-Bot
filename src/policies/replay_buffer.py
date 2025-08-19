from __future__ import annotations
from collections import deque
from typing import Deque, Tuple
import numpy as np

class ReplayBuffer:
    """Basic replay buffer for value-based methods."""
    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in idxs))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)
