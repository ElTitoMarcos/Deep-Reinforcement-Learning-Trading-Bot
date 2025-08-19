from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from .nets import MLP
from .replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    buffer_size: int = 10000
    batch_size: int = 64
    target_update: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "relu"
    dropout: float | None = None
    model_path: str = "models/dqn.pt"
    device: str = "cpu"
    seed: int = 0


class ValueBasedPolicy:
    """Simple value-based policy using a DQN-style network."""

    def __init__(self, obs_dim: int, n_actions: int, config: dict | DQNConfig | None = None):
        if config is None:
            config = DQNConfig()
        elif isinstance(config, dict):
            config = DQNConfig(**config)
        self.cfg = config
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)

        self.q_net = MLP(
            obs_dim,
            config.hidden_sizes,
            n_actions,
            activation=config.activation,
            dropout=config.dropout,
        ).to(self.device)
        self.target_net = MLP(
            obs_dim,
            config.hidden_sizes,
            n_actions,
            activation=config.activation,
            dropout=config.dropout,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=config.learning_rate)
        self.buffer = ReplayBuffer(config.buffer_size, seed=config.seed)

        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.eps_end = config.epsilon_end
        self.eps_decay = (config.epsilon_start - config.epsilon_end) / max(1, config.epsilon_decay_steps)
        self.target_update = config.target_update
        self.step_count = 0
        self.model_path = config.model_path

        self.load(self.model_path)  # load weights if available

    # --------- Inference ---------
    def act(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.shape != (self.obs_dim,):
            raise ValueError(f"expected obs shape {(self.obs_dim,)}, got {obs_arr.shape}")
        if np.random.rand() < self.epsilon:
            action = int(np.random.randint(0, self.n_actions))
        else:
            obs_t = torch.from_numpy(obs_arr).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.q_net(obs_t)
            assert q_vals.shape == (1, self.n_actions), "q-net output shape mismatch"
            action = int(torch.argmax(q_vals, dim=1).item())
        # decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay)
        return action

    # --------- Memory ---------
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # --------- Training ---------
    def train_step(self) -> float | None:
        if len(self.buffer) < self.cfg.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.cfg.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_next = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * max_next
        loss = F.mse_loss(q, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    # --------- Persistence ---------
    def save(self, path: str | None = None) -> None:
        path = path or self.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str | None = None) -> None:
        path = path or self.model_path
        if os.path.exists(path):
            self.q_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.q_net.state_dict())
