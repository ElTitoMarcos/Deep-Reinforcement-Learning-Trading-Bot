from __future__ import annotations
import torch.nn as nn

class MLP(nn.Module):
    """Simple multi-layer perceptron"""
    def __init__(self, in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
