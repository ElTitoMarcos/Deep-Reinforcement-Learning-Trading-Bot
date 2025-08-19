from __future__ import annotations

"""Reusable neural network building blocks for policies."""

from typing import Iterable, Sequence

import torch
import torch.nn as nn


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
    None: nn.Identity,
}


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Parameters
    ----------
    in_dim:
        Size of the input features.
    hidden_sizes:
        Iterable with the number of units for each hidden layer.
    out_dim:
        Size of the final output layer.
    activation:
        Either a single activation name applied to all hidden layers or a
        sequence providing one activation per hidden layer. Supported names are
        ``relu``, ``tanh``, ``sigmoid``, ``gelu`` and ``leaky_relu``.
    dropout:
        Optional dropout probability applied after each activation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Iterable[int],
        out_dim: int,
        *,
        activation: str | Sequence[str] = "relu",
        dropout: float | None = None,
    ) -> None:
        super().__init__()

        hidden_sizes = list(hidden_sizes)
        if isinstance(activation, str):
            acts = [activation] * len(hidden_sizes)
        else:
            acts = list(activation)
            if len(acts) != len(hidden_sizes):
                raise ValueError("activation list must match hidden_sizes length")

        layers: list[nn.Module] = []
        last = in_dim
        for h, act_name in zip(hidden_sizes, acts):
            linear = nn.Linear(last, h)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            act_cls = _ACTIVATIONS.get(act_name.lower() if isinstance(act_name, str) else act_name)
            if act_cls is None:
                raise KeyError(f"Unknown activation '{act_name}'")
            layers.append(act_cls())
            if dropout:
                layers.append(nn.Dropout(dropout))
            last = h

        out = nn.Linear(last, out_dim)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

