from __future__ import annotations

from typing import Dict, Any
import numpy as np
import os
import re

# Optional import for LLM assistance
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    openai = None  # type: ignore


class HybridPolicy:
    """Blend multiple policies with adaptive weights.

    Parameters
    ----------
    policies : Dict[str, Any]
        Mapping from policy name to policy instance. Each policy must expose an
        ``act`` method compatible with the simulator.
    initial_weights : Dict[str, float]
        Starting weights for each policy. Missing entries default to uniform
        values.
    block_size : int, optional
        Number of trades after which weights are refreshed. Defaults to 100.
    """

    def __init__(
        self,
        policies: Dict[str, Any],
        initial_weights: Dict[str, float],
        block_size: int = 100,
    ):
        self.policies = policies
        if not policies:
            raise ValueError("At least one policy is required")
        n = len(policies)
        # normalise weights and ensure every policy has an entry
        weights = np.array([initial_weights.get(k, 1.0 / n) for k in policies], dtype=float)
        weights /= weights.sum()
        self.weights = {k: float(w) for k, w in zip(policies, weights)}
        # store history for analysis/regularisation
        self.metrics_history: list[Dict[str, Any]] = []
        self.block_size = int(block_size)
        self._pending: list[Dict[str, Dict[str, float]]] = []

    # ------------------------------------------------------------------
    def record_trade(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Accumulate trade metrics and update weights every ``block_size`` trades."""

        self._pending.append(metrics)
        if len(self._pending) >= self.block_size:
            agg: Dict[str, Dict[str, float]] = {}
            for m in self._pending:
                for name, vals in m.items():
                    entry = agg.setdefault(name, {"pnl": 0.0, "max_drawdown": 0.0, "n": 0})
                    entry["pnl"] += float(vals.get("pnl", 0.0))
                    entry["max_drawdown"] += float(vals.get("max_drawdown", 0.0))
                    entry["n"] += 1
            averaged = {
                k: {"pnl": v["pnl"] / max(v["n"], 1), "max_drawdown": v["max_drawdown"] / max(v["n"], 1)}
                for k, v in agg.items()
            }
            self.update_weights(averaged)
            self._pending.clear()

    # ------------------------------------------------------------------
    def act(self, obs: np.ndarray) -> int:
        """Return an action using the current weight mixture.

        The policy index is sampled according to ``self.weights`` and the
        underlying policy's ``act`` method is invoked. If the wrapped policy
        returns ``(action, ...)`` only the first element is used.
        """

        names = list(self.policies)
        probs = np.array([self.weights[n] for n in names], dtype=float)
        probs /= probs.sum()
        chosen = np.random.choice(names, p=probs)
        res = self.policies[chosen].act(obs)
        if isinstance(res, tuple):
            return int(res[0])
        return int(res)

    # ------------------------------------------------------------------
    def _heuristic_weights(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Simple heuristic: favour higher PnL and lower drawdown."""

        scores = {}
        for name, m in metrics.items():
            pnl = float(m.get("pnl", 0.0))
            dd = abs(float(m.get("max_drawdown", 0.0)))
            scores[name] = pnl - 0.5 * dd  # penalise drawdown
        vals = np.array([scores[n] for n in self.policies], dtype=float)
        vals -= vals.min()
        vals += 1e-6  # keep positive
        vals /= vals.sum()
        return {k: float(v) for k, v in zip(self.policies, vals)}

    def update_weights(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """Update mixture weights based on recent performance metrics.

        ``metrics`` should map policy names to dictionaries containing at least
        ``pnl`` and ``max_drawdown``.  A simple exponentially smoothed update is
        applied to avoid oscillations.  When ``OPENAI_API_KEY`` is available,
        the metrics summary is sent to an LLM which may return suggested
        weights.  Any failure in the LLM call gracefully falls back to the
        heuristic update.
        """

        self.metrics_history.append(metrics)
        target = None

        if openai is not None and os.getenv("OPENAI_API_KEY"):
            try:  # pragma: no cover - network interaction
                openai.api_key = os.getenv("OPENAI_API_KEY")
                summary = ", ".join(
                    f"{k}: pnl={v.get('pnl',0):.4f} dd={v.get('max_drawdown',0):.4f}"
                    for k, v in metrics.items()
                )
                prompt = (
                    "Sugerir pesos normalizados para las políticas dada la siguiente "
                    f"info: {summary}. Responde solo una lista de números "
                    "separados por comas."
                )
                resp = openai.ChatCompletion.create(  # type: ignore
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                )
                text = resp["choices"][0]["message"]["content"]
                nums = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", text)]
                if len(nums) == len(self.policies):
                    target = {k: max(0.0, n) for k, n in zip(self.policies, nums)}
            except Exception:
                target = None

        if target is None:
            target = self._heuristic_weights(metrics)

        # normalise and apply exponential moving average to smooth updates
        arr = np.array([target[k] for k in self.policies], dtype=float)
        arr /= arr.sum()
        beta = 0.3  # update rate
        for k, new_w in zip(self.policies, arr):
            self.weights[k] = float((1 - beta) * self.weights[k] + beta * new_w)

