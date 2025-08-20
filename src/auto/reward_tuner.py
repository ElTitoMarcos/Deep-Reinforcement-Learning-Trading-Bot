"""Adaptive reward weight tuner based on local sensitivity and bandits."""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
import logging
from pathlib import Path
from typing import Dict, Tuple, Any


@dataclass
class _BanditArm:
    success: int = 1
    trials: int = 1

    def sample(self) -> float:
        """Sample from a Beta distribution for Thompson sampling."""
        return random.betavariate(self.success, self.trials - self.success)

    def update(self, success: bool) -> None:
        self.trials += 1
        if success:
            self.success += 1


class RewardTuner:
    """Propose small adjustments to reward weights using local sensitivity.

    The tuner keeps track of past modifications and evaluates them with a
    multi-armed bandit (one arm per weight-direction).  Proposals are stored in
    ``memory_file`` as JSON lines allowing persistence across runs.
    """

    def __init__(
        self,
        init_weights: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        memory_file: Path,
        delta: float = 0.05,
        score_weights: Dict[str, float] | None = None,
    ) -> None:
        self.weights = dict(init_weights)
        self.bounds = bounds
        self.memory_file = Path(memory_file)
        self.delta = float(delta)
        sw = score_weights or {}
        self.lambda_dd = float(sw.get("lambda_dd", 1.0))
        self.kappa_cons = float(sw.get("kappa_consistency", 0.5))
        self.mu_act = float(sw.get("mu_activity", 0.1))
        self.bandit: Dict[str, Dict[str, _BanditArm]] = {
            k: {"up": _BanditArm(), "down": _BanditArm()} for k in self.weights
        }
        self.last_action: Dict[str, Any] | None = None
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def score(self, metrics: Dict[str, float]) -> float:
        """Composite performance score.

        ``metrics`` should contain keys ``pnl``, ``drawdown``, ``consistency``
        and ``activity``.  Missing keys default to ``0.0``.
        """

        pnl = float(metrics.get("pnl", 0.0))
        dd = float(metrics.get("drawdown", 0.0))
        cons = float(metrics.get("consistency", 0.0))
        act = float(metrics.get("activity", 0.0))
        return pnl - self.lambda_dd * dd + self.kappa_cons * cons - self.mu_act * act

    # ------------------------------------------------------------------
    def propose(self, recent_metrics: Dict[str, float]) -> Dict[str, float]:
        """Return new weights given ``recent_metrics``.

        Local sensitivity is approximated from the sign and magnitude of the
        metrics.  A Thompson-sampling bandit decides whether to increase or
        decrease the chosen weight.
        """

        gradients = {
            "w_pnl": recent_metrics.get("pnl", 0.0),
            "w_drawdown": -recent_metrics.get("drawdown", 0.0),
            "w_volatility": -recent_metrics.get("volatility", 0.0),
            "w_turnover": -recent_metrics.get("turnover", 0.0),
        }
        mags = [abs(v) for v in gradients.values()]
        if sum(mags) == 0:
            key = random.choice(list(self.weights))
        else:
            r = random.random() * sum(mags)
            cum = 0.0
            key = list(self.weights)[0]
            for k, m in gradients.items():
                cum += abs(m)
                if r <= cum:
                    key = k
                    break

        arms = self.bandit[key]
        dir_up = arms["up"].sample()
        dir_down = arms["down"].sample()
        direction = "up" if dir_up >= dir_down else "down"

        factor = 1.0 + self.delta if direction == "up" else 1.0 - self.delta
        prev_weights = dict(self.weights)
        new_val = self.weights[key] * factor
        low, high = self.bounds.get(key, (float("-inf"), float("inf")))
        new_val = max(low, min(high, new_val))
        self.weights[key] = new_val

        self.last_action = {
            "key": key,
            "direction": direction,
            "prev_weights": prev_weights,
            "prev_metrics": recent_metrics,
            "score_before": self.score(
                {
                    "pnl": recent_metrics.get("pnl", 0.0),
                    "drawdown": recent_metrics.get("drawdown", 0.0),
                    "consistency": recent_metrics.get("consistency", 0.0),
                    "activity": recent_metrics.get("activity", 0.0),
                }
            ),
        }
        reason = f"grad={gradients[key]:.4f}"
        self.last_action["reason"] = reason

        delta_w = new_val - prev_weights[key]
        expected = gradients[key] * delta_w
        sens = "alta" if abs(gradients[key]) > 1 else "media" if abs(gradients[key]) > 0.1 else "baja"
        name = human_names().get(key, key)
        arrow = "↑" if delta_w > 0 else "↓" if delta_w < 0 else "→"
        msg = (
            f"{name} {arrow} {delta_w:+.2f} (sensibilidad {sens}; mejora esperada de score {expected:+.2f})"
        )
        logging.getLogger().info(msg, extra={"kind": "reward_tuner"})
        return dict(self.weights)

    # ------------------------------------------------------------------
    def confirm(self, new_metrics: Dict[str, float]) -> None:
        """Persist result of last proposal and update bandit statistics."""

        if not self.last_action:
            return
        key = self.last_action["key"]
        direction = self.last_action["direction"]
        prev_w = self.last_action["prev_weights"]
        score_before = self.last_action.get("score_before", 0.0)
        score_after = self.score(new_metrics)
        success = score_after >= score_before

        self.bandit[key][direction].update(success)
        record = {
            "before": prev_w,
            "after": dict(self.weights),
            "metrics_before": self.last_action.get("prev_metrics"),
            "metrics_after": new_metrics,
            "score_before": score_before,
            "score_after": score_after,
            "success": success,
            "direction": direction,
            "weight": key,
        }
        with self.memory_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        if not success:
            self.weights = prev_w

        self.last_action = None


def human_names() -> Dict[str, str]:
    """Return human friendly names for weight keys."""

    return {
        "w_pnl": "Beneficio",
        "w_drawdown": "Protección ante rachas",
        "w_volatility": "Suavidad",
        "w_turnover": "Control de actividad",
    }


__all__ = ["RewardTuner", "human_names"]
