"""Meta-controller to map algorithms to sub-tasks."""
from __future__ import annotations

from typing import Dict, Any
import logging


class AlgoController:
    """Assign RL algorithms to trading subtasks.

    The controller uses simple heuristics based on stage information, data
    profile and algorithm stability metrics.  It can optionally be *fixed* to
    keep returning the last mapping, useful for debugging from the UI.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self.llm = llm
        self.last_mapping: Dict[str, str] = {
            "entries_exits": "dqn",
            "risk_limits": "ppo",
            "position_sizing": "ppo",
        }
        self.fixed = False

    # ------------------------------------------------------------------
    def decide(
        self,
        stage_info: Dict[str, Any],
        data_profile: Dict[str, Any],
        stability: Dict[str, Any],
    ) -> Dict[str, str]:
        """Return mapping ``subtask -> algo`` according to heuristics."""

        if self.fixed:
            return self.last_mapping

        mapping = dict(self.last_mapping)

        vol = float(data_profile.get("volatility", 0.0))
        intraminute = bool(stage_info.get("intraminute", False))
        if intraminute and vol > 0.02:
            mapping["entries_exits"] = "dqn"
        else:
            mapping["entries_exits"] = "ppo"

        td_error = float(stability.get("td_error", 0.0))
        q_abs = abs(float(stability.get("q_abs", 0.0)))
        dd = float(stability.get("drawdown", 0.0))
        if td_error > 1.0 or q_abs > 1e3 or dd > 0.2:
            mapping["risk_limits"] = "ppo"
            mapping["position_sizing"] = "ppo"

        activity = float(data_profile.get("activity", 0.0))
        if activity > 1.0:
            mapping["risk_limits"] = "ppo"
            mapping["position_sizing"] = "ppo"

        if mapping != self.last_mapping:
            reason = self.explain(mapping)
            msg = (
                "mapping: entries_exits={entry}, risk_limits={risk}, sizing={size} (motivo: {reason})"
            ).format(
                entry=mapping.get("entries_exits"),
                risk=mapping.get("risk_limits"),
                size=mapping.get("position_sizing"),
                reason=reason,
            )
            logging.getLogger().info(msg, extra={"kind": "algo_controller"})
            self.last_mapping = mapping
        return mapping

    # ------------------------------------------------------------------
    def explain(self, mapping: Dict[str, str]) -> str:
        """Return human readable explanation for ``mapping``."""

        parts: list[str] = []
        if mapping.get("entries_exits") == "dqn":
            parts.append("DQN maneja las entradas y salidas por rapidez")
        else:
            parts.append("PPO maneja las entradas y salidas por estabilidad")
        if mapping.get("risk_limits") == "ppo":
            parts.append("PPO controla los límites de riesgo")
        else:
            parts.append("DQN controla los límites de riesgo")
        if mapping.get("position_sizing") == "ppo":
            parts.append("PPO ajusta el tamaño de posiciones")
        else:
            parts.append("DQN ajusta el tamaño de posiciones")
        return "; ".join(parts)


__all__ = ["AlgoController"]
