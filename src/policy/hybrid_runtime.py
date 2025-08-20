"""Runtime wrapper combining DQN signals with PPO risk controls."""
from __future__ import annotations

from typing import Any, Dict

import logging
from src.auto import AlgoController


class HybridRuntime:
    """Route actions through specialised algorithms based on controller mapping."""

    def __init__(self, dqn_signal: Any, ppo_control: Any, controller: AlgoController) -> None:
        self.dqn_signal = dqn_signal
        self.ppo_control = ppo_control
        self.controller = controller

    # ------------------------------------------------------------------
    def act(
        self,
        obs: Any,
        stage_info: Dict | None = None,
        data_profile: Dict | None = None,
        stability: Dict | None = None,
    ) -> Any:
        """Return action after applying signal and control algorithms."""

        mapping = self.controller.decide(
            stage_info or {}, data_profile or {}, stability or {}
        )
        if mapping.get("entries_exits") == "dqn" or not hasattr(self.ppo_control, "act"):
            signal = self.dqn_signal.act(obs)
        else:
            signal = self.ppo_control.act(obs)

        action = signal
        if mapping.get("risk_limits") == "ppo" or mapping.get("position_sizing") == "ppo":
            action = self.ppo_control.filter(signal, obs)
            if action != signal:
                logging.getLogger().info(
                    "control ajusta qty %s→%s price %s→%s",
                    signal.get("qty"),
                    action.get("qty"),
                    signal.get("price"),
                    action.get("price"),
                    extra={"kind": "ppo_control"},
                )
        return action


__all__ = ["HybridRuntime"]
