from __future__ import annotations

import json
from queue import Queue
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # pragma: no cover - fallback when sb3 is missing
    class BaseCallback:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def _on_step(self) -> bool:
            return True

from ..utils import paths


class UiHeartbeatCallback(BaseCallback):
    """Callback that emits training metrics to a queue and timeline file."""

    def __init__(
        self,
        run_id: str,
        queue: Queue | None = None,
        *,
        every_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.queue = queue
        self.every_steps = every_steps
        self._run_dir = paths.reports_dir() / f"run_{run_id}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._timeline = self._run_dir / "timeline.jsonl"

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:  # type: ignore[override]
        if self.num_timesteps % self.every_steps != 0:
            return True

        def _get(attr: str, default: float = 0.0) -> float:
            try:
                val = self.training_env.get_attr(attr)[0]
                return float(val)
            except Exception:
                return float(default)

        steps = int(self.num_timesteps)
        reward_mean = _get("recent_reward_mean")
        pnl = _get("equity")
        peak = _get("equity_peak")
        dd = (peak - pnl) / (peak + 1e-12)
        hit = _get("hit_ratio")
        orders = _get("n_orders", 0.0)
        metrics = {
            "steps": steps,
            "reward_mean": reward_mean,
            "pnl": pnl,
            "dd": dd,
            "hit": hit,
            "orders": orders,
        }

        if self.queue is not None:
            try:
                self.queue.put_nowait(metrics)
            except Exception:
                pass

        try:
            with self._timeline.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(metrics) + "\n")
        except Exception:
            pass

        return True
