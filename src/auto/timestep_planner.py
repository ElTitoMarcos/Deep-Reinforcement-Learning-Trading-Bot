from __future__ import annotations

"""Heuristics to choose training timesteps per stage."""

import json
from typing import Any, Dict, List


def plan_timesteps(
    stage_info: Dict[str, Any],
    data_rate: float,
    stability: Dict[str, float],
    prev_runs: List[int],
    llm: Any | None = None,
) -> int:
    """Return the planned number of timesteps for the next training block.

    Parameters
    ----------
    stage_info:
        Dictionary from :class:`~src.auto.stage_scheduler.StageScheduler` with
        at least the key ``"stage"``.
    data_rate:
        Approximate rate of useful learning signals (0-1).
    stability:
        Metrics such as ``td_var`` or ``drawdown`` summarising recent
        performance.
    prev_runs:
        Historical timesteps used for previous blocks. It will be appended with
        the chosen value.
    llm:
        Optional LLM client with an ``ask(system, prompt)`` method. If provided
        it may adjust the heuristic by ±30%.
    """

    stage = stage_info.get("stage", "warmup")
    reason_parts: list[str] = []

    if stage == "warmup":
        ts = 10_000
        reason_parts.append("fase inicial")
    elif stage == "exploration":
        if data_rate > 0.5:
            ts = 80_000
            reason_parts.append("señales de aprendizaje")
        else:
            ts = 50_000
            reason_parts.append("pocas señales")
    elif stage == "consolidation":
        score_up = stability.get("score_trend", 0.0) > 0
        stable = stability.get("td_var", 1.0) < 0.02
        if score_up and stable:
            ts = 200_000
            reason_parts.append("score↑ y estabilidad")
        else:
            ts = 120_000
            reason_parts.append("monitoreo")
    else:  # fine-tune or fallback
        ts = 50_000
        reason_parts.append("fase estándar")

    if stability.get("drawdown", 0.0) > 0.1 or stability.get("overfit"):
        ts = int(ts * 0.5)
        reason_parts.append("recortado por riesgo")

    if llm is not None:
        context = {
            "stage": stage,
            "base": ts,
            "data_rate": data_rate,
            "stability": stability,
            "prev_runs": prev_runs[-3:],
        }
        try:  # pragma: no cover - network access
            resp = llm.ask("timestep planner", json.dumps(context))
            data = json.loads(resp)
            factor = float(data.get("factor", 1.0))
            factor = max(0.7, min(1.3, factor))
            ts = int(ts * factor)
            reason_parts.append(f"ajuste LLM {factor:.2f}x")
        except Exception:
            reason_parts.append("LLM falló")

    prev_runs.append(ts)
    plan_timesteps.last_reason = ", ".join(reason_parts)
    return ts


__all__ = ["plan_timesteps"]
