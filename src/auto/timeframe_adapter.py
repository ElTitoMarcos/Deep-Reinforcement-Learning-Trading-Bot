"""Simple timeframe adaptation heuristics."""

from __future__ import annotations

from typing import Dict, Any
import json


def propose_timeframe(
    stats: Dict[str, Any],
    vol_profile: Dict[str, float],
    latency_budget: float,
    llm: Any | None = None,
) -> Dict[str, str]:
    """Return a timeframe suggestion based on recent stats.

    Parameters
    ----------
    stats:
        Dictionary with keys ``recent_volatility``, ``gap_ratio``, ``device``,
        ``batch_size``, ``base_tf`` and ``current_tf``.
    vol_profile:
        Thresholds with ``high`` and ``low`` volatility markers.
    latency_budget:
        Available compute budget on a 0-1 scale where higher allows finer
        resampling.
    llm:
        Optional LLM client implementing ``ask``. When provided, the heuristic
        proposal is sent for feedback and may be refined.
    """

    base_tf = stats.get("base_tf", "1m")
    current_tf = stats.get("current_tf", base_tf)
    vol = float(stats.get("recent_volatility", 0.0))
    gap_ratio = float(stats.get("gap_ratio", 0.0))

    # Heuristic proposal -------------------------------------------------
    if gap_ratio > 0.3:
        resample = current_tf
        reason = "muchos huecos"
    elif vol > vol_profile.get("high", 0.02):
        resample = "5s" if latency_budget >= 1.0 else "15s"
        reason = "alta volatilidad"
    elif vol < vol_profile.get("low", 0.005):
        resample = "30s"
        reason = "baja volatilidad"
    else:
        resample = "15s"
        reason = "volatilidad moderada"

    proposal = {"base_tf": base_tf, "resample_to": resample, "reason": reason}

    # LLM refinement -----------------------------------------------------
    if llm is not None:
        ctx = (
            f"vol={vol:.4f}, gaps={gap_ratio:.2f}, actual={current_tf},"
            f" device={stats.get('device')}, batch={stats.get('batch_size')}"
        )
        prompt = (
            "Sugiere timeframe (5s,15s,30s,1m) para entrenamiento dado: "
            f"{ctx}. Devuelve JSON con claves 'resample_to' y 'reason'."
        )
        try:  # pragma: no cover - network interaction
            resp = llm.ask("Asistente timeframe", prompt)
            data = json.loads(resp)
            cand = data.get("resample_to")
            if cand in {"5s", "15s", "30s", "1m"}:
                proposal["resample_to"] = cand
                proposal["reason"] = data.get("reason", "llm")
        except Exception:
            pass

    return proposal


__all__ = ["propose_timeframe"]

