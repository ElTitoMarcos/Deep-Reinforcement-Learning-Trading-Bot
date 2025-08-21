from __future__ import annotations

"""Render evaluation metrics in a human friendly way.

This module provides two helpers:
- ``render_panel`` draws a Streamlit summary panel with tooltips.
- ``write_readme`` persists a short Markdown summary for the run.
"""

from pathlib import Path
from typing import Mapping, Sequence, Any, Dict

import numpy as np


# Basic translations for log kinds to more readable Spanish phrases
_KIND_TRANSLATIONS = {
    "datos": "Evento de datos",
    "incremental_update": "Actualización incremental completada",
    "qc": "Validación de datos completada",
    "reward_tuner": "Ajuste de recompensas",
    "dqn_stability": "Revisión de estabilidad del DQN",
    "checkpoints": "Checkpoint guardado",
    "hybrid_weights": "Pesos híbridos actualizados",
    "performance": "Evaluación de performance",
    "llm": "Mensaje del LLM",
    "riesgo": "Aviso de riesgo",
}


def _pnl_light(pnl: float) -> str:
    """Return a traffic light emoji for profit."""
    if pnl > 0.05:
        return "\U0001F7E2"  # green circle
    if pnl > -0.05:
        return "\U0001F7E1"  # yellow circle
    return "\U0001F534"      # red circle


def _dd_light(drawdown: float) -> str:
    """Return a traffic light emoji for max drawdown."""
    if drawdown < 0.05:
        return "\U0001F7E2"  # green circle
    if drawdown < 0.15:
        return "\U0001F7E1"  # yellow circle
    return "\U0001F534"      # red circle


def write_readme(results: Mapping[str, Any], run_dir: Path) -> None:
    """Write a short Markdown summary with friendly names."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pnl = float(results.get("pnl", 0.0))
    dd = float(results.get("max_drawdown", 0.0))
    rets: Sequence[float] | None = results.get("returns")  # type: ignore[assignment]
    if rets is not None:
        arr = np.asarray(rets, dtype=float)
        consistency = float(arr.mean() / (arr.std(ddof=0) + 1e-12))
    else:
        consistency = float(results.get("sharpe", 0.0))
    hit = float(results.get("hit_ratio", 0.0))
    turn = float(results.get("turnover", 0.0))

    lines = [
        "# Resumen amigable",
        "",
        f"- Ganancia total: {pnl*100:.2f}% {_pnl_light(pnl)}",
        f"- Caída máxima: {dd*100:.2f}% {_dd_light(dd)}",
        f"- Consistencia: {consistency:.2f} (media/vol de retornos)",
        f"- Acierto: {hit*100:.2f}% (de cada 10, acierta {hit*10:.1f})",
        f"- Actividad: {turn:.2f} (cuánto mueve el bot)",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def kpi_humano(metrics: Mapping[str, Any]) -> Dict[str, float]:
    """Return key performance indicators with friendly Spanish names."""

    pnl = float(metrics.get("pnl", 0.0))
    dd = float(metrics.get("dd", metrics.get("max_drawdown", 0.0)))
    rets: Sequence[float] | None = metrics.get("returns")  # type: ignore[assignment]
    if rets is not None:
        arr = np.asarray(rets, dtype=float)
        consistency = float(arr.mean() / (arr.std(ddof=0) + 1e-12))
    else:
        consistency = float(metrics.get("sharpe", 0.0))
    hit = float(metrics.get("hit", metrics.get("hit_ratio", 0.0)))
    turn = float(metrics.get("orders", metrics.get("turnover", 0.0)))

    return {
        "Ganancia total": pnl,
        "Caída máxima": dd,
        "Consistencia": consistency,
        "Acierto": hit,
        "Actividad": turn,
    }


def render_panel(results: Mapping[str, Any]) -> None:
    """Render a Streamlit panel summarising metrics with tooltips."""
    import streamlit as st
    pnl = float(results.get("pnl", 0.0))
    dd = float(results.get("max_drawdown", 0.0))
    rets: Sequence[float] | None = results.get("returns")  # type: ignore[assignment]
    if rets is not None:
        arr = np.asarray(rets, dtype=float)
        consistency = float(arr.mean() / (arr.std(ddof=0) + 1e-12))
    else:
        consistency = float(results.get("sharpe", 0.0))
    hit = float(results.get("hit_ratio", 0.0))
    turn = float(results.get("turnover", 0.0))

    st.subheader("Resumen amigable")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Ganancia total",
            f"{pnl*100:.2f}% {_pnl_light(pnl)}",
            help="Rentabilidad total del periodo",
        )
        st.metric(
            "Caída máxima",
            f"{dd*100:.2f}% {_dd_light(dd)}",
            help="Pérdida máxima desde un pico; color indica riesgo",
        )
        st.metric(
            "Consistencia",
            f"{consistency:.2f}",
            help="media/vol de retornos; mayor es mejor",
        )
    with c2:
        st.metric(
            "Acierto",
            f"{hit*100:.2f}%",
            help=f"de cada 10, acierta {hit*10:.1f}",
        )
        st.metric(
            "Actividad",
            f"{turn:.2f}",
            help="Cuánto opera el bot",
        )


def episode_sentence(metrics: Mapping[str, Any]) -> str:
    """Return a short friendly sentence summarising recent performance.

    Parameters
    ----------
    metrics:
        Mapping containing at least ``pnl``, ``consistency`` and ``turnover``
        values.  Optionally ``window`` may describe the lookback period.
    """

    window = str(metrics.get("window", "la última hora"))
    pnl = float(metrics.get("pnl", 0.0))
    cons = float(metrics.get("consistency", 0.0))
    turn = float(metrics.get("turnover", 0.0))

    pnl_str = f"{pnl*100:+.1f}%"

    if cons > 1.0:
        cons_lbl = "alta"
    elif cons > 0.5:
        cons_lbl = "media"
    else:
        cons_lbl = "baja"

    if turn > 1.0:
        act_lbl = "alta"
    elif turn > 0.2:
        act_lbl = "moderada"
    else:
        act_lbl = "baja"

    return f"Acumulas {pnl_str} en {window}; consistencia {cons_lbl}; actividad {act_lbl}."


def to_human(msg: Mapping[str, Any]) -> str:
    """Translate a log message mapping into a short human friendly sentence."""

    # If a pre-rendered message is present, prefer it
    text = msg.get("message")
    if isinstance(text, str) and text:
        return text

    kind = str(msg.get("kind", ""))
    if kind in _KIND_TRANSLATIONS:
        return _KIND_TRANSLATIONS[kind]

    event = str(msg.get("event", ""))
    if event in _KIND_TRANSLATIONS:
        return _KIND_TRANSLATIONS[event]

    return str(text or kind or event or "")

