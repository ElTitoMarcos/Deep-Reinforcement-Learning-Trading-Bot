from __future__ import annotations

"""Render evaluation metrics in a human friendly way.

This module provides two helpers:
- ``render_panel`` draws a Streamlit summary panel with tooltips.
- ``write_readme`` persists a short Markdown summary for the run.
"""

from pathlib import Path
from typing import Mapping


def _pnl_face(pnl: float) -> str:
    """Return an emoji representing profit sentiment."""
    if pnl > 0.05:
        return "\U0001F642"  # 🙂
    if pnl > -0.05:
        return "\U0001F610"  # 😐
    return "\U0001F641"      # 🙁


def _dd_light(drawdown: float) -> str:
    """Return a traffic light emoji for max drawdown."""
    if drawdown < 0.05:
        return "\U0001F7E2"  # green circle
    if drawdown < 0.15:
        return "\U0001F7E1"  # yellow circle
    return "\U0001F534"      # red circle


def write_readme(metrics: Mapping[str, float], run_dir: Path) -> None:
    """Write a short Markdown summary with friendly names."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    pnl = metrics.get("pnl", 0.0)
    dd = metrics.get("max_drawdown", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    hit = metrics.get("hit_ratio", 0.0)
    turn = metrics.get("turnover", 0.0)

    lines = [
        "# Resumen amigable",
        "",
        f"- Ganancia total: {pnl*100:.2f}% {_pnl_face(pnl)}",
        f"- Caída máxima: {dd*100:.2f}% {_dd_light(dd)}",
        f"- Consistencia: {sharpe:.2f} (estabilidad de resultados)",
        f"- Acierto: {hit*100:.2f}% (de cada 10, acierta {hit*10:.1f})",
        f"- Actividad: {turn:.2f} (cuánto mueve el bot)",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def render_panel(metrics: Mapping[str, float]) -> None:
    """Render a Streamlit panel summarising metrics with tooltips."""
    import streamlit as st

    pnl = metrics.get("pnl", 0.0)
    dd = metrics.get("max_drawdown", 0.0)
    sharpe = metrics.get("sharpe", 0.0)
    hit = metrics.get("hit_ratio", 0.0)
    turn = metrics.get("turnover", 0.0)

    st.subheader("Resumen amigable")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "Ganancia total",
            f"{pnl*100:.2f}% {_pnl_face(pnl)}",
            help="Rentabilidad total del periodo",
        )
        st.metric(
            "Caída máxima",
            f"{dd*100:.2f}% {_dd_light(dd)}",
            help="Pérdida máxima desde un pico; color indica riesgo",
        )
        st.metric(
            "Consistencia",
            f"{sharpe:.2f}",
            help="Qué tan estables son los resultados; mayor es mejor",
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
