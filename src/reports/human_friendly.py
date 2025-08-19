from __future__ import annotations

"""Render evaluation metrics in a human friendly way.

This module provides two helpers:
- ``render_panel`` draws a Streamlit summary panel with tooltips.
- ``write_readme`` persists a short Markdown summary for the run.
"""

from pathlib import Path
from typing import Mapping, Sequence, Any

import numpy as np


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
