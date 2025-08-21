from __future__ import annotations

from queue import Queue, Empty
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils import paths
from src.reports.human_friendly import kpi_humano


def _load_timeline(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "timeline.jsonl"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _drain_queue(queue: Queue | None) -> list[dict]:
    items: list[dict] = []
    if queue is None:
        return items
    while True:
        try:
            items.append(queue.get_nowait())
        except Empty:
            break
    return items


def _render(df: pd.DataFrame) -> None:
    if df.empty:
        st.write("Sin datos")
        return
    latest = df.iloc[-1].to_dict()
    kpis = kpi_humano(latest)
    cols = st.columns(len(kpis))
    for col, (name, val) in zip(cols, kpis.items()):
        col.metric(name, f"{val:.4f}")
    st.line_chart(df.set_index("steps")["pnl"])
    if "orders" in df:
        st.line_chart(df["orders"].diff().fillna(0))


def show(queue: Queue | None, run_id: str) -> None:
    """Render training/backtest progress in real time."""
    run_dir = paths.reports_dir() / f"run_{run_id}"
    train_tab, eval_tab = st.tabs(["Entrenamiento", "Evaluación"])
    with train_tab:
        df = _load_timeline(run_dir)
        updates = _drain_queue(queue)
        if updates:
            df = pd.concat([df, pd.DataFrame(updates)], ignore_index=True)
        _render(df)
    with eval_tab:
        df = _load_timeline(run_dir)
        _render(df)
