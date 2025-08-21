"""Compatibility adapter producing a dataset similar to the private BOT_v5.

The original bot emitted plain text files with Spanish column names.  For the
public project we keep a reduced mapping so that downstream components used for
reinforcement learning can consume a compact parquet file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd


LEGACY_COLUMNS = {
    "thr_entry_up": "porcentaje_subida_minima_para_operar",
    "thr_exit_down": "porcentaje_bajada_minima_para_cerrar",
    "ask_peak_price": "resistencia_detectada",
    "last": "precio_ultimo",
}


def to_legacy_row(micro_row: Dict[str, Any]) -> Dict[str, Any]:
    """Map a Micro V5 row to the legacy BOT_v5 naming scheme."""

    out = {legacy: micro_row.get(src) for src, legacy in LEGACY_COLUMNS.items()}
    # Additional derived fields used by the old bot
    out["precio_minimo_para_operar"] = micro_row.get("last") * (1 + micro_row.get("thr_entry_up", 0))
    out["precio_maximo_para_cerrar"] = micro_row.get("last") * (1 - micro_row.get("thr_exit_down", 0))
    return out


def export_legacy_dataset(parquet_path: str | Path, out_path: str | Path) -> Path:
    """Read a ``micro_v5`` parquet and export a legacy view."""

    df = pd.read_parquet(parquet_path)
    rows = [to_legacy_row(rec._asdict() if hasattr(rec, "_asdict") else rec) for rec in df.to_dict("records")]
    out_df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_path

