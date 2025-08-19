from __future__ import annotations

"""Simple data validation utilities."""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class QualityReport:
    """Aggregated quality information."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_ohlcv(df: pd.DataFrame) -> QualityReport:
    report = QualityReport()
    if df is None or df.empty:
        report.errors.append("sin datos OHLCV")
        return report

    if not df.index.is_monotonic_increasing:
        report.errors.append("timestamps no crecientes")

    if df.index.duplicated().any():
        report.errors.append("timestamps duplicados")

    if (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
    ).any():
        report.errors.append("OHLC inconsistentes")

    diffs = df.index.to_series().diff().dropna()
    if not diffs.empty:
        expected = diffs.mode().iloc[0]
        if (diffs > expected * 1.5).any():
            report.warnings.append("huecos en timestamps")

    return report


def validate_metadata(meta: Dict[str, any]) -> QualityReport:
    report = QualityReport()
    if not meta:
        report.errors.append("metadata vacía")
        return report

    if meta.get("status") != "TRADING":
        report.errors.append("status no TRADING")

    filters = meta.get("filters", {})
    for key in ("PRICE_FILTER", "LOT_SIZE", "MIN_NOTIONAL"):
        val = filters.get(key)
        if val is None or (isinstance(val, (int, float)) and val <= 0):
            report.errors.append(f"filtro {key} inválido")

    return report


def validate_trades(df: pd.DataFrame) -> QualityReport:
    report = QualityReport()
    if df is None or df.empty:
        report.warnings.append("sin trades")
        return report

    if "timestamp" in df.columns:
        series = pd.to_datetime(df["timestamp"], unit="ms")
        if not series.is_monotonic_increasing:
            report.errors.append("trades fuera de orden")
    return report


def passes(report: QualityReport) -> bool:
    return not report.errors


def summarize(report: QualityReport) -> str:
    parts: List[str] = []
    if report.errors:
        parts.append("Errores: " + "; ".join(report.errors))
    if report.warnings:
        parts.append("Avisos: " + "; ".join(report.warnings))
    return ", ".join(parts) if parts else "OK"

