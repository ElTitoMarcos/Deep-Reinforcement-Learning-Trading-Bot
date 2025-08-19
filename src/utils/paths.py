from __future__ import annotations

"""Helper utilities for resolving common project directories."""

from pathlib import Path
from typing import Any, Mapping


def _paths(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``paths`` section from the config or an empty mapping."""
    return cfg.get("paths", {}) if isinstance(cfg, Mapping) else {}


def get_raw_dir(cfg: Mapping[str, Any]) -> Path:
    """Return the directory containing raw market data."""
    return Path(_paths(cfg).get("raw_dir", "data/raw"))


def get_reports_dir(cfg: Mapping[str, Any]) -> Path:
    """Return the directory where evaluation reports are stored."""
    return Path(_paths(cfg).get("reports_dir", "reports"))


def get_checkpoints_dir(cfg: Mapping[str, Any]) -> Path:
    """Return the directory for model checkpoints."""
    return Path(_paths(cfg).get("checkpoints_dir", "checkpoints"))


def ensure_dirs_exist(cfg: Mapping[str, Any]) -> None:
    """Create common project directories if they do not already exist."""
    for directory in [get_raw_dir(cfg), get_reports_dir(cfg), get_checkpoints_dir(cfg)]:
        directory.mkdir(parents=True, exist_ok=True)


def symbol_to_dir(symbol: str) -> str:
    """Return the on-disk representation for a trading pair.

    ``"ETH/USDT"`` -> ``"ETH-USDT"``
    """

    return symbol.replace("/", "-")


def raw_parquet_path(
    exchange: str,
    symbol: str,
    timeframe: str,
    root: Path | str | None = None,
) -> Path:
    """Return the path to the raw OHLCV parquet file."""

    base = Path(root) if root else Path("data/raw")
    return base / exchange / symbol_to_dir(symbol) / f"{timeframe}.parquet"
