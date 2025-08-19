"""Thin wrapper around :mod:`src.training.train_drl`.

This allows invoking the trainer via ``python scripts/train.py`` while the
actual implementation resides in the package so it can also be executed with
``python -m src.training.train_drl``.
"""

from __future__ import annotations

from src.training.train_drl import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

