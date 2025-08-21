"""Thin wrapper around :mod:`src.training.train_drl`.

This allows invoking the trainer via ``python scripts/train.py`` while the
actual implementation resides in the package so it can also be executed with
``python -m src.training.train_drl``.
"""

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
import os
_DOTENV = find_dotenv(usecwd=True)
load_dotenv(_DOTENV, override=True)
if __name__ == "__main__" or os.getenv("DEBUG_DOTENV") == "1":
    print(f"[.env] Cargado: {_DOTENV or 'NO ENCONTRADO'}")

from src.training.train_drl import main


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

