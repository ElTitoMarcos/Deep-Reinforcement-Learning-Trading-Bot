from __future__ import annotations
import logging
import time
from collections import deque
from datetime import datetime, UTC
from typing import Deque, Dict, Iterator, Optional, Set

from src.reports.human_friendly import episode_sentence

_LOG_BUFFER: Deque[Dict[str, object]] = deque(maxlen=1000)

_EVENT_MAP = {
    "trade_executed": (
        lambda r: f"Ejecutada orden {getattr(r, 'side', '?')} {getattr(r, 'qty', '?')} en {getattr(r, 'symbol', '?')} a {getattr(r, 'price', '?')}",
        "trades",
    ),
    "risk_blocked_min_notional": (
        lambda r: "Orden cancelada: mínima notional no alcanzada",
        "riesgo",
    ),
    "slippage_est": (
        lambda r: f"Slippage estimado {getattr(r, 'pct', 0.0):.2%} por tamaño {getattr(r, 'usd', '?')}",
        "datos",
    ),
    "checkpoint_saved": (
        lambda r: f"Checkpoint guardado (pasos={getattr(r, 'steps', '?')})",
        "checkpoints",
    ),
    "episode_metrics": (
        lambda r: episode_sentence(getattr(r, "metrics", {})),
        "metricas",
    ),
}

class _StreamHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - passthrough
        event = getattr(record, "event", None)
        formatter = _EVENT_MAP.get(event)
        if formatter:
            msg = formatter[0](record)
            kind = formatter[1]
        else:
            msg = record.getMessage()
            kind = getattr(record, "kind", event) or "general"
        _LOG_BUFFER.append(
            {
                "time": datetime.fromtimestamp(record.created, UTC),
                "level": record.levelname.lower(),
                "levelno": record.levelno,
                "kind": kind,
                "message": msg,
            }
        )

_handler_installed = False

def _ensure_handler() -> None:
    global _handler_installed
    if not _handler_installed:
        logging.getLogger().addHandler(_StreamHandler())
        _handler_installed = True

_ensure_handler()

def subscribe(level: str = "info", kinds: Optional[Set[str]] = None) -> Iterator[Dict[str, object]]:
    """Yield log entries filtered by level and kinds."""
    levelno = getattr(logging, level.upper(), logging.INFO)
    kinds = kinds or set()
    idx = 0
    while True:
        while idx < len(_LOG_BUFFER):
            item = _LOG_BUFFER[idx]
            idx += 1
            if item["levelno"] >= levelno and (not kinds or item["kind"] in kinds):
                yield item
        time.sleep(0.1)
