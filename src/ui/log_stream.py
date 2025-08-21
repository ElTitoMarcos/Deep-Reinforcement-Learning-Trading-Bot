from __future__ import annotations
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, UTC, timedelta
from typing import Deque, Dict, Iterator, Optional, Set, Tuple

from src.reports.human_friendly import episode_sentence

MAX_PER_KIND = 200
_LOG_BUFFER: Deque[Dict[str, object]] = deque()
_KIND_BUFFERS: Dict[str, Deque[Dict[str, object]]] = defaultdict(deque)
_COALESCE: Dict[str, Dict[str, object]] = {}

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

_PROFILES: Dict[str, Set[str]] = {
    "training": {"reward_tuner", "dqn_stability", "checkpoints"},
    "evaluation": {"hybrid_weights", "performance"},
    "data": {"incremental_update", "qc"},
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

        if event == "slippage_est":
            state = _COALESCE.setdefault(
                "slippage_est",
                {
                    "count": 0,
                    "min_pct": None,
                    "max_pct": None,
                    "level": record.levelname.lower(),
                    "levelno": record.levelno,
                },
            )
            pct = getattr(record, "pct", 0.0)
            state["count"] += 1
            state["min_pct"] = pct if state["min_pct"] is None else min(state["min_pct"], pct)
            state["max_pct"] = pct if state["max_pct"] is None else max(state["max_pct"], pct)
            if state["count"] >= 20:
                msg = (
                    f"slippage_est x{state['count']}, rango "
                    f"[{state['min_pct']:.2%}–{state['max_pct']:.2%}]"
                )
                _append_event(
                    {
                        "time": datetime.fromtimestamp(record.created, UTC),
                        "level": state["level"],
                        "levelno": state["levelno"],
                        "kind": "datos",
                        "message": msg,
                    }
                )
                _COALESCE.pop("slippage_est", None)
            return

        _append_event(
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
        if idx > len(_LOG_BUFFER):
            idx = len(_LOG_BUFFER)
        while idx < len(_LOG_BUFFER):
            item = _LOG_BUFFER[idx]
            idx += 1
            if item["levelno"] >= levelno and (not kinds or item["kind"] in kinds):
                yield item
        time.sleep(0.1)


def get_auto_profile(stage: str) -> Set[str]:
    """Return default event kinds for a given app stage."""
    stage = (stage or "").lower()
    if stage in {"warmup", "exploration", "consolidation", "fine-tune", "training"}:
        key = "training"
    elif stage.startswith("eval"):
        key = "evaluation"
    else:
        key = "data"
    return _PROFILES[key]


def _append_event(event: Dict[str, object]) -> None:
    kind = event["kind"]
    buf = _KIND_BUFFERS[kind]
    buf.append(event)
    if len(buf) > MAX_PER_KIND:
        old = buf.popleft()
        try:
            _LOG_BUFFER.remove(old)
        except ValueError:
            pass
    _LOG_BUFFER.append(event)


def recent_counts(window_secs: int = 30) -> Tuple[int, Dict[str, int]]:
    cutoff = datetime.now(UTC) - timedelta(seconds=window_secs)
    total = 0
    counts: Dict[str, int] = defaultdict(int)
    for item in reversed(_LOG_BUFFER):
        if item["time"] < cutoff:
            break
        total += 1
        counts[item["kind"]] += 1
    return total, dict(counts)
