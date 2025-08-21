import json
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict

from .paths import reports_dir

_LOG_PATH = reports_dir() / "experiments.jsonl"
_RUN_CACHE: Dict[str, Dict[str, Any]] = {}

def _append(obj: Dict[str, Any]) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")

def log_run_start(config_snapshot: Dict[str, Any]) -> str:
    """Log the beginning of a training/evaluation run.

    Parameters
    ----------
    config_snapshot: dict
        Effective configuration for the run.
    Returns
    -------
    str
        Generated run identifier.
    """
    run_id = uuid.uuid4().hex
    ts = datetime.now(UTC).isoformat()
    _RUN_CACHE[run_id] = config_snapshot
    rec: Dict[str, Any] = {
        "event": "start",
        "run_id": run_id,
        "timestamp": ts,
    }
    rec.update(config_snapshot)
    _append(rec)
    return run_id

def log_run_update(run_id: str, partial_metrics: Dict[str, Any]) -> None:
    """Log intermediate metrics for a run."""
    ts = datetime.now(UTC).isoformat()
    rec = {
        "event": "update",
        "run_id": run_id,
        "timestamp": ts,
        "metrics": partial_metrics,
    }
    _append(rec)

def log_run_end(run_id: str, final_metrics: Dict[str, Any]) -> None:
    """Log the end of a run, including the initial config."""
    ts = datetime.now(UTC).isoformat()
    cfg = _RUN_CACHE.pop(run_id, {})
    rec: Dict[str, Any] = {
        "event": "end",
        "run_id": run_id,
        "timestamp": ts,
    }
    rec.update(cfg)
    rec["metrics_final"] = final_metrics
    _append(rec)
