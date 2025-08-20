"""Utilities to run background tasks without blocking Streamlit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict
import uuid

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

# Global thread pool for UI background jobs
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

_JOBS: Dict[str, Future] = {}
_PROGRESS: Dict[str, str] = {}


def run_bg(
    name: str,
    func: Callable[..., Any],
    *args: Any,
    job_id: str | None = None,
    **kwargs: Any,
) -> str:
    """Run ``func`` in a background thread and return a job id.

    Parameters
    ----------
    name: str
        Descriptive name for the job; used to build the identifier when none is
        provided.
    func: Callable
        The function to execute in the background.
    job_id: str | None
        Optional explicit id; otherwise a unique one is generated.

    Any additional ``args`` and ``kwargs`` are passed to ``func``.  ``kwargs`` may
    include a ``progress_cb`` parameter, which will be invoked as usual by the
    task.  To report progress to the polling side, the caller should wrap this
    callback and call :func:`set_progress` with the job id.
    """

    jid = job_id or f"{name}-{uuid.uuid4().hex[:8]}"
    parent_ctx = get_script_run_ctx()

    def wrapped() -> Any:
        # Propagate Streamlit's run context into the worker thread
        add_script_run_ctx(ctx=parent_ctx)
        return func(*args, **kwargs)

    fut = _EXECUTOR.submit(wrapped)
    _JOBS[jid] = fut
    return jid


def set_progress(job_id: str, msg: str) -> None:
    """Record a progress message for a background job."""

    _PROGRESS[job_id] = msg


def poll(job_id: str) -> Dict[str, Any]:
    """Return status information for a job started with :func:`run_bg`."""

    fut = _JOBS.get(job_id)
    if fut is None:
        return {"state": "error", "error": "unknown job"}
    prog = _PROGRESS.get(job_id)

    if not fut.done():
        return {"state": "running", "progress": prog}

    try:
        res = fut.result()
        return {"state": "done", "result": res, "progress": prog}
    except Exception as err:  # pragma: no cover - surfaced to UI
        return {"state": "error", "error": str(err), "progress": prog}

