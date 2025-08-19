import os
import time

try:
    import psutil
except Exception as e:  # pragma: no cover - psutil missing
    psutil = None
    _psutil_import_err = e
else:
    _psutil_import_err = None


def memory_guard(max_gb: float = 10) -> bool:
    """Return True if current process RSS memory is below ``max_gb``.

    Parameters
    ----------
    max_gb:
        Maximum allowed resident memory in gigabytes.
    """
    if psutil is None:  # pragma: no cover - psutil missing
        raise RuntimeError("psutil is required for memory_guard") from _psutil_import_err
    rss_bytes = psutil.Process(os.getpid()).memory_info().rss
    return rss_bytes / (1024 ** 3) <= max_gb


def time_guard(last_heartbeat: float, timeout_min: float = 15) -> bool:
    """Return True if the elapsed time since ``last_heartbeat`` is within ``timeout_min``.

    Parameters
    ----------
    last_heartbeat:
        A timestamp (as returned by :func:`time.time`) marking the last progress moment.
    timeout_min:
        Minutes allowed to elapse before considering the process stalled.
    """
    return (time.time() - last_heartbeat) <= timeout_min * 60
