from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Deque, Iterator, Optional, Any


class SnapshotCache:
    """Simple LRU cache for top-of-book snapshots."""

    def __init__(self, maxlen: int = 10) -> None:
        self.maxlen = maxlen
        self._data: Deque[Any] = deque(maxlen=maxlen)

    def add(self, snapshot: Any) -> None:
        """Store a new snapshot in the cache."""
        self._data.append(snapshot)

    def latest(self) -> Optional[Any]:
        """Return the most recent snapshot or ``None``."""
        return self._data[-1] if self._data else None

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:  # pragma: no cover - trivial
        return iter(self._data)


def top_of_book_stream(
    ws_source: Optional[Callable[[], Iterator[dict]]] = None,
    rest_fetch: Optional[Callable[[], dict]] = None,
    cache: Optional[SnapshotCache] = None,
    cache_size: int = 10,
    backoff: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[dict]:
    """Yield top-of-book snapshots from WS or REST fallback.

    Parameters
    ----------
    ws_source: callable returning an iterator of snapshots.
        When provided, it is consumed first.  Any exception will trigger a
        fallback to ``rest_fetch``.
    rest_fetch: callable returning a single snapshot.
        Used when the websocket source is unavailable.
    cache: optional ``SnapshotCache`` to store recent snapshots.
    cache_size: size of the cache if ``cache`` is ``None``.
    backoff: base delay (in seconds) between REST polls.  After each failure
        the delay doubles up to ``backoff * 32``.
    sleep: function used to sleep between REST polls (useful for tests).
    """

    if cache is None:
        cache = SnapshotCache(maxlen=cache_size)

    if ws_source is not None:
        try:
            for snap in ws_source():
                cache.add(snap)
                yield snap
        except Exception:  # pragma: no cover - network errors are expected
            logging.warning("websocket stream failed, switching to REST", exc_info=True)

    if rest_fetch is None:
        return

    delay = backoff
    while True:
        try:
            snap = rest_fetch()
            cache.add(snap)
            yield snap
            delay = backoff
        except Exception:
            logging.warning("REST fetch failed", exc_info=True)
            delay = min(delay * 2, backoff * 32)
        sleep(delay)
