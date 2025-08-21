from __future__ import annotations

from typing import Callable, List
import logging

from .ccxt_loader import get_exchange, fetch_ohlcv
from .symbol_discovery import discover_symbols
from .enrichment import fetch_symbol_metadata
from .quality import validate_metadata, validate_ohlcv
from .ensure import ensure_ohlcv
from .incremental import update_all
from .refresh_worker import start_refresh_worker
from .microv5_loader import MicroV5Collector


logger = logging.getLogger(__name__)


def prepare_data(
    auto_refresh: bool = True,
    refresh_every_min: int = 5,
    *,
    progress_cb: Callable[[str], None] | None = None,
    timeframe: str = "1m",
    top_n: int = 20,
) -> List[str]:
    """Run the full data preparation pipeline and return discovered symbols."""

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)
        logger.info(msg)

    ex = get_exchange()
    report("Descubriendo…")
    symbols = discover_symbols(ex, top_n=top_n)

    report("Validando…")
    meta = fetch_symbol_metadata(symbols)
    for sym in symbols:
        validate_metadata(meta.get(sym, {}))
        df = fetch_ohlcv(ex, sym, timeframe)
        validate_ohlcv(df)

    report("Descargando…")
    for sym in symbols:
        ensure_ohlcv(ex.id if hasattr(ex, "id") else "binance", sym, timeframe)

    report("Actualizando…")
    update_all(symbols, timeframe)

    # Start microstructure collectors (light weight seed)
    collectors: List[MicroV5Collector] = []
    for sym in symbols:
        try:
            col = MicroV5Collector(ex, sym, win_secs=120)
            # Grab a single snapshot so the UI can report something immediately
            col.step()
            collectors.append(col)
        except Exception:
            # Non critical; network errors are ignored at this stage
            continue

    if auto_refresh:
        start_refresh_worker(symbols, timeframe, every=refresh_every_min)
        report("Refresco en marcha ✔")
    report("Microestructura V5 activa")
    return symbols
