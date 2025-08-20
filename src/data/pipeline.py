from __future__ import annotations

from typing import Callable, List

from .ccxt_loader import get_exchange, fetch_ohlcv
from .symbol_discovery import discover_symbols
from .enrichment import fetch_symbol_metadata
from .quality import validate_metadata, validate_ohlcv
from .ensure import ensure_ohlcv
from .incremental import update_all
from .refresh_worker import start_refresh_worker


def prepare_data(
    auto_refresh: bool = True,
    refresh_every_min: int = 5,
    *,
    progress_cb: Callable[[str], None] | None = None,
    timeframe: str = "1m",
    top_n: int = 20,
) -> List[str]:
    """Run the full data preparation pipeline and return discovered symbols."""

    def report(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

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

    if auto_refresh:
        start_refresh_worker(symbols, timeframe, every=refresh_every_min)
        report("Refresco en marcha ✔")
    return symbols
