import os
from pathlib import Path

RAW_DIR = Path(os.getenv("DRLTB_RAW_DIR", "data/raw"))
REPORTS_DIR = Path(os.getenv("DRLTB_REPORTS_DIR", "reports"))
CHECKPOINTS_DIR = Path(os.getenv("DRLTB_CHECKPOINTS_DIR", "checkpoints"))

def symbol_to_dir(sym: str) -> str:
    """Convert a market symbol to a filesystem-friendly name.

    Examples
    --------
    >>> symbol_to_dir("ETH/USDT")
    'ETH-USDT'
    """
    return sym.replace("/", "-")

def dir_to_symbol(d: str) -> str:
    """Convert on-disk directory name back to market symbol."""
    return d.replace("-", "/")

def raw_parquet_path(exchange: str, symbol: str, timeframe: str) -> Path:
    """Path for raw OHLCV parquet file."""
    return RAW_DIR / exchange / symbol_to_dir(symbol) / f"{timeframe}.parquet"

def reports_dir() -> Path:
    return REPORTS_DIR

def checkpoints_dir() -> Path:
    return CHECKPOINTS_DIR

def ensure_dirs_exist() -> None:
    for p in [RAW_DIR, REPORTS_DIR, CHECKPOINTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

def posix(p: Path) -> str:
    return p.as_posix()
