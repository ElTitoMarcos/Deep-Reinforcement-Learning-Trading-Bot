"""CCXT data helpers.

Currently only provides a tiny utility used in the tests to generate
synthetic high frequency data from minute bars.  The function is kept
lightweight to avoid heavy dependencies during testing.
"""

from __future__ import annotations

import pandas as pd


def simulate_1s_from_1m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Expand 1 minute OHLCV data into 1 second bars.

    The implementation performs a simple linear interpolation between the
    open and close of each minute bar and divides the volume evenly.  It is
    intended purely for testing and does not attempt to model real market
    micro-structure.
    """

    rows = []
    for _, row in df_1m.iterrows():
        start_ts = int(row["ts"])
        open_ = float(row["open"])
        close = float(row["close"])
        volume = float(row.get("volume", 0.0)) / 60.0
        for i in range(60):
            ts = start_ts + i * 1000
            # simple linear interpolation for the price
            if 59 > 0:
                frac = i / 59.0
            else:
                frac = 0.0
            price = open_ + (close - open_) * frac
            rows.append(
                {
                    "ts": ts,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": volume,
                }
            )

    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])

