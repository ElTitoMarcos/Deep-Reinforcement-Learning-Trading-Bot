from __future__ import annotations
from typing import Dict, Any
import numpy as np

class DeterministicPolicy:
    """Threshold-based entry with simple trailing logic. Placeholder."""
    def __init__(self, threshold: float = 0.001):
        self.threshold = float(threshold)

    def act(self, obs) -> int:
        # obs = [price_norm, vol_norm, volat, dd, wall, pos, trailing_norm]
        price_norm = float(obs[0])
        pos = int(obs[5])
        # Enter if price rising above small threshold
        if pos == 0 and price_norm > (1.0 + self.threshold):
            return 1  # open/long
        # Close if price falls
        if pos == 1 and price_norm < (1.0 - self.threshold):
            return 2  # close
        return 0  # hold
