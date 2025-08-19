from __future__ import annotations

import numpy as np
import pandas as pd


def pnl(returns: pd.Series) -> float:
    """Total return over the period."""

    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def sharpe(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free / periods_per_year
    mu = excess.mean() * periods_per_year
    sigma = excess.std(ddof=1) * np.sqrt(periods_per_year)
    return float(mu / (sigma + 1e-12))


def sortino(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    downside = returns.clip(upper=0.0)
    dd = downside.std(ddof=1) * np.sqrt(periods_per_year)
    mu = (returns - risk_free / periods_per_year).mean() * periods_per_year
    return float(mu / (dd + 1e-12))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return float(drawdown.min())


def hit_ratio(trades: list) -> float:
    """Fraction of profitable trades."""

    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0.0) > 0)
    return float(wins / len(trades))


def turnover(trades: list) -> float:
    return float(len(trades))

