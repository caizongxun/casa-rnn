"""
backtest.py -- lightweight vectorised backtester

Used by trainer for backtest-in-the-loop reward signal.
"""

from __future__ import annotations
import numpy as np


def run_backtest(
    actions: np.ndarray,    # (N,)  0=LONG 1=FLAT 2=SHORT
    close:   np.ndarray,    # (N,)
    fee:     float = 0.001,
) -> dict:
    """
    Simple bar-by-bar backtest.
    Returns dict with sharpe, max_dd, win_rate, total_return.
    """
    n       = min(len(actions), len(close) - 1)
    rets    = np.diff(np.log(close + 1e-8))[:n]
    signs   = np.where(actions[:n] == 0,  1.0,
              np.where(actions[:n] == 2, -1.0, 0.0))
    costs   = (np.diff(actions[:n+1]) != 0).astype(float) * fee
    pnl     = signs * rets - costs

    cumret  = np.cumsum(pnl)
    total   = cumret[-1] if len(cumret) else 0.0

    roll_max = np.maximum.accumulate(cumret)
    dd       = cumret - roll_max
    max_dd   = dd.min() if len(dd) else 0.0

    sharpe  = 0.0
    if pnl.std() > 1e-8:
        sharpe = pnl.mean() / pnl.std() * np.sqrt(365 * 24)

    wins    = (pnl[signs != 0] > 0).mean() if (signs != 0).any() else 0.0

    return {
        "sharpe":       float(sharpe),
        "max_dd":       float(max_dd),
        "win_rate":     float(wins),
        "total_return": float(total),
    }
