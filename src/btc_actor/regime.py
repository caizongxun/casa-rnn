"""
regime.py -- HMM-based market regime labeller

Requires: hmmlearn>=0.3.0
Fallback: if hmmlearn not available, uses a simple rule-based regime
          (trend = abs(20h momentum) > 0.01, volatile = ATR/close > 0.02)
"""

from __future__ import annotations
import numpy as np


def fit_regime_labels(
    close: np.ndarray,
    atr:   np.ndarray,
    n_states: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """
    Returns array of shape (N,) with values in {0=TREND, 1=RANGE, 2=VOLATILE}.
    Tries hmmlearn first; falls back to rule-based.
    """
    features = _build_hmm_features(close, atr)

    try:
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            random_state=random_state,
            verbose=False,
        )
        model.fit(features)
        raw = model.predict(features)
        return _remap_states(raw, features, n_states)
    except Exception:
        return _rule_based(close, atr)


def _build_hmm_features(close, atr):
    ret   = np.zeros(len(close))
    ret[1:] = np.diff(np.log(close + 1e-8))
    vol   = np.abs(ret)
    norm_atr = atr / (close + 1e-8)
    feat  = np.column_stack([ret, vol, norm_atr])
    # z-score
    feat  = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    return feat


def _remap_states(labels, features, n_states):
    """
    Map HMM states to semantic names based on volatility level.
    State with highest mean |return| -> VOLATILE (2)
    State with lowest  mean |return| -> RANGE    (1)
    Middle                           -> TREND     (0)
    """
    vol_by_state = [
        np.abs(features[labels == s, 0]).mean() if (labels == s).any() else 0.0
        for s in range(n_states)
    ]
    order = np.argsort(vol_by_state)  # low -> high volatility
    remap = np.zeros(n_states, dtype=int)
    remap[order[0]] = 1   # RANGE
    remap[order[-1]] = 2  # VOLATILE
    remap[order[1 if n_states == 3 else 0]] = 0  # TREND
    return remap[labels]


def _rule_based(close, atr):
    n      = len(close)
    labels = np.ones(n, dtype=int)  # default RANGE
    mom20  = np.zeros(n)
    for i in range(20, n):
        mom20[i] = (close[i] - close[i - 20]) / (close[i - 20] + 1e-8)
    norm_atr = atr / (close + 1e-8)
    labels[np.abs(mom20) > 0.015]  = 0   # TREND
    labels[norm_atr      > 0.025]  = 2   # VOLATILE
    return labels
