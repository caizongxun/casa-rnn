"""Financial feature engineering utilities."""
import numpy as np
import torch
from typing import Optional


def make_financial_features(
    prices: np.ndarray,
    window: int = 20,
    extra_features: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Compute standard financial features from a price series.

    Features produced (per timestep):
        0: log return
        1: rolling volatility  (std of log returns over window)
        2: momentum            (sum of log returns over last 5 steps)
        3: z-score             (standardized log return)
        4: vol-of-vol          (std of rolling vol over window)
        5: vol ratio           (current vol / mean vol -- regime indicator)

    Args:
        prices:         1-D array of closing prices, shape (T,)
        window:         rolling window size (default 20)
        extra_features: optional extra feature matrix shape (T-1, F)
                        appended as additional columns
    Returns:
        torch.FloatTensor of shape (T-1, n_features)
    """
    prices = np.asarray(prices, dtype=np.float64)
    r      = np.diff(np.log(prices + 1e-8))  # (T-1,)
    T      = len(r)

    vol = np.array([
        r[max(0, i - window):i].std() if i > 1 else 0.0
        for i in range(1, T + 1)
    ])
    mom = np.array([
        r[max(0, i - 5):i].sum()
        for i in range(T)
    ])
    eps   = r.std() + 1e-8
    z     = (r - r.mean()) / eps
    vov   = np.array([
        vol[max(0, i - window):i].std() if i > 1 else 0.0
        for i in range(T)
    ])
    mean_vol  = vol.mean() + 1e-8
    vol_ratio = vol / mean_vol

    feats = np.stack([r, vol, mom, z, vov, vol_ratio], axis=-1)  # (T, 6)

    if extra_features is not None:
        extra_features = np.asarray(extra_features)
        assert extra_features.shape[0] == T, "extra_features must have same length as T-1"
        feats = np.concatenate([feats, extra_features], axis=-1)

    return torch.FloatTensor(feats)


def make_sequences(
    features: torch.Tensor,
    targets:  torch.Tensor,
    seq_len:  int = 60,
    stride:   int = 1,
):
    """
    Slide a window over features and targets to create (X, y) pairs.

    Args:
        features: (T, F)  feature tensor
        targets:  (T, O)  target tensor
        seq_len:  lookback window length
        stride:   step between windows
    Returns:
        X: (N, seq_len, F)
        y: (N, seq_len, O)
    """
    X_list, y_list = [], []
    T = features.shape[0]
    for start in range(0, T - seq_len + 1, stride):
        X_list.append(features[start: start + seq_len])
        y_list.append(targets [start: start + seq_len])
    return torch.stack(X_list), torch.stack(y_list)
