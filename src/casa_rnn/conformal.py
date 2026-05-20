"""
Conformal Prediction wrapper — distribution-free coverage guarantee.

Wraps any (mean, std) output and calibrates prediction intervals
to guarantee exactly `coverage` empirical coverage on the calibration set.

Usage:
    cp = ConformalWrapper(coverage=0.90)
    cp.calibrate(means_cal, stds_cal, y_cal)   # one-time calibration
    lower, upper = cp.predict_interval(means, stds)

No gradients flow through the calibration step; this is a post-hoc
correction layer that doesn't affect training.

Also provides a `sharpness_loss` that can be included in training loss
to push the model to produce tighter (sharper) intervals while still
meeting coverage — this creates a training-time incentive.
"""
import torch
import torch.nn as nn


class ConformalWrapper(nn.Module):
    def __init__(self, coverage: float = 0.90):
        super().__init__()
        self.coverage = coverage
        self.register_buffer("q_hat", torch.tensor(1.0))
        self.calibrated = False

    @torch.no_grad()
    def calibrate(
        self,
        means: torch.Tensor,   # (N,) or (N,1)
        stds:  torch.Tensor,   # (N,) or (N,1)
        y:     torch.Tensor,   # (N,) or (N,1)
    ) -> float:
        """
        Compute the (1-alpha) empirical quantile of nonconformity scores.
        nonconformity = |y - mean| / std
        Returns q_hat.
        """
        scores = ((y.flatten() - means.flatten()).abs() / stds.flatten().clamp(min=1e-6))
        n = len(scores)
        level = min(1.0, (math.ceil((n + 1) * self.coverage)) / n)
        self.q_hat = torch.quantile(scores, level)
        self.calibrated = True
        return self.q_hat.item()

    def predict_interval(
        self,
        means: torch.Tensor,
        stds:  torch.Tensor,
    ):
        """Returns (lower, upper) with guaranteed coverage."""
        half = self.q_hat * stds
        return means - half, means + half

    def sharpness_loss(self, stds: torch.Tensor) -> torch.Tensor:
        """
        Training-time incentive: minimise interval width.
        This is added to the task loss with a small weight so the model
        is rewarded for confident (sharp) predictions.
        """
        return stds.mean()


import math  # noqa: E402  (placed after class to keep class docstring clean)
