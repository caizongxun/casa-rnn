"""
Danger Signal Detector — immune-system inspired anomaly gate.

The innate immune system flags 'danger' based on structural abnormality
(Damage-Associated Molecular Patterns), not learned templates. Analogously,
we compute the Mahalanobis distance of the current hidden state from a
rolling empirical distribution (mean + diagonal covariance via EMA).

High danger_score -> model activates circuit-breaker behaviour:
  1. Injects danger_score into RNN as an extra feature
  2. Scales NE upward (more norepinephrine = higher alertness)
  3. Signals adaptive stride to slow down (shorter windows, finer resolution)

All thresholds are LEARNABLE so the model decides its own sensitivity.

Changelog:
  - ema_alpha: 0.01 -> 0.05  (faster EMA update, tracks regime shifts more closely)
"""
import torch
import torch.nn as nn


class DangerSignalDetector(nn.Module):
    def __init__(self, hidden_size: int, ema_alpha: float = 0.05):
        super().__init__()
        self.alpha     = ema_alpha
        # learnable threshold (sigmoid -> 0-1, scaled to reasonable Mahal range)
        self.log_thresh = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("mu",  torch.zeros(hidden_size))
        self.register_buffer("var", torch.ones(hidden_size))
        self.register_buffer("n",   torch.tensor(0.0))

    @property
    def threshold(self) -> torch.Tensor:
        return (self.log_thresh.exp() + 0.5).clamp(0.5, 10.0)

    @torch.no_grad()
    def update(self, h: torch.Tensor) -> None:
        """Online EMA update of mean and diagonal variance."""
        h_mean = h.mean(dim=0).detach()
        self.mu  = (1 - self.alpha) * self.mu  + self.alpha * h_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * (h_mean - self.mu).pow(2)
        self.n   = self.n + 1

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, hidden_size)
        Returns danger_score: (B, 1)  in [0, inf)
        and is_danger: (B, 1) bool mask (score > learnable threshold)
        """
        if self.n < 10:  # not enough history yet
            return torch.zeros(h.shape[0], 1, device=h.device)
        var_safe = self.var.clamp(min=1e-6)
        diff = h - self.mu.unsqueeze(0)
        mahal = (diff.pow(2) / var_safe.unsqueeze(0)).mean(dim=-1, keepdim=True)
        return mahal.sqrt()

    def ne_boost(self, danger_score: torch.Tensor, ne_raw: torch.Tensor) -> torch.Tensor:
        """Boost NE proportional to danger; clamp to [0,1]."""
        boost = (danger_score / (self.threshold + 1e-6)).clamp(0, 1)
        return (ne_raw + 0.3 * boost).clamp(0, 1)
