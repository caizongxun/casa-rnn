"""
Danger Signal Detector — immune-system inspired anomaly gate.

Changelog:
  - ema_alpha: 0.01 -> 0.05
  - selective_reset(): Direction B
  - Fix: variance init back to 1.0; forward adds / hidden_size normalization
    so Danger score lands in ~1-2 range instead of 4-6
"""
import torch
import torch.nn as nn


class DangerSignalDetector(nn.Module):
    def __init__(self, hidden_size: int, ema_alpha: float = 0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha       = ema_alpha
        self.log_thresh  = nn.Parameter(torch.tensor(1.0))
        self.reset_proj  = nn.Linear(1, 1, bias=True)
        self.register_buffer("mu",  torch.zeros(hidden_size))
        self.register_buffer("var", torch.ones(hidden_size))   # back to 1.0
        self.register_buffer("n",   torch.tensor(0.0))

    @property
    def threshold(self) -> torch.Tensor:
        return (self.log_thresh.exp() + 0.5).clamp(0.5, 10.0)

    @torch.no_grad()
    def update(self, h: torch.Tensor) -> None:
        h_mean   = h.mean(dim=0).detach()
        self.mu  = (1 - self.alpha) * self.mu  + self.alpha * h_mean
        self.var = (1 - self.alpha) * self.var + self.alpha * (h_mean - self.mu).pow(2)
        self.n   = self.n + 1

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Returns danger_score: (B, 1).
        Divided by hidden_size so score ~ O(1) instead of O(sqrt(hidden_size)).
        """
        if self.n < 10:
            return torch.zeros(h.shape[0], 1, device=h.device)
        var_safe = self.var.clamp(min=1e-6)
        diff     = h - self.mu.unsqueeze(0)
        # sum (not mean) then divide by hidden_size -> stable ~1 for normal input
        mahal    = (diff.pow(2) / var_safe.unsqueeze(0)).sum(dim=-1, keepdim=True)
        return (mahal / self.hidden_size).sqrt()

    def ne_boost(self, danger_score: torch.Tensor, ne_raw: torch.Tensor) -> torch.Tensor:
        boost = (danger_score / (self.threshold + 1e-6)).clamp(0, 1)
        return (ne_raw + 0.3 * boost).clamp(0, 1)

    def selective_reset(self, h_slow: torch.Tensor, danger_score: torch.Tensor) -> torch.Tensor:
        reset_gate = torch.sigmoid(self.reset_proj(danger_score))
        return (1.0 - reset_gate) * h_slow
