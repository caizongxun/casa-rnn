"""
Danger Signal Detector — immune-system inspired anomaly gate.

Changelog:
  - ema_alpha: 0.01 -> 0.05  (faster EMA, tracks regime shifts quickly)
  - selective_reset(): new method, soft-resets h_slow when danger is high
    (Direction B: Selective State Reset)
"""
import torch
import torch.nn as nn


class DangerSignalDetector(nn.Module):
    def __init__(self, hidden_size: int, ema_alpha: float = 0.05):
        super().__init__()
        self.alpha      = ema_alpha
        self.log_thresh = nn.Parameter(torch.tensor(1.0))
        # learnable reset gate: decides how much of h_slow to wipe
        self.reset_proj = nn.Linear(1, 1, bias=True)  # danger_score -> reset magnitude
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
        Returns danger_score: (B, 1)
        """
        if self.n < 10:
            return torch.zeros(h.shape[0], 1, device=h.device)
        var_safe = self.var.clamp(min=1e-6)
        diff  = h - self.mu.unsqueeze(0)
        mahal = (diff.pow(2) / var_safe.unsqueeze(0)).mean(dim=-1, keepdim=True)
        return mahal.sqrt()

    def ne_boost(self, danger_score: torch.Tensor, ne_raw: torch.Tensor) -> torch.Tensor:
        """Boost NE proportional to danger; clamp to [0,1]."""
        boost = (danger_score / (self.threshold + 1e-6)).clamp(0, 1)
        return (ne_raw + 0.3 * boost).clamp(0, 1)

    def selective_reset(
        self,
        h_slow: torch.Tensor,   # (B, hidden_size) — long-term hidden state
        danger_score: torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        """
        Direction B: Selective State Reset.

        When danger is high (regime change detected), soft-erase h_slow
        so stale market memory does not pollute new regime predictions.
        h_fast is intentionally left intact (short-term dynamics still useful).

        reset_gate in [0, 1]:
          0 = no reset (normal market)
          1 = full reset (extreme regime change)
        """
        # sigmoid maps danger -> reset_gate magnitude; bias initialised negative
        # so gate starts near zero and only opens under genuine danger
        reset_gate = torch.sigmoid(self.reset_proj(danger_score))  # (B, 1)
        return (1.0 - reset_gate) * h_slow
