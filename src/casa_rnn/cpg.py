"""
Central Pattern Generator (CPG) — market rhythm encoder.

Inspired by spinal cord CPGs that produce rhythmic motor output
without higher-brain involvement. In finance, encodes known market
periodic structures (intraday U-shape, weekly, monthly) as explicit
oscillator features instead of letting the RNN discover them.

Each oscillator has a learnable period T (clamped to sensible range).
Output: concat of [cos(phi), sin(phi)] for each oscillator -> injected
as extra input channels to the RNN.
"""
import torch
import torch.nn as nn
import math
from typing import List


class CPGOscillator(nn.Module):
    """Single oscillator with learnable period and phase offset."""
    def __init__(self, init_period: float, period_min: float, period_max: float):
        super().__init__()
        # Store log-period so unconstrained optimisation stays positive
        self.log_T    = nn.Parameter(torch.tensor(math.log(init_period)))
        self.phi0     = nn.Parameter(torch.zeros(1))
        self.T_min    = period_min
        self.T_max    = period_max

    @property
    def period(self) -> torch.Tensor:
        return self.log_T.exp().clamp(self.T_min, self.T_max)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, T_seq) integer step index. Returns (B, T_seq, 2)."""
        phi = 2 * math.pi * t.float() / self.period + self.phi0
        return torch.stack([phi.cos(), phi.sin()], dim=-1)


class CPGEncoder(nn.Module):
    """
    Bank of oscillators with default periods tuned for minute-bar data:
      - 390 bars = 1 trading day
      - 1950 bars = 1 trading week
      - 8580 bars = 1 trading month
    Pass custom init_periods to override.
    """
    def __init__(
        self,
        init_periods: List[float] = (390.0, 1950.0, 8580.0),
        period_min: float = 10.0,
        period_max: float = 50000.0,
    ):
        super().__init__()
        self.oscillators = nn.ModuleList([
            CPGOscillator(p, period_min, period_max) for p in init_periods
        ])
        self.out_dim = len(init_periods) * 2  # cos+sin per oscillator

    def forward(self, seq_len: int, batch: int, device: torch.device, t_offset: int = 0) -> torch.Tensor:
        """Returns (B, T_seq, out_dim) CPG features."""
        t = torch.arange(t_offset, t_offset + seq_len, device=device).unsqueeze(0).expand(batch, -1)
        parts = [osc(t) for osc in self.oscillators]   # each (B, T, 2)
        return torch.cat(parts, dim=-1)                 # (B, T, out_dim)

    def period_loss(self) -> torch.Tensor:
        """Soft regulariser: keep oscillators spread (no two collapse to same period)."""
        periods = torch.stack([o.period for o in self.oscillators])
        n = len(periods)
        loss = torch.zeros(1, device=periods.device)
        for i in range(n):
            for j in range(i + 1, n):
                loss = loss + torch.exp(-((periods[i] - periods[j]).abs() / 100.0))
        return loss
