"""Uncertainty-Gated Output Head.

Changelog:
  v0.6.1 - clamp_log_var tightened 4.0 -> 2.0 so std is bounded to
           exp(1) ~ 2.7 max; prevents q_hat blowup in conformal calibration
"""
import torch
import torch.nn as nn


class UncertaintyGatedHead(nn.Module):
    """
    Output head that produces mean + std instead of a point estimate.

    High uncertainty -> model acknowledges it doesn't know.
    Loss: Negative Log Likelihood (NLL) instead of MSE.
    This stops the model from chasing noise in financial data.
    """

    def __init__(self, hidden_size: int, output_size: int, clamp_log_var: float = 2.0):
        super().__init__()
        self.mean_proj = nn.Linear(hidden_size, output_size)
        self.var_proj  = nn.Linear(hidden_size, output_size)
        self.clamp     = clamp_log_var

    def forward(self, h: torch.Tensor):
        mean    = self.mean_proj(h)
        log_var = self.var_proj(h).clamp(-self.clamp, self.clamp)
        std     = torch.exp(0.5 * log_var)
        return mean, std

    @staticmethod
    def nll_loss(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Normal(mean, std + 1e-6)
        return -dist.log_prob(target).mean()
