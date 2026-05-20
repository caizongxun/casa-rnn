"""Uncertainty-Gated Output Head."""
import torch
import torch.nn as nn


class UncertaintyGatedHead(nn.Module):
    """
    Output head that produces mean + std instead of a point estimate.

    High uncertainty -> model acknowledges it doesn't know.
    Loss: Negative Log Likelihood (NLL) instead of MSE.
    This stops the model from chasing noise in financial data.

    Args:
        hidden_size: input hidden dimension
        output_size: output dimension
        clamp_log_var: range to clamp log variance (prevents explosion)
    """

    def __init__(self, hidden_size: int, output_size: int, clamp_log_var: float = 4.0):
        super().__init__()
        self.mean_proj = nn.Linear(hidden_size, output_size)
        self.var_proj  = nn.Linear(hidden_size, output_size)
        self.clamp     = clamp_log_var

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: (batch, hidden_size)
        Returns:
            mean: (batch, output_size)
            std:  (batch, output_size)  -- always positive
        """
        mean    = self.mean_proj(h)
        log_var = self.var_proj(h).clamp(-self.clamp, self.clamp)
        std     = torch.exp(0.5 * log_var)
        return mean, std

    @staticmethod
    def nll_loss(mean: torch.Tensor, std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Gaussian NLL loss.
        Forces model to widen std when uncertain instead of making wrong point predictions.
        """
        dist = torch.distributions.Normal(mean, std + 1e-6)
        return -dist.log_prob(target).mean()
