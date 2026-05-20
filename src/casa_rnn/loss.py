"""Counterfactual Loss for CASA-RNN training."""
import torch
import torch.nn as nn


class CounterfactualLoss(nn.Module):
    """
    Combined loss:
      L = L_factual + alpha * L_pred_coding + beta * L_regime_entropy

    - L_factual: standard prediction loss (MSE or BCE)
    - L_pred_coding: predictive coding error from each CASA cell
    - L_regime_entropy: encourages regime detector to be decisive (low entropy)

    Args:
        alpha: weight for predictive coding loss (default 0.1)
        beta: weight for regime entropy regularization (default 0.01)
        task: 'regression' (MSE) or 'classification' (BCE)
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.01, task: str = "regression"):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.task = task
        self.base_loss = nn.MSELoss() if task == "regression" else nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,      # (batch, seq_len, output_size)
        target: torch.Tensor,    # (batch, seq_len, output_size)
        extra: dict,
    ) -> torch.Tensor:
        # Factual loss
        l_factual = self.base_loss(pred, target)

        # Predictive coding loss from model internals
        l_pred = extra.get("pred_coding_loss", torch.tensor(0.0, device=pred.device))

        # Regime entropy: encourage decisive regime detection (binary entropy minimization)
        regime_probs = extra.get("regime_probs", None)
        if regime_probs is not None:
            p = regime_probs.clamp(1e-6, 1 - 1e-6)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log())
            l_regime = entropy.mean()
        else:
            l_regime = torch.tensor(0.0, device=pred.device)

        total = l_factual + self.alpha * l_pred + self.beta * l_regime
        return total
