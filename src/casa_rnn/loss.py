"""Combined loss functions for CASA-RNN training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .heads import UncertaintyGatedHead


class ContrastiveRegimeLoss(nn.Module):
    """
    Encourages hidden states from different market regimes to be
    well-separated in the representation space.

    Bull regime h_slow vs Bear regime h_slow should be far apart.
    Same-regime states should cluster together.

    Uses InfoNCE-style contrastive objective.

    Args:
        temperature: softmax temperature (lower = sharper separation)
        threshold:   regime_prob boundaries for positive/negative assignment
    """

    def __init__(self, temperature: float = 0.1, threshold: float = 0.2):
        super().__init__()
        self.tau       = temperature
        self.threshold = threshold

    def forward(self, h_slow: torch.Tensor, regime_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_slow:      (batch, hidden_size)  slow hidden state
            regime_prob: (batch,)              scalar regime probability
        """
        p = regime_prob.squeeze(-1)
        mask_1 = p > (0.5 + self.threshold)
        mask_0 = p < (0.5 - self.threshold)

        if mask_1.sum() < 2 or mask_0.sum() < 2:
            return torch.tensor(0.0, device=h_slow.device)

        h1 = F.normalize(h_slow[mask_1], dim=-1)   # regime 1 states
        h0 = F.normalize(h_slow[mask_0], dim=-1)   # regime 0 states

        # Pull same-regime states together
        sim_pos = torch.matmul(h1, h1.T) / self.tau
        # Push different-regime states apart
        sim_neg = torch.matmul(h1, h0.T) / self.tau

        # InfoNCE: maximize pos similarity relative to neg
        loss = -sim_pos.mean() + torch.logsumexp(sim_neg, dim=-1).mean()
        return loss.clamp(min=0.0)


class CounterfactualLoss(nn.Module):
    """
    Combined training objective:

      L = L_nll
        + alpha * L_pred_coding
        + beta  * L_regime_entropy
        + gamma * L_contrastive

    Components:
      L_nll:            Gaussian NLL from UncertaintyGatedHead (replaces MSE)
      L_pred_coding:    predictive coding error accumulated inside CASA cells
      L_regime_entropy: encourages decisive (low-entropy) regime detection
      L_contrastive:    separates bull/bear regime hidden states

    Args:
        alpha: weight for predictive coding loss   (default 0.1)
        beta:  weight for regime entropy           (default 0.01)
        gamma: weight for contrastive regime loss  (default 0.05)
        use_nll: if True use NLL loss, else fall back to MSE
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta:  float = 0.01,
        gamma: float = 0.05,
        use_nll: bool = True,
    ):
        super().__init__()
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        self.use_nll = use_nll
        self.contrastive = ContrastiveRegimeLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred,            # mean: (B, T, out)  OR  (mean, std) tuple
        target: torch.Tensor,
        extra:  dict,
        h_slow: torch.Tensor = None,
    ) -> torch.Tensor:

        # Support both (mean, std) tuple from MultiScaleCASARNN
        # and plain tensor from original CASARNNModel
        if isinstance(pred, tuple):
            mean, std = pred
        else:
            mean, std = pred, None

        # Factual loss
        if self.use_nll and std is not None:
            l_factual = UncertaintyGatedHead.nll_loss(mean, std, target)
        else:
            l_factual = self.mse(mean, target)

        # Predictive coding loss
        l_pred = extra.get("pred_coding_loss", torch.tensor(0.0, device=mean.device))

        # Regime entropy regularization
        regime_probs = extra.get("regime_probs", None)
        if regime_probs is not None:
            p = regime_probs.clamp(1e-6, 1 - 1e-6)
            entropy  = -(p * p.log() + (1 - p) * (1 - p).log())
            l_regime = entropy.mean()
        else:
            l_regime = torch.tensor(0.0, device=mean.device)

        # Contrastive regime loss
        if h_slow is not None and regime_probs is not None:
            regime_flat = regime_probs.reshape(-1, regime_probs.shape[-1]).mean(-1)
            h_slow_flat = h_slow.reshape(-1, h_slow.shape[-1]) if h_slow.dim() == 3 else h_slow
            l_contrast  = self.contrastive(h_slow_flat, regime_flat)
        else:
            l_contrast = torch.tensor(0.0, device=mean.device)

        total = (l_factual
                 + self.alpha * l_pred
                 + self.beta  * l_regime
                 + self.gamma * l_contrast)
        return total
