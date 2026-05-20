"""Combined loss functions for CASA-RNN training."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .heads import UncertaintyGatedHead


class ContrastiveRegimeLoss(nn.Module):
    """
    InfoNCE-style contrastive loss that separates bull/bear regime hidden states.
    """
    def __init__(self, temperature: float = 0.1, threshold: float = 0.2):
        super().__init__()
        self.tau       = temperature
        self.threshold = threshold

    def forward(self, h_slow: torch.Tensor, regime_prob: torch.Tensor) -> torch.Tensor:
        p      = regime_prob.squeeze(-1)
        mask_1 = p > (0.5 + self.threshold)
        mask_0 = p < (0.5 - self.threshold)
        if mask_1.sum() < 2 or mask_0.sum() < 2:
            return torch.tensor(0.0, device=h_slow.device)
        h1 = F.normalize(h_slow[mask_1], dim=-1)
        h0 = F.normalize(h_slow[mask_0], dim=-1)
        sim_pos = torch.matmul(h1, h1.T) / self.tau
        sim_neg = torch.matmul(h1, h0.T) / self.tau
        loss = -sim_pos.mean() + torch.logsumexp(sim_neg, dim=-1).mean()
        return loss.clamp(min=0.0)


class CounterfactualLoss(nn.Module):
    """
    Combined training objective with adaptive NLL clamping.

    Key fix: NLL loss is clamped to prevent std explosion from
    dominating the loss during regime transitions.
    High uncertainty is still allowed, but not rewarded infinitely.

    L = L_nll_clamped + alpha*L_pred + beta*L_entropy + gamma*L_contrast
    """

    def __init__(
        self,
        alpha: float = 0.1,
        beta:  float = 0.01,
        gamma: float = 0.05,
        use_nll: bool = True,
        nll_clamp: float = 3.0,   # clamp NLL per-element to prevent explosion
    ):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.gamma     = gamma
        self.use_nll   = use_nll
        self.nll_clamp = nll_clamp
        self.contrastive = ContrastiveRegimeLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred,
        target: torch.Tensor,
        extra:  dict,
        h_slow: torch.Tensor = None,
    ) -> torch.Tensor:

        if isinstance(pred, tuple):
            mean, std = pred
        else:
            mean, std = pred, None

        # Clamped NLL: per-element loss clamped before averaging
        # prevents a single std explosion from derailing training
        if self.use_nll and std is not None:
            dist      = torch.distributions.Normal(mean, std.clamp(min=1e-4))
            nll_elem  = -dist.log_prob(target)                    # (B, T, out)
            l_factual = nll_elem.clamp(-self.nll_clamp, self.nll_clamp).mean()
        else:
            l_factual = self.mse(mean, target)

        l_pred = extra.get("pred_coding_loss", torch.tensor(0.0, device=mean.device))

        regime_probs = extra.get("regime_probs", None)
        if regime_probs is not None:
            p        = regime_probs.clamp(1e-6, 1 - 1e-6)
            l_regime = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        else:
            l_regime = torch.tensor(0.0, device=mean.device)

        if h_slow is not None and regime_probs is not None:
            regime_flat = regime_probs.reshape(-1, regime_probs.shape[-1]).mean(-1)
            h_slow_flat = h_slow.reshape(-1, h_slow.shape[-1]) if h_slow.dim() == 3 else h_slow
            l_contrast  = self.contrastive(h_slow_flat, regime_flat)
        else:
            l_contrast = torch.tensor(0.0, device=mean.device)

        return l_factual + self.alpha*l_pred + self.beta*l_regime + self.gamma*l_contrast
