"""
Loss functions for CASA-RNN.

Changelog:
  v0.6 - DirectionLoss added: penalises wrong-sign predictions on
         timesteps where |target| > median*0.5 (filters near-zero noise)
       - nll_clamp raised 3.0 -> 10.0 so std cannot inflate without cost
       - std_penalty added: L1 on mean(std) pushes model to produce
         tight intervals, fixing q_hat blowup
       - BioConstraintLoss wired into CounterfactualLoss.forward via
         ne/da/vov/rpe tensors from extra dict (was defined but never called)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveRegimeLoss(nn.Module):
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
        h1      = F.normalize(h_slow[mask_1], dim=-1)
        h0      = F.normalize(h_slow[mask_0], dim=-1)
        sim_pos = torch.matmul(h1, h1.T) / self.tau
        sim_neg = torch.matmul(h1, h0.T) / self.tau
        return (-sim_pos.mean() + torch.logsumexp(sim_neg, dim=-1).mean()).clamp(min=0.0)


class BioConstraintLoss(nn.Module):
    """
    Semantic alignment constraints for neuromodulators.

    1. NE should be positively correlated with vol_of_vol
    2. DA should be positively correlated with RPE
    """
    def __init__(self, ne_weight: float = 0.05, da_weight: float = 0.05):
        super().__init__()
        self.ne_w = ne_weight
        self.da_w = da_weight

    def _soft_corr_penalty(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.flatten()
        b = b.flatten()
        if a.numel() < 2:
            return torch.tensor(0.0, device=a.device)
        a_c   = a - a.mean()
        b_c   = b - b.mean()
        denom = (a_c.norm() * b_c.norm()).clamp(min=1e-8)
        corr  = (a_c * b_c).sum() / denom
        return F.relu(-corr)

    def forward(
        self,
        ne: torch.Tensor,
        vol_of_vol: torch.Tensor,
        da: torch.Tensor,
        rpe: torch.Tensor,
    ) -> torch.Tensor:
        ne_penalty = self._soft_corr_penalty(ne, vol_of_vol)
        da_penalty = self._soft_corr_penalty(da, rpe)
        return self.ne_w * ne_penalty + self.da_w * da_penalty


class DirectionLoss(nn.Module):
    """
    v0.6: Penalises wrong-sign predictions.

    Only applied on timesteps where |target| > dynamic threshold
    (median * 0.5) to avoid penalising near-zero log returns where
    direction is meaningless noise.

    Loss = mean( ReLU( -sign(pred) * sign(target) ) ) on masked steps
         = 0 if sign matches, 1 if sign wrong
    """
    def __init__(self, weight: float = 0.3, threshold_factor: float = 0.5):
        super().__init__()
        self.weight           = weight
        self.threshold_factor = threshold_factor

    def forward(self, pred_mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_target = target.abs()
        threshold  = abs_target.median() * self.threshold_factor
        mask       = (abs_target > threshold).float()
        # +1 if wrong direction, 0 if correct
        wrong      = F.relu(-(pred_mean * target).sign().detach() *
                            torch.ones_like(pred_mean))
        # use soft version: hinge on pred*target
        wrong_soft = F.relu(1.0 - pred_mean * target / (abs_target.clamp(min=1e-8)))
        return self.weight * (wrong_soft * mask).mean()


class CounterfactualLoss(nn.Module):
    """
    Combined training objective with regime-aware scaling.

    v0.6 changes:
    - nll_clamp: 3.0 -> 10.0  (allow real NLL signal through)
    - std_penalty: L1 regulariser on predicted std to prevent inflation
    - DirectionLoss wired in
    - BioConstraintLoss wired in via extra dict tensors
    """
    def __init__(
        self,
        alpha: float          = 0.1,
        beta:  float          = 0.01,
        gamma: float          = 0.05,
        direction_weight: float = 0.3,
        bio_weight: float     = 0.05,
        std_penalty_weight: float = 0.02,
        use_nll:   bool       = True,
        nll_clamp: float      = 10.0,
        regime_scale: bool    = True,
    ):
        super().__init__()
        self.alpha              = alpha
        self.beta               = beta
        self.gamma              = gamma
        self.direction_weight   = direction_weight
        self.bio_weight         = bio_weight
        self.std_penalty_weight = std_penalty_weight
        self.use_nll            = use_nll
        self.nll_clamp          = nll_clamp
        self.regime_scale       = regime_scale
        self.contrastive        = ContrastiveRegimeLoss()
        self.direction          = DirectionLoss(weight=direction_weight)
        self.bio_constraint     = BioConstraintLoss()
        self.mse                = nn.MSELoss()

    def forward(
        self,
        pred,
        target: torch.Tensor,
        extra:  dict,
        h_slow: torch.Tensor = None,
    ) -> torch.Tensor:
        mean, std = pred if isinstance(pred, tuple) else (pred, None)

        # --- factual NLL / MSE ---
        if self.use_nll and std is not None:
            dist      = torch.distributions.Normal(mean, std.clamp(min=1e-4))
            nll_elem  = -dist.log_prob(target)
            l_factual = nll_elem.clamp(-self.nll_clamp, self.nll_clamp).mean()
        else:
            l_factual = self.mse(mean, target)

        # v0.6: std penalty -- pushes predicted std to stay small/calibrated
        l_std_penalty = torch.tensor(0.0, device=mean.device)
        if std is not None:
            l_std_penalty = self.std_penalty_weight * std.mean()

        # --- regime scaling ---
        regime_probs = extra.get("regime_probs", None)
        if self.regime_scale and regime_probs is not None:
            certainty  = (regime_probs.mean() - 0.5).abs() * 2
            loss_scale = 0.3 + 0.7 * certainty
            l_factual  = l_factual * loss_scale

        # --- auxiliary losses ---
        l_pred = extra.get("pred_coding_loss", torch.tensor(0.0, device=mean.device))

        if regime_probs is not None:
            p        = regime_probs.clamp(1e-6, 1 - 1e-6)
            l_regime = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        else:
            l_regime = torch.tensor(0.0, device=mean.device)

        if h_slow is not None and regime_probs is not None:
            regime_flat = regime_probs.reshape(-1, regime_probs.shape[-1]).mean(-1)
            h_flat      = h_slow.reshape(-1, h_slow.shape[-1]) if h_slow.dim() == 3 else h_slow
            l_contrast  = self.contrastive(h_flat, regime_flat)
        else:
            l_contrast = torch.tensor(0.0, device=mean.device)

        # v0.6: direction loss
        l_direction = self.direction(mean, target)

        # v0.6: bio constraint loss -- NE~vov, DA~RPE
        l_bio = torch.tensor(0.0, device=mean.device)
        neuro_tensors = extra.get("neuro_tensors", {})
        vov_full      = extra.get("vol_of_vol", None)
        rpe_full      = extra.get("rpe_tensor", None)
        if neuro_tensors and vov_full is not None and rpe_full is not None:
            ne_t = neuro_tensors.get("norepinephrine", None)
            da_t = neuro_tensors.get("dopamine", None)
            if ne_t is not None and da_t is not None:
                l_bio = self.bio_weight * self.bio_constraint(
                    ne=ne_t, vol_of_vol=vov_full,
                    da=da_t, rpe=rpe_full,
                )

        return (
            l_factual
            + self.alpha   * l_pred
            + self.beta    * l_regime
            + self.gamma   * l_contrast
            + l_direction
            + l_bio
            + l_std_penalty
        )
