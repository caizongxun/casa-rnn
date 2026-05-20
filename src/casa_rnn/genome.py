"""FeatureGenome: differentiable automatic feature evolution layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _rolling_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    B, T, D = x.shape
    kernel = torch.ones(D, 1, w, device=x.device) / w
    xp = F.pad(x.transpose(1, 2), (w - 1, 0))
    return F.conv1d(xp, kernel, groups=D).transpose(1, 2)


def _rolling_std(x: torch.Tensor, w: int) -> torch.Tensor:
    mu  = _rolling_mean(x, w)
    mu2 = _rolling_mean(x ** 2, w)
    return (mu2 - mu ** 2).clamp(min=0).sqrt()


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.abs() + 1e-6)


def _momentum(x: torch.Tensor, lag: int = 5) -> torch.Tensor:
    return x - x.roll(lag, dims=1)


def _soft_sign(x: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of sign via tanh."""
    return torch.tanh(x * 10.0)


class SoftOpBank(nn.Module):
    """
    DARTS-style continuous op relaxation.

    Gradient flow fixes vs previous version:
    - torch.sign removed -> soft_sign (tanh*10), has gradient everywhere
    - LayerNorm removed -> learnable gain/bias per feature
      LayerNorm normalises all outputs to same scale, so alpha cannot
      distinguish between ops by their magnitude -> grad ~= 0 on alpha.
      gain/bias preserves magnitude differences so alpha gets real signal.
    - alpha init scale increased to 2.0: makes initial softmax less uniform
      (entropy starts lower, easier to converge)
    - alpha_entropy_loss(): auxiliary loss added to training objective
      with warmup schedule so it dominates early training
    """

    OPS = [
        ("identity",  lambda x: x),
        ("lag1",      lambda x: x.roll(1, dims=1)),
        ("lag5",      lambda x: x.roll(5, dims=1)),
        ("lag20",     lambda x: x.roll(20, dims=1)),
        ("diff1",     lambda x: x - x.roll(1, dims=1)),
        ("diff5",     lambda x: _momentum(x, 5)),
        ("log",       _safe_log),
        ("soft_sign", _soft_sign),
        ("square",    lambda x: x ** 2),
        ("ma5",       lambda x: _rolling_mean(x, 5)),
        ("ma20",      lambda x: _rolling_mean(x, 20)),
        ("vol5",      lambda x: _rolling_std(x, 5)),
        ("vol20",     lambda x: _rolling_std(x, 20)),
        ("zscore20",  lambda x: (x - _rolling_mean(x, 20)) /
                                 (_rolling_std(x, 20) + 1e-6)),
        ("relu",      F.relu),
        ("tanh",      torch.tanh),
    ]
    N_OPS = len(OPS)

    def __init__(self, raw_size: int, feat_size: int):
        super().__init__()
        self.raw_size  = raw_size
        self.feat_size = feat_size
        # Larger init scale -> less uniform softmax -> lower starting entropy
        self.alpha = nn.Parameter(
            torch.randn(feat_size, raw_size, self.N_OPS) * 2.0
        )
        # Replace LayerNorm with affine scale so alpha sees magnitude differences
        self.gain = nn.Parameter(torch.ones(feat_size))
        self.bias = nn.Parameter(torch.zeros(feat_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops_out = torch.stack([op(x) for _, op in self.OPS], dim=-1)
        ops_out = ops_out.clamp(-10.0, 10.0)
        w = F.softmax(self.alpha, dim=-1)
        z = torch.einsum('btdo,fdo->btf', ops_out, w)
        return z * self.gain + self.bias

    def alpha_entropy_loss(self) -> torch.Tensor:
        """Minimising this pushes alpha toward sharp op choices (low entropy)."""
        w   = F.softmax(self.alpha, dim=-1)
        ent = -(w * (w + 1e-8).log()).sum(-1).mean()
        return ent

    def get_dominant_ops(self) -> dict:
        w        = F.softmax(self.alpha.detach(), dim=-1)
        dominant = w.argmax(dim=-1)
        names    = [n for n, _ in self.OPS]
        return {f"feat{f}_raw{d}": names[dominant[f, d].item()]
                for f in range(self.feat_size) for d in range(self.raw_size)}

    def get_op_entropy(self) -> float:
        w = F.softmax(self.alpha.detach(), dim=-1)
        return (-(w * (w + 1e-8).log()).sum(-1).mean()).item()


class CrossFeatureInteraction(nn.Module):
    def __init__(self, feat_size: int, top_k: int = 8):
        super().__init__()
        self.feat_size  = feat_size
        self.top_k      = top_k
        self.interact_w = nn.Parameter(torch.randn(feat_size, feat_size) * 0.3)
        self.out_proj   = nn.Linear(feat_size + top_k, feat_size)
        self.norm       = nn.LayerNorm(feat_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        mask = torch.sigmoid(self.interact_w)
        eye  = torch.eye(D, device=z.device)
        mask = mask * (1 - eye)
        mask = (mask + mask.T) / 2
        flat            = mask.reshape(-1)
        topk_vals, topk_idx = flat.topk(self.top_k)
        row_idx = topk_idx // D
        col_idx = topk_idx % D
        interact_terms = [z[:, :, row_idx[k].item()] * z[:, :, col_idx[k].item()] *
                          topk_vals[k] for k in range(self.top_k)]
        interact_out = torch.stack(interact_terms, dim=-1)
        return self.norm(self.out_proj(torch.cat([z, interact_out], dim=-1)))

    def get_top_interactions(self) -> list:
        mask = torch.sigmoid(self.interact_w.detach())
        flat = mask.reshape(-1)
        _, topk = flat.topk(self.top_k)
        return [(idx.item() // self.feat_size, idx.item() % self.feat_size, flat[idx].item())
                for idx in topk if idx.item() // self.feat_size != idx.item() % self.feat_size]


class TemporalGenome(nn.Module):
    LAGS = [1, 2, 3, 5, 8, 13, 20, 34, 55]

    def __init__(self, feat_size: int, regime_size: int = 1):
        super().__init__()
        n_lags        = len(self.LAGS)
        self.lag_gate = nn.Sequential(nn.Linear(regime_size, 32), nn.Tanh(), nn.Linear(32, n_lags))
        self.out_proj = nn.Linear(feat_size * n_lags, feat_size)
        self.norm     = nn.LayerNorm(feat_size)

    def forward(self, z: torch.Tensor, regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = z.shape
        lagged  = torch.stack([z.roll(lag, dims=1) for lag in self.LAGS], dim=2)
        if regime is not None:
            lag_w = F.softmax(self.lag_gate(regime), dim=-1).unsqueeze(-1)
        else:
            n     = len(self.LAGS)
            lag_w = torch.ones(B, T, n, 1, device=z.device) / n
        return self.norm(self.out_proj((lagged * lag_w).reshape(B, T, -1)))


class FeatureGenome(nn.Module):
    def __init__(self, raw_size: int, feat_size: int = 16, top_k_pairs: int = 8):
        super().__init__()
        self.soft_op  = SoftOpBank(raw_size, feat_size)
        self.cross    = CrossFeatureInteraction(feat_size, top_k=top_k_pairs)
        self.temporal = TemporalGenome(feat_size, regime_size=1)
        self.fusion   = nn.Sequential(
            nn.Linear(feat_size * 2, feat_size), nn.GELU(), nn.LayerNorm(feat_size)
        )
        self.feat_size = feat_size

    def forward(self, x: torch.Tensor, regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_op = self.soft_op(x)
        z_x  = self.cross(z_op)
        z_t  = self.temporal(z_op, regime)
        return self.fusion(torch.cat([z_x, z_t], dim=-1))

    def alpha_entropy_loss(self) -> torch.Tensor:
        return self.soft_op.alpha_entropy_loss()

    def get_evolved_feature_report(self) -> dict:
        return {
            "dominant_ops":     self.soft_op.get_dominant_ops(),
            "top_interactions": self.cross.get_top_interactions(),
            "op_entropy":       self.soft_op.get_op_entropy(),
        }
