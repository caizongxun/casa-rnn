"""FeatureGenome: differentiable automatic feature evolution layer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Utility: rolling statistics (differentiable, causal)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. SoftOpBank  (DARTS-style continuous relaxation)
# ---------------------------------------------------------------------------

class SoftOpBank(nn.Module):
    OPS = [
        ("identity",  lambda x: x),
        ("lag1",      lambda x: x.roll(1, dims=1)),
        ("lag5",      lambda x: x.roll(5, dims=1)),
        ("lag20",     lambda x: x.roll(20, dims=1)),
        ("diff1",     lambda x: x - x.roll(1, dims=1)),
        ("diff5",     lambda x: _momentum(x, 5)),
        ("log",       _safe_log),
        ("sign",      torch.sign),
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
        # Random init: break symmetry so different features explore different ops
        self.alpha = nn.Parameter(
            torch.randn(feat_size, raw_size, self.N_OPS) * 0.5
        )
        self.norm      = nn.LayerNorm(feat_size)
        self.raw_size  = raw_size
        self.feat_size = feat_size
        # Learnable output mix: raw_size ops -> feat_size
        self.out_mix = nn.Linear(raw_size, feat_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops_out = torch.stack([op(x) for _, op in self.OPS], dim=-1)  # (B,T,D,N_OPS)
        w = F.softmax(self.alpha, dim=-1)                              # (feat,raw,N_OPS)
        z = torch.einsum('btdo,fdo->btf', ops_out, w)                 # (B,T,feat)
        return self.norm(z)

    def get_dominant_ops(self) -> dict:
        w        = F.softmax(self.alpha.detach(), dim=-1)
        dominant = w.argmax(dim=-1)
        names    = [n for n, _ in self.OPS]
        result   = {}
        for f in range(self.feat_size):
            for d in range(self.raw_size):
                result[f"feat{f}_raw{d}"] = names[dominant[f, d].item()]
        return result

    def get_op_entropy(self) -> float:
        """Low entropy = ops have converged to sharp choices. High = still exploring."""
        w = F.softmax(self.alpha.detach(), dim=-1)
        ent = -(w * (w + 1e-8).log()).sum(-1).mean()
        return ent.item()


# ---------------------------------------------------------------------------
# 2. CrossFeatureInteraction  (learned sparse interaction matrix)
# ---------------------------------------------------------------------------

class CrossFeatureInteraction(nn.Module):
    """
    Learns multiplicative interactions between feature pairs.
    interact_w is initialized with random noise so different pairs
    start with different weights -> gradient can differentiate them.
    """

    def __init__(self, feat_size: int, top_k: int = 8):
        super().__init__()
        self.feat_size = feat_size
        self.top_k     = top_k
        # KEY FIX: random init instead of zeros
        # zeros -> sigmoid(0)=0.5 everywhere -> all pairs identical -> no gradient signal
        # random -> different starting weights -> gradient can select winners
        self.interact_w = nn.Parameter(
            torch.randn(feat_size, feat_size) * 0.3
        )
        self.out_proj = nn.Linear(feat_size + top_k, feat_size)
        self.norm     = nn.LayerNorm(feat_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        mask    = torch.sigmoid(self.interact_w)
        eye     = torch.eye(D, device=z.device)
        mask    = mask * (1 - eye)           # zero diagonal (no self-interact)
        mask    = (mask + mask.T) / 2        # symmetry

        flat            = mask.reshape(-1)
        topk_vals, topk_idx = flat.topk(self.top_k)
        row_idx         = topk_idx // D
        col_idx         = topk_idx % D

        interact_terms = []
        for k in range(self.top_k):
            i    = row_idx[k].item()
            j    = col_idx[k].item()
            term = z[:, :, i] * z[:, :, j] * topk_vals[k]
            interact_terms.append(term.unsqueeze(-1))

        interact_out = torch.cat(interact_terms, dim=-1)     # (B,T,top_k)
        combined     = torch.cat([z, interact_out], dim=-1)  # (B,T,D+top_k)
        return self.norm(self.out_proj(combined))

    def get_top_interactions(self) -> list:
        mask            = torch.sigmoid(self.interact_w.detach())
        flat            = mask.reshape(-1)
        _, topk_idx     = flat.topk(self.top_k)
        pairs = []
        for idx in topk_idx:
            i = idx.item() // self.feat_size
            j = idx.item() % self.feat_size
            if i != j:  # skip self-interactions in report
                w = flat[idx].item()
                pairs.append((i, j, w))
        return pairs


# ---------------------------------------------------------------------------
# 3. TemporalGenome  (regime-conditioned dynamic lag selection)
# ---------------------------------------------------------------------------

class TemporalGenome(nn.Module):
    LAGS = [1, 2, 3, 5, 8, 13, 20, 34, 55]  # Fibonacci lags

    def __init__(self, feat_size: int, regime_size: int = 1):
        super().__init__()
        n_lags          = len(self.LAGS)
        self.lag_gate   = nn.Sequential(
            nn.Linear(regime_size, 32),
            nn.Tanh(),
            nn.Linear(32, n_lags),
        )
        self.out_proj = nn.Linear(feat_size * n_lags, feat_size)
        self.norm     = nn.LayerNorm(feat_size)

    def forward(self, z: torch.Tensor, regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D  = z.shape
        lagged   = torch.stack([z.roll(lag, dims=1) for lag in self.LAGS], dim=2)  # (B,T,n,D)
        if regime is not None:
            lag_w = F.softmax(self.lag_gate(regime), dim=-1).unsqueeze(-1)  # (B,T,n,1)
        else:
            n     = len(self.LAGS)
            lag_w = torch.ones(B, T, n, 1, device=z.device) / n
        weighted = (lagged * lag_w).reshape(B, T, -1)
        return self.norm(self.out_proj(weighted))


# ---------------------------------------------------------------------------
# Full FeatureGenome
# ---------------------------------------------------------------------------

class FeatureGenome(nn.Module):
    def __init__(self, raw_size: int, feat_size: int = 16, top_k_pairs: int = 8):
        super().__init__()
        self.soft_op  = SoftOpBank(raw_size, feat_size)
        self.cross    = CrossFeatureInteraction(feat_size, top_k=top_k_pairs)
        self.temporal = TemporalGenome(feat_size, regime_size=1)
        self.fusion   = nn.Sequential(
            nn.Linear(feat_size * 2, feat_size),
            nn.GELU(),
            nn.LayerNorm(feat_size),
        )
        self.feat_size = feat_size

    def forward(self, x: torch.Tensor, regime: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_op = self.soft_op(x)
        z_x  = self.cross(z_op)
        z_t  = self.temporal(z_op, regime)
        return self.fusion(torch.cat([z_x, z_t], dim=-1))

    def get_evolved_feature_report(self) -> dict:
        return {
            "dominant_ops":     self.soft_op.get_dominant_ops(),
            "top_interactions": self.cross.get_top_interactions(),
            "op_entropy":       self.soft_op.get_op_entropy(),
        }
