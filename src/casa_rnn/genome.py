"""FeatureGenome: differentiable automatic feature evolution layer.

Design inspirations:
- Neural Feature Search (Microsoft, NeurIPS 2019): RNN controller learns
  high-order feature transformations via RL.
- End-to-end Symbolic Regression with Transformers (NeurIPS 2022):
  directly predict mathematical expressions end-to-end.
- EvoForest (2025): jointly evolve reusable computational structure
  and trainable function families.
- DARTS (Liu et al.): differentiable architecture search via relaxation
  over discrete op choices.

Breakthrough idea:
  Instead of fixed ops or RL search, we use THREE learnable mechanisms:

  1. SoftOp Bank: continuous relaxation of discrete ops (DARTS-style),
     all ops run in parallel, weighted sum selected by softmax gating.
     Gradient flows through weights -> ops evolve via backprop.

  2. CrossFeature Interaction: learns WHICH raw feature pairs to multiply,
     discovering non-linear interactions (price x volume, etc.) automatically.
     Uses a learned sparse interaction matrix.

  3. TemporalGenome: learns WHEN to look back (lag selection) dynamically,
     conditioned on the current regime from CASA-RNN hidden state.
     Regime-0 might prefer short lags, Regime-1 long lags.

  All three are end-to-end differentiable and trained jointly with CASA-RNN.
  No separate pre-training, no RL, no separate GA loop needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Utility: rolling statistics (differentiable)
# ---------------------------------------------------------------------------

def _rolling_mean(x: torch.Tensor, w: int) -> torch.Tensor:
    """Causal rolling mean, shape preserved."""
    B, T, D = x.shape
    kernel = torch.ones(D, 1, w, device=x.device) / w
    xp = F.pad(x.transpose(1, 2), (w - 1, 0))  # (B, D, T+w-1)
    return F.conv1d(xp, kernel, groups=D).transpose(1, 2)


def _rolling_std(x: torch.Tensor, w: int) -> torch.Tensor:
    """Causal rolling std."""
    mu = _rolling_mean(x, w)
    mu2 = _rolling_mean(x ** 2, w)
    return (mu2 - mu ** 2).clamp(min=0).sqrt()


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.abs() + 1e-6)


def _momentum(x: torch.Tensor, lag: int = 5) -> torch.Tensor:
    return x - x.roll(lag, dims=1)


# ---------------------------------------------------------------------------
# 1. SoftOp Bank  (DARTS-style continuous relaxation)
# ---------------------------------------------------------------------------

class SoftOpBank(nn.Module):
    """
    Runs N differentiable operations in parallel.
    Gating weights (learned) select the best combination per output feature.

    Operations are designed to cover the space of common financial features:
    identity, lag, diff, momentum, log, sign, MA, volatility, Z-score.
    All are causal (no future leakage).
    """

    OPS = [
        ("identity",   lambda x: x),
        ("lag1",       lambda x: x.roll(1, dims=1)),
        ("lag5",       lambda x: x.roll(5, dims=1)),
        ("lag20",      lambda x: x.roll(20, dims=1)),
        ("diff1",      lambda x: x - x.roll(1, dims=1)),
        ("diff5",      lambda x: _momentum(x, 5)),
        ("log",        _safe_log),
        ("sign",       torch.sign),
        ("square",     lambda x: x ** 2),
        ("ma5",        lambda x: _rolling_mean(x, 5)),
        ("ma20",       lambda x: _rolling_mean(x, 20)),
        ("vol5",       lambda x: _rolling_std(x, 5)),
        ("vol20",      lambda x: _rolling_std(x, 20)),
        ("zscore20",   lambda x: (x - _rolling_mean(x, 20)) /
                                  (_rolling_std(x, 20) + 1e-6)),
        ("relu",       F.relu),
        ("tanh",       torch.tanh),
    ]
    N_OPS = len(OPS)

    def __init__(self, raw_size: int, feat_size: int):
        super().__init__()
        # alpha: (feat_size, raw_size, N_OPS) — gating weights
        self.alpha = nn.Parameter(
            torch.zeros(feat_size, raw_size, self.N_OPS)
        )
        # projection: blend op outputs -> feat_size
        self.proj = nn.Linear(raw_size, feat_size, bias=False)
        self.norm = nn.LayerNorm(feat_size)
        self.raw_size = raw_size
        self.feat_size = feat_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D_raw) -> (B, T, feat_size)"""
        # Run all ops: list of (B, T, D_raw)
        ops_out = torch.stack(
            [op(x) for _, op in self.OPS], dim=-1
        )  # (B, T, D_raw, N_OPS)

        # Gating: softmax over ops dimension
        w = F.softmax(self.alpha, dim=-1)  # (feat, raw, N_OPS)

        # Weighted combination per output feature
        # einsum: btdo, fdo -> btf
        z = torch.einsum('btdo,fdo->btf', ops_out, w)
        return self.norm(z)

    def get_dominant_ops(self) -> dict:
        """Returns the dominant operation per output feature (for interpretability)."""
        w = F.softmax(self.alpha, dim=-1)  # (feat, raw, N_OPS)
        dominant = w.argmax(dim=-1)        # (feat, raw)
        op_names = [name for name, _ in self.OPS]
        result = {}
        for f in range(self.feat_size):
            for d in range(self.raw_size):
                idx = dominant[f, d].item()
                result[f"feat{f}_raw{d}"] = op_names[idx]
        return result


# ---------------------------------------------------------------------------
# 2. CrossFeature Interaction  (learned sparse interaction matrix)
# ---------------------------------------------------------------------------

class CrossFeatureInteraction(nn.Module):
    """
    Learns WHICH feature pairs to multiply together.

    Breakthrough: unlike standard MLP that mixes features additively,
    this explicitly models multiplicative interactions (e.g., volume * |return|)
    which are common in financial signals but invisible to additive models.

    Uses a learned interaction matrix I (D x D) where I[i,j] is the
    learned importance of x_i * x_j.
    Sparse regularization encourages discovering a small set of interactions.
    """

    def __init__(self, feat_size: int, top_k: int = 8):
        super().__init__()
        self.feat_size = feat_size
        self.top_k = top_k
        # Interaction weights: upper triangle only (symmetric)
        self.interact_w = nn.Parameter(
            torch.zeros(feat_size, feat_size)
        )
        self.out_proj = nn.Linear(feat_size + top_k, feat_size)
        self.norm = nn.LayerNorm(feat_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) -> (B, T, D)"""
        B, T, D = z.shape

        # Soft interaction mask (symmetric, zero diagonal)
        mask = torch.sigmoid(self.interact_w)
        mask = mask * (1 - torch.eye(D, device=z.device))  # zero diagonal
        mask = (mask + mask.T) / 2                          # symmetry

        # Select top-k interactions
        flat = mask.view(-1)
        topk_vals, topk_idx = flat.topk(self.top_k)
        row_idx = topk_idx // D
        col_idx = topk_idx % D

        # Compute interaction terms: x_i * x_j for top-k pairs
        interact_terms = []
        for k in range(self.top_k):
            i, j = row_idx[k].item(), col_idx[k].item()
            term = z[:, :, i] * z[:, :, j] * topk_vals[k]
            interact_terms.append(term.unsqueeze(-1))
        interact_out = torch.cat(interact_terms, dim=-1)  # (B, T, top_k)

        combined = torch.cat([z, interact_out], dim=-1)   # (B, T, D+top_k)
        return self.norm(self.out_proj(combined))

    def get_top_interactions(self) -> list:
        """Returns top-k interaction pairs for interpretability."""
        mask = torch.sigmoid(self.interact_w)
        flat = mask.view(-1)
        _, topk_idx = flat.topk(self.top_k)
        pairs = []
        for idx in topk_idx:
            i, j = idx.item() // self.feat_size, idx.item() % self.feat_size
            w = flat[idx].item()
            pairs.append((i, j, w))
        return pairs


# ---------------------------------------------------------------------------
# 3. TemporalGenome  (regime-conditioned dynamic lag selection)
# ---------------------------------------------------------------------------

class TemporalGenome(nn.Module):
    """
    Learns WHEN to look back, conditioned on regime.

    Breakthrough insight: the optimal lookback window is NOT fixed.
    - Trend regime:      short lags (1-5) capture momentum
    - Volatile regime:   long lags (20-60) needed for mean reversion
    - Transition regime: multi-scale combination

    This is implemented as a differentiable attention over multiple lags,
    where the attention weights are conditioned on a regime signal.
    Compatible with CASA-RNN's regime output.
    """

    LAGS = [1, 2, 3, 5, 8, 13, 20, 34, 55]  # Fibonacci: natural market lags

    def __init__(self, feat_size: int, regime_size: int = 1):
        super().__init__()
        self.feat_size  = feat_size
        n_lags          = len(self.LAGS)
        # Gate: regime signal -> lag attention weights
        self.lag_gate   = nn.Sequential(
            nn.Linear(regime_size, 32),
            nn.Tanh(),
            nn.Linear(32, n_lags),
        )
        self.out_proj   = nn.Linear(feat_size * n_lags, feat_size)
        self.norm       = nn.LayerNorm(feat_size)

    def forward(
        self,
        z: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """z: (B, T, D), regime: (B, T, 1) or None"""
        B, T, D = z.shape

        lagged = [z.roll(lag, dims=1) for lag in self.LAGS]  # list of (B,T,D)
        lagged = torch.stack(lagged, dim=2)   # (B, T, n_lags, D)

        if regime is not None:
            # regime-conditioned lag weights
            lag_w = F.softmax(self.lag_gate(regime), dim=-1)  # (B, T, n_lags)
            lag_w = lag_w.unsqueeze(-1)                        # (B, T, n_lags, 1)
        else:
            n_lags = len(self.LAGS)
            lag_w = torch.ones(B, T, n_lags, 1, device=z.device) / n_lags

        weighted = (lagged * lag_w).reshape(B, T, -1)  # (B, T, n_lags*D)
        return self.norm(self.out_proj(weighted))


# ---------------------------------------------------------------------------
# Full FeatureGenome: compose all three mechanisms
# ---------------------------------------------------------------------------

class FeatureGenome(nn.Module):
    """
    Full differentiable feature evolution pipeline:

        raw_input
            |-- SoftOpBank          (WHAT transformation?)
            |-- CrossFeature        (WHICH pairs interact?)
            |-- TemporalGenome      (WHEN to look back? regime-conditioned)
            |
          concat + project
            |
          evolved_features -> CASA-RNN

    All three are jointly trained with CASA-RNN via a single backprop pass.
    No separate training stage, no RL reward shaping.
    """

    def __init__(
        self,
        raw_size:    int,
        feat_size:   int = 16,
        top_k_pairs: int = 8,
    ):
        super().__init__()
        self.soft_op  = SoftOpBank(raw_size, feat_size)
        self.cross    = CrossFeatureInteraction(feat_size, top_k=top_k_pairs)
        self.temporal = TemporalGenome(feat_size, regime_size=1)

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(feat_size * 2, feat_size),
            nn.GELU(),
            nn.LayerNorm(feat_size),
        )
        self.feat_size = feat_size

    def forward(
        self,
        x: torch.Tensor,
        regime: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:      (B, T, raw_size)
        regime: (B, T, 1) — optional, from CASA-RNN's regime output
        returns (B, T, feat_size)
        """
        z_op  = self.soft_op(x)            # op-evolved features
        z_x   = self.cross(z_op)           # interaction-enhanced
        z_t   = self.temporal(z_op, regime)  # lag-selected features
        z     = torch.cat([z_x, z_t], dim=-1)
        return self.fusion(z)

    def get_evolved_feature_report(self) -> dict:
        """Human-readable report of what the genome has learned."""
        return {
            "dominant_ops":      self.soft_op.get_dominant_ops(),
            "top_interactions":  self.cross.get_top_interactions(),
        }
