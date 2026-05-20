"""
Topological Data Analysis feature extractor.

Uses approximate Betti numbers from correlation matrices as topology
features. Full persistent homology (ripser/giotto-tda) is optional;
the lightweight fallback computes:
  B0: number of connected components (correlation threshold graph)
  B1: approximate 1-cycle count via Euler characteristic

These capture:
  B0 high  -> market fragmentation (regime divergence)
  B1 high  -> correlated loops (systemic risk: A->B->C->A co-movement)

All thresholds are LEARNABLE (passed through sigmoid + scale), so the
model can tune its own topological sensitivity during training.
"""
import torch
import torch.nn as nn


class TDAFeatureExtractor(nn.Module):
    """
    Inputs:  x  (B, T_window, F)  — sliding window of features
    Outputs: topo (B, 3)  — [B0_norm, B1_approx, spectral_gap]
    """
    def __init__(self, feat_dim: int, window: int = 30):
        super().__init__()
        self.feat_dim = feat_dim
        self.window   = window
        # learnable correlation thresholds
        self.log_thresh_b0 = nn.Parameter(torch.tensor(0.0))   # sigmoid -> ~0.5
        self.log_thresh_b1 = nn.Parameter(torch.tensor(-0.5))  # sigmoid -> ~0.38
        self.proj = nn.Linear(3, 3)  # optional learned mixing of topo features

    @property
    def thresh_b0(self) -> torch.Tensor:
        return torch.sigmoid(self.log_thresh_b0)     # 0-1

    @property
    def thresh_b1(self) -> torch.Tensor:
        return torch.sigmoid(self.log_thresh_b1)     # 0-1

    def _corr_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> corr (B, F, F)"""
        x_c = x - x.mean(dim=1, keepdim=True)
        std = x_c.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_n = x_c / std
        corr = torch.bmm(x_n.transpose(1, 2), x_n) / x.shape[1]
        return corr.clamp(-1, 1)

    def _betti_approx(self, corr: torch.Tensor) -> tuple:
        """Differentiable soft-threshold Betti approximation."""
        # Soft adjacency: edge weight = sigmoid((|corr| - thresh) * 10)
        adj_b0 = torch.sigmoid((corr.abs() - self.thresh_b0) * 10.0)
        adj_b1 = torch.sigmoid((corr.abs() - self.thresh_b1) * 10.0)
        F_dim  = corr.shape[-1]

        # B0: V - E (normalised connected components proxy)
        E0 = adj_b0.sum(dim=(-2, -1)) / 2.0
        B0 = (F_dim - E0 / F_dim).clamp(min=0) / F_dim

        # B1: E - V + 1 (cycle rank proxy, Euler characteristic)
        E1 = adj_b1.sum(dim=(-2, -1)) / 2.0
        B1 = ((E1 / F_dim) - F_dim + 1).clamp(min=0) / F_dim

        # Spectral gap: difference of two smallest eigenvalues of Laplacian
        degree = adj_b0.sum(dim=-1)  # (B, F)
        L = torch.diag_embed(degree) - adj_b0
        try:
            eigvals = torch.linalg.eigvalsh(L)       # (B, F) ascending
            gap = (eigvals[:, 1] - eigvals[:, 0]).clamp(min=0) / (F_dim + 1e-6)
        except Exception:
            gap = torch.zeros(corr.shape[0], device=corr.device)

        return B0.unsqueeze(-1), B1.unsqueeze(-1), gap.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_window, F) -> topo: (B, 3)"""
        corr = self._corr_matrix(x)
        B0, B1, gap = self._betti_approx(corr)
        raw = torch.cat([B0, B1, gap], dim=-1)       # (B, 3)
        return self.proj(raw)                         # learnable mixing
