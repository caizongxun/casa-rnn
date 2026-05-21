"""
encoder.py  v5 -- Multi-scale PatchEncoder

New in v5
---------
* RMSNorm  replaces LayerNorm (faster, more stable on small models)
* RoPE     replaces sinusoidal + learned residual PE
* Multi-scale patching: patch_sizes=[4,12,24] each projected to d_model//3,
  then concatenated -> d_model before transformer
* Variable Selection Network (VSN) gates each input feature channel
  before patching, letting the model learn per-step feature importance
* 1D-Conv stem (depthwise) extracts local patterns before patch embedding
* ZScoreNorm extended: volume-class channels (idx>=4) all log1p-transformed
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.w * x / rms


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------
def _rope_freqs(d: int, max_len: int = 512, base: int = 10000) -> torch.Tensor:
    theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    t     = torch.arange(max_len).float()
    freqs = torch.outer(t, theta)           # (max_len, d//2)
    return torch.cat([freqs, freqs], dim=-1) # (max_len, d)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # x: (B, T, d)   freqs: (T, d)
    cos = freqs.cos().unsqueeze(0)   # (1, T, d)
    sin = freqs.sin().unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# ZScore normalisation  (extended for arbitrary #channels)
# ---------------------------------------------------------------------------
class ZScoreNorm(nn.Module):
    """
    Per-window per-channel z-score.  Channels at index >= 4 are
    log1p-transformed first (volume-class features).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        if x.shape[-1] > 4:
            x[..., 4:] = torch.log1p(x[..., 4:].clamp(min=0))
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1,  keepdim=True).clamp(min=1e-6)
        return (x - mean) / std


# ---------------------------------------------------------------------------
# Variable Selection Network (VSN)
# ---------------------------------------------------------------------------
class VSN(nn.Module):
    """
    Learns a soft per-channel gate at each timestep.
    Input:  (B, T, C)  ->  Output: (B, T, C)
    """
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate computed from the mean across time -> (B, C)
        gate = self.fc(x.mean(dim=1))        # (B, C)
        return x * gate.unsqueeze(1)         # (B, T, C)


# ---------------------------------------------------------------------------
# Single-scale patch branch
# ---------------------------------------------------------------------------
class PatchBranch(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, out_dim: int):
        super().__init__()
        self.patch_size = patch_size
        # depthwise 1D-conv before patch (kernel=3, groups=in_channels)
        self.conv = nn.Conv1d(
            in_channels, in_channels,
            kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.stem = nn.Linear(in_channels * patch_size, out_dim)
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        # pad to multiple of patch_size
        pad = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        # conv on (B, C, T)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        n_p = x.shape[1] // self.patch_size
        x   = x.reshape(B, n_p, self.patch_size * C)
        return self.stem(x)   # (B, n_patches, out_dim)


# ---------------------------------------------------------------------------
# PatchEncoder  (main class)
# ---------------------------------------------------------------------------
class PatchEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int  = 31,
        patch_sizes:  list = None,   # default [4, 12, 24]
        d_model: int       = 128,
        n_heads: int       = 4,
        n_layers: int      = 3,
        dropout: float     = 0.1,
        max_patches: int   = 256,
    ):
        super().__init__()
        if patch_sizes is None:
            patch_sizes = [4, 12, 24]
        self.patch_sizes  = patch_sizes
        self.d_model      = d_model
        self.in_channels  = in_channels
        # legacy compat
        self.patch_size   = patch_sizes[0]

        branch_dim = d_model // len(patch_sizes)
        remainder  = d_model - branch_dim * (len(patch_sizes) - 1)
        branch_dims = [branch_dim] * (len(patch_sizes) - 1) + [remainder]

        self.norm   = ZScoreNorm()
        self.vsn    = VSN(in_channels)
        self.branches = nn.ModuleList([
            PatchBranch(in_channels, ps, bd)
            for ps, bd in zip(patch_sizes, branch_dims)
        ])
        self.branch_proj = nn.Linear(d_model, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # RoPE freqs buffer (longest branch uses fewest patches)
        self.max_patches = max_patches
        rope_f = _rope_freqs(d_model, max_patches + 1)
        self.register_buffer("rope_freqs", rope_f)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.norm(x)
        x = self.vsn(x)

        # multi-scale branches: each (B, n_pi, branch_dim)
        # interpolate to the same length as branch[0]
        branches = [br(x) for br in self.branches]
        ref_len  = branches[0].shape[1]
        aligned  = []
        for b in branches:
            if b.shape[1] != ref_len:
                b = b.permute(0, 2, 1)                  # (B, d, n)
                b = F.interpolate(b, size=ref_len, mode="linear", align_corners=False)
                b = b.permute(0, 2, 1)
            aligned.append(b)
        tokens = self.branch_proj(torch.cat(aligned, dim=-1))  # (B, ref_len, d_model)

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, ref_len+1, d_model)

        # RoPE
        n_seq  = tokens.shape[1]
        freqs  = self.rope_freqs[:n_seq]
        tokens = apply_rope(tokens, freqs)

        tokens = self.transformer(tokens)
        out    = self.out_norm(tokens[:, 0])
        return out
