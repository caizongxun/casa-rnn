"""
encoder.py -- PatchEncoder  (v4)

Fix log
-------
* ZScoreNorm now log1p-transforms the volume channel (index 4) before
  z-scoring.  Raw volume has a vastly different scale from OHLC prices;
  without log compression the volume channel dominates the norm and
  washes out price gradient signals.
* Everything else unchanged from v3.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ZScoreNorm(nn.Module):
    """
    Per-window, per-channel z-score normalisation.
    Volume channel (index 4) is log1p-transformed first.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)  columns = [open, high, low, close, volume]
        x = x.clone()
        if x.shape[-1] > 4:
            x[..., 4] = torch.log1p(x[..., 4].clamp(min=0))
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1,  keepdim=True).clamp(min=1e-6)
        return (x - mean) / std


class SoftFeatureGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Parameter(torch.empty(d_model).uniform_(-1.0, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate)


def _sinusoidal_encoding(n_pos: int, d_model: int) -> torch.Tensor:
    pe  = torch.zeros(n_pos, d_model)
    pos = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class PatchEncoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 5,
        patch_size: int = 4,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_patches: int = 256,
    ):
        super().__init__()
        self.patch_size  = patch_size
        self.d_model     = d_model
        self.max_patches = max_patches

        self.norm = ZScoreNorm()

        self.stem = nn.Linear(in_channels * patch_size, d_model)
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        sin_pe = _sinusoidal_encoding(max_patches + 1, d_model)
        self.register_buffer("sin_pe", sin_pe)
        self.pos_residual = nn.Embedding(max_patches + 1, d_model)
        nn.init.normal_(self.pos_residual.weight, std=0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.gate     = SoftFeatureGate(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.norm(x)

        pad = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        n_patches = x.shape[1] // self.patch_size

        x   = x.reshape(B, n_patches, self.patch_size * C)
        x   = self.stem(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        n_seq     = x.shape[1]
        positions = torch.arange(n_seq, device=x.device)
        x = x + self.sin_pe[:n_seq] + self.pos_residual(positions)

        x = self.transformer(x)
        x = x[:, 0]
        x = self.gate(x)
        x = self.out_norm(x)
        return x
