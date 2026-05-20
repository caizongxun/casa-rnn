"""
encoder.py -- PatchEncoder

Converts raw OHLCV candle sequences into rich latent representations
without any hand-crafted features.  The model is free to discover
whatever structure is useful.

Design
------
* Input  : (B, T, 5)  raw [open, high, low, close, volume], all normalised
           per-window via robust scaling inside the encoder.
* Patch  : T candles split into non-overlapping patches of size P.
           Each patch is projected to d_model via a linear stem.
           This gives the model a local receptive field first.
* Temporal Transformer : multi-head self-attention across patches.
           The model attends freely -- no positional bias forced on it.
* Gate   : a learned soft feature-importance vector, initialised with
           random noise so dimensions diverge from the start.
           Over training it evolves to weight market regimes differently.
* Output : (B, d_model)  context vector, one per window.

Changes (v2)
------------
* Gate init: uniform random in [-1, 1] instead of ones -> forces diversity
* Positional encoding: sinusoidal base + learnable residual
  (sinusoid gives a warm start; learnable residual lets model adapt freely)
* stem init: Xavier uniform for better initial gradient flow
* UserWarning suppressed: switched enable_nested_tensor=False explicitly
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RobustNorm(nn.Module):
    """Per-window, per-channel robust normalisation (median / IQR)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        med = x.median(dim=1, keepdim=True).values
        q75 = x.quantile(0.75, dim=1, keepdim=True)
        q25 = x.quantile(0.25, dim=1, keepdim=True)
        iqr = (q75 - q25).clamp(min=1e-6)
        return (x - med) / iqr


class SoftFeatureGate(nn.Module):
    """
    Learnable gate over the d_model dimension.

    Initialised with random values so each dimension starts at a different
    importance level -- this forces diversity from epoch 1 and gives the
    model something to differentiate during backprop.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Random init in [-1, 1]: sigmoid maps this to (0.27, 0.73)
        # so no dimension is fully open or fully closed at start,
        # but they are all different -> gradient can flow asymmetrically
        gate_init = torch.empty(d_model).uniform_(-1.0, 1.0)
        self.gate = nn.Parameter(gate_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate)


def _sinusoidal_encoding(n_pos: int, d_model: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (n_pos, d_model)."""
    pe = torch.zeros(n_pos, d_model)
    pos = torch.arange(n_pos, dtype=torch.float).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float)
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class PatchEncoder(nn.Module):
    """
    Parameters
    ----------
    in_channels : int   number of raw input channels (default 5 = OHLCV)
    patch_size  : int   candles per patch (default 4)
    d_model     : int   transformer hidden dim
    n_heads     : int   attention heads
    n_layers    : int   transformer depth
    dropout     : float
    max_patches : int   maximum number of patches (default 256)
    """

    def __init__(
        self,
        in_channels: int = 5,
        patch_size: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_patches: int = 256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.max_patches = max_patches

        self.norm = RobustNorm()

        # Patch stem: flatten patch -> project to d_model
        self.stem = nn.Linear(in_channels * patch_size, d_model)
        nn.init.xavier_uniform_(self.stem.weight)
        nn.init.zeros_(self.stem.bias)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding:
        # sinusoidal base (frozen) + learnable residual (free to adapt)
        sin_pe = _sinusoidal_encoding(max_patches + 1, d_model)  # +1 for CLS
        self.register_buffer("sin_pe", sin_pe)
        self.pos_residual = nn.Embedding(max_patches + 1, d_model)
        nn.init.normal_(self.pos_residual.weight, std=0.01)  # small residual

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            # suppress the nested_tensor warning on CPU
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.gate = SoftFeatureGate(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C)  raw OHLCV
        returns (B, d_model)
        """
        B, T, C = x.shape
        x = self.norm(x)                         # robust normalise

        # Pad T to be divisible by patch_size
        pad = (self.patch_size - T % self.patch_size) % self.patch_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        n_patches = x.shape[1] // self.patch_size

        # Reshape into patches
        x = x.reshape(B, n_patches, self.patch_size * C)  # (B, n_patches, P*C)
        x = self.stem(x)                                    # (B, n_patches, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                     # (B, 1+n_patches, d_model)

        # Positional encoding: sinusoidal warm-start + learnable residual
        n_seq = x.shape[1]
        positions = torch.arange(n_seq, device=x.device)
        pe = self.sin_pe[:n_seq] + self.pos_residual(positions)
        x = x + pe

        x = self.transformer(x)                            # (B, 1+n_patches, d_model)
        x = x[:, 0]                                        # CLS token = global context
        x = self.gate(x)
        x = self.out_norm(x)
        return x                                           # (B, d_model)
