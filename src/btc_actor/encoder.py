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
* Gate   : a learned soft feature-importance vector updated each forward
           pass.  Acts as a trainable "what matters now" signal.
           Over training it evolves to weight market regimes differently.
* Output : (B, d_model)  context vector, one per window.

The encoder is shared between the supervised pre-training stage and
the RL fine-tune stage -- only the head changes.
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
    Gives the model a mechanism to express "I care about dimension k"
    differently for different market states -- the gate weights evolve
    freely during training.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate)


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
    """

    def __init__(
        self,
        in_channels: int = 5,
        patch_size: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        self.norm = RobustNorm()

        # Patch stem: flatten patch -> project to d_model
        self.stem = nn.Linear(in_channels * patch_size, d_model)

        # Learnable [CLS] token (aggregates global context)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings (no fixed sinusoid -- let model decide)
        # max 256 patches should be more than enough for any window size
        self.pos_emb = nn.Embedding(257, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

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
            x = F.pad(x, (0, 0, 0, pad))        # pad time dimension
        T_pad = x.shape[1]
        n_patches = T_pad // self.patch_size

        # Reshape into patches
        x = x.reshape(B, n_patches, self.patch_size * C)  # (B, n_patches, P*C)
        x = self.stem(x)                                    # (B, n_patches, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                     # (B, 1+n_patches, d_model)

        # Positional embeddings
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.pos_emb(positions)

        x = self.transformer(x)                            # (B, 1+n_patches, d_model)
        x = x[:, 0]                                        # CLS token = global context
        x = self.gate(x)
        x = self.out_norm(x)
        return x                                           # (B, d_model)
