"""
PatternExtractor — 1D CNN front-end for local shape recognition.

Mimics how a trader visually reads chart patterns (head-and-shoulders,
double-bottom, etc.) before passing context to the RNN.

Changelog:
  v0.5 — initial implementation
"""
import torch
import torch.nn as nn


class PatternExtractor(nn.Module):
    """
    Lightweight 1D CNN that extracts local temporal patterns from raw input
    features.  Output has the same time-steps and feature dimension as the
    input so it can be added as a residual prior to the RNN cells.

    Architecture
    ------------
    Conv1d(in, mid, k=5) -> GELU -> Conv1d(mid, mid, k=3) -> GELU
    -> Conv1d(mid, in, k=1) -> LayerNorm  (+residual)

    The three kernel sizes capture:
      k=5  short-term momentum / micro-patterns
      k=3  medium-term shape confirmation
      k=1  channel mixing / projection back to input_size
    """

    def __init__(self, input_size: int, mid_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, mid_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(mid_channels, input_size, kernel_size=1),
        )
        self.norm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, F)  raw feature sequence

        Returns
        -------
        out : (B, T, F)  pattern-enriched sequence (residual added)
        """
        # Conv1d expects (B, F, T)
        x_t = x.transpose(1, 2)
        out = self.conv(x_t).transpose(1, 2)   # back to (B, T, F)
        return self.norm(x + out)
