"""Regime Detector: standalone module for market regime classification."""
import torch
import torch.nn as nn


class RegimeDetector(nn.Module):
    """
    Lightweight regime detector.
    Outputs: 0 = calm/trending, 1 = volatile/crisis
    Can be used standalone or plugged into CASARNNCell.
    """

    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_size) -> regime_prob: (batch, 1)"""
        return self.net(x)
