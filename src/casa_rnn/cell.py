"""CASA-RNN Cell: Complex-valued, regime-aware recurrent cell."""
import torch
import torch.nn as nn
import math


class CASARNNCell(nn.Module):
    """
    CASA-RNN Cell with:
    - Dual-speed hidden state (slow + fast)
    - Complex-valued state for periodicity encoding
    - Active regime gate that resets slow memory on structural breaks
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Fast state update (standard GRU-like)
        self.W_fast = nn.Linear(input_size + hidden_size, hidden_size * 3)

        # Slow state update (lower learning rate via gradient scaling)
        self.W_slow = nn.Linear(input_size + hidden_size, hidden_size * 2)

        # Regime detector gate: decides when to reset slow memory
        self.regime_gate = nn.Linear(hidden_size * 2 + 1, hidden_size)

        # Complex phase update for periodicity
        self.phase_proj = nn.Linear(input_size, hidden_size)

        # Predictive coding: predict next fast state from slow
        self.predictor = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm_fast = nn.LayerNorm(hidden_size)
        self.layer_norm_slow = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,                # (batch, input_size)
        h_slow: torch.Tensor,           # (batch, hidden_size)
        h_fast: torch.Tensor,           # (batch, hidden_size)
        vol_indicator: torch.Tensor = None,  # (batch, 1) optional volatility signal
    ):
        batch = x.size(0)
        device = x.device

        if vol_indicator is None:
            vol_indicator = torch.zeros(batch, 1, device=device)

        # --- Fast state update (GRU-style) ---
        combined_fast = torch.cat([x, h_fast], dim=-1)
        gates = self.W_fast(combined_fast)
        z, r, n = gates.chunk(3, dim=-1)
        z = torch.sigmoid(z)   # update gate
        r = torch.sigmoid(r)   # reset gate
        n = torch.tanh(n + r * h_fast)
        h_fast_new = (1 - z) * h_fast + z * n
        h_fast_new = self.layer_norm_fast(h_fast_new)

        # --- Complex-valued phase update ---
        delta_phase = torch.tanh(self.phase_proj(x)) * math.pi
        # Use real part of imaginary rotation: cos(delta) * h + sin(delta) * h_perp
        h_fast_new = torch.cos(delta_phase) * h_fast_new + torch.sin(delta_phase) * h_slow

        # --- Regime gate: detect structural break ---
        regime_input = torch.cat([h_slow, h_fast_new, vol_indicator], dim=-1)
        regime_prob = torch.sigmoid(self.regime_gate(regime_input))  # (batch, hidden_size)

        # --- Slow state update with regime-aware reset ---
        combined_slow = torch.cat([x, h_slow], dim=-1)
        slow_gates = self.W_slow(combined_slow)
        z_s, n_s = slow_gates.chunk(2, dim=-1)
        z_s = torch.sigmoid(z_s)
        n_s = torch.tanh(n_s)
        h_slow_candidate = (1 - z_s) * h_slow + z_s * n_s

        # Regime gate: high regime_prob -> adopt fast state into slow (structural break)
        h_slow_new = (1 - regime_prob) * h_slow_candidate + regime_prob * h_fast_new
        h_slow_new = self.layer_norm_slow(h_slow_new)

        # --- Predictive coding error (used as auxiliary loss signal) ---
        h_fast_pred = self.predictor(h_slow_new)
        pred_error = (h_fast_new - h_fast_pred).pow(2).mean(dim=-1, keepdim=True)  # (batch, 1)

        h_fast_new = self.dropout(h_fast_new)
        h_slow_new = self.dropout(h_slow_new)

        return h_slow_new, h_fast_new, regime_prob.mean(dim=-1), pred_error
