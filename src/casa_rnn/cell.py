"""CASA-RNN Cell: Complex-valued, regime-aware recurrent cell.

Changelog:
  v0.6.2 - Add dedicated scalar regime_head: Linear(hidden*2+1, 1) -> Sigmoid
           The old regime_prob.mean(dim=-1) over 128 sigmoid outputs converged
           to 0.5 by law of large numbers regardless of input, causing Reg=0.50
           to be stuck across all training versions.  The new regime_head is a
           single scalar gate trained end-to-end to detect structural breaks,
           giving genuine dynamic range to the regime signal and unblocking
           DA/NE neuromodulator variation.
"""
import torch
import torch.nn as nn
import math


class CASARNNCell(nn.Module):
    """
    CASA-RNN Cell with:
    - Dual-speed hidden state (slow + fast)
    - Complex-valued state for periodicity encoding
    - Active regime gate that resets slow memory on structural breaks
    - v0.6.2: dedicated scalar regime_head for unambiguous regime signal
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size

        # Fast state update (GRU-like)
        self.W_fast = nn.Linear(input_size + hidden_size, hidden_size * 3)

        # Slow state update
        self.W_slow = nn.Linear(input_size + hidden_size, hidden_size * 2)

        # Regime gate: controls slow-state mixing (vector, unchanged role)
        self.regime_gate = nn.Linear(hidden_size * 2 + 1, hidden_size)

        # v0.6.2: scalar regime head -- single probability with real dynamic range
        # Input: [h_slow_mean, h_fast_mean, vol_indicator] -> 1 scalar
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        # Complex phase update
        self.phase_proj = nn.Linear(input_size, hidden_size)

        # Predictive coding
        self.predictor = nn.Linear(hidden_size, hidden_size)

        self.dropout         = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_norm_fast = nn.LayerNorm(hidden_size)
        self.layer_norm_slow = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor,
        vol_indicator: torch.Tensor = None,
    ):
        batch  = x.size(0)
        device = x.device

        if vol_indicator is None:
            vol_indicator = torch.zeros(batch, 1, device=device)

        # --- Fast state update (GRU-style) ---
        combined_fast = torch.cat([x, h_fast], dim=-1)
        gates = self.W_fast(combined_fast)
        z, r, n = gates.chunk(3, dim=-1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        n = torch.tanh(n + r * h_fast)
        h_fast_new = (1 - z) * h_fast + z * n
        h_fast_new = self.layer_norm_fast(h_fast_new)

        # --- Complex-valued phase update ---
        delta_phase = torch.tanh(self.phase_proj(x)) * math.pi
        h_fast_new  = torch.cos(delta_phase) * h_fast_new + torch.sin(delta_phase) * h_slow

        # --- Regime gate input (shared by both gate and head) ---
        regime_input = torch.cat([h_slow, h_fast_new, vol_indicator], dim=-1)

        # Vector gate: controls slow-state mixing (unchanged role)
        regime_prob_vec = torch.sigmoid(self.regime_gate(regime_input))  # (B, H)

        # v0.6.2: scalar regime head -- genuine dynamic range
        regime_scalar = self.regime_head(regime_input).squeeze(-1)  # (B,)

        # --- Slow state update ---
        combined_slow    = torch.cat([x, h_slow], dim=-1)
        slow_gates       = self.W_slow(combined_slow)
        z_s, n_s         = slow_gates.chunk(2, dim=-1)
        z_s              = torch.sigmoid(z_s)
        n_s              = torch.tanh(n_s)
        h_slow_candidate = (1 - z_s) * h_slow + z_s * n_s

        # Mix using vector gate (unchanged)
        h_slow_new = (1 - regime_prob_vec) * h_slow_candidate + regime_prob_vec * h_fast_new
        h_slow_new = self.layer_norm_slow(h_slow_new)

        # --- Predictive coding error ---
        h_fast_pred = self.predictor(h_slow_new)
        pred_error  = (h_fast_new - h_fast_pred).pow(2).mean(dim=-1, keepdim=True)

        h_fast_new = self.dropout(h_fast_new)
        h_slow_new = self.dropout(h_slow_new)

        # Return regime_scalar (B,) instead of mean of 128-dim sigmoid
        return h_slow_new, h_fast_new, regime_scalar, pred_error
