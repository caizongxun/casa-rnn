"""CASA-RNN Cell: Complex-valued, regime-aware recurrent cell.

Changelog:
  v0.6.2 - dedicated scalar regime_head (Linear->Tanh->Linear->Sigmoid)
  v0.7   - self-supervised regime learning:
           regime_head is trained with pred_error as soft target.
           High pred_error (model surprised) -> target=1.0 (regime change)
           Low  pred_error (model stable)    -> target=0.0 (stable regime)
           pe_target = sigmoid(pred_error * pe_scale) normalises to (0,1)
           regime_self_loss = BCE(regime_scalar, pe_target) added to return
           regime_entropy_bonus pushes regime away from 0.5 collapse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CASARNNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        regime_loss_weight: float = 0.5,
        pe_scale: float = 10.0,
    ):
        super().__init__()
        self.input_size         = input_size
        self.hidden_size        = hidden_size
        self.regime_loss_weight = regime_loss_weight
        self.pe_scale           = pe_scale

        self.W_fast = nn.Linear(input_size + hidden_size, hidden_size * 3)
        self.W_slow = nn.Linear(input_size + hidden_size, hidden_size * 2)
        self.regime_gate = nn.Linear(hidden_size * 2 + 1, hidden_size)
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        self.phase_proj      = nn.Linear(input_size, hidden_size)
        self.predictor       = nn.Linear(hidden_size, hidden_size)
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

        # --- Fast GRU ---
        combined_fast = torch.cat([x, h_fast], dim=-1)
        gates = self.W_fast(combined_fast)
        z, r, n = gates.chunk(3, dim=-1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        n = torch.tanh(n + r * h_fast)
        h_fast_new = (1 - z) * h_fast + z * n
        h_fast_new = self.layer_norm_fast(h_fast_new)

        # --- Complex phase ---
        delta_phase = torch.tanh(self.phase_proj(x)) * math.pi
        h_fast_new  = torch.cos(delta_phase) * h_fast_new + torch.sin(delta_phase) * h_slow

        # --- Regime gate input ---
        regime_input = torch.cat([h_slow, h_fast_new, vol_indicator], dim=-1)

        # Vector gate for slow-state mixing
        regime_prob_vec = torch.sigmoid(self.regime_gate(regime_input))

        # Scalar regime probability (real dynamic range)
        regime_scalar = self.regime_head(regime_input).squeeze(-1)  # (B,)

        # --- Slow state ---
        combined_slow    = torch.cat([x, h_slow], dim=-1)
        z_s, n_s         = self.W_slow(combined_slow).chunk(2, dim=-1)
        z_s              = torch.sigmoid(z_s)
        n_s              = torch.tanh(n_s)
        h_slow_candidate = (1 - z_s) * h_slow + z_s * n_s
        h_slow_new = (1 - regime_prob_vec) * h_slow_candidate + regime_prob_vec * h_fast_new
        h_slow_new = self.layer_norm_slow(h_slow_new)

        # --- Predictive coding error ---
        h_fast_pred = self.predictor(h_slow_new)
        pred_error  = (h_fast_new - h_fast_pred).pow(2).mean(dim=-1, keepdim=True)  # (B,1)

        # --- v0.7: self-supervised regime loss ---
        # Map pred_error to (0,1): high error -> target 1.0 (regime change)
        pe_target = torch.sigmoid(pred_error.detach() * self.pe_scale)  # (B,1)
        regime_self_loss = F.binary_cross_entropy(
            regime_scalar.unsqueeze(-1).clamp(1e-6, 1-1e-6),
            pe_target,
            reduction="mean",
        )
        # Entropy bonus: -H(p) penalises staying at 0.5
        # We MINIMISE -H so regime_head is pushed to be decisive
        p = regime_scalar.clamp(1e-6, 1-1e-6)
        entropy_bonus = (p * p.log() + (1-p) * (1-p).log()).mean()  # negative entropy
        regime_loss = self.regime_loss_weight * regime_self_loss + 0.1 * entropy_bonus

        h_fast_new = self.dropout(h_fast_new)
        h_slow_new = self.dropout(h_slow_new)

        return h_slow_new, h_fast_new, regime_scalar, pred_error, regime_loss
