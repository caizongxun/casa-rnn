"""Multi-Scale CASA-RNN: three parallel temporal paths fused with attention."""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from .cell import CASARNNCell
from .memory import MemoryBank
from .heads import UncertaintyGatedHead


class MultiScaleCASARNN(nn.Module):
    """
    Three parallel CASARNNCells operating at different temporal scales,
    fused with a learned attention mechanism.

    Scales:
        fast  -- tick / 1-min level dynamics
        mid   -- hourly / swing dynamics
        slow  -- daily / trend dynamics

    After fusion, output goes through UncertaintyGatedHead (mean + std)
    and an optional MemoryBank for long-range regime memory.

    Args:
        input_size:     number of input features
        hidden_size:    hidden dimension per scale
        output_size:    prediction output dimension
        dropout:        dropout between scale outputs
        use_memory:     attach MemoryBank for regime memory
        memory_slots:   number of memory slots if use_memory=True
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        dropout: float = 0.1,
        use_memory: bool = True,
        memory_slots: int = 32,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.cell_fast = CASARNNCell(input_size, hidden_size, dropout=dropout)
        self.cell_mid  = CASARNNCell(input_size, hidden_size, dropout=dropout)
        self.cell_slow = CASARNNCell(input_size, hidden_size, dropout=dropout)

        # Attention-based scale fusion
        self.scale_attn = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

        # Optional memory bank
        self.use_memory = use_memory
        if use_memory:
            self.memory = MemoryBank(hidden_size, num_slots=memory_slots)

        # Uncertainty head
        self.head = UncertaintyGatedHead(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size

    def _init_hidden(self, batch: int, device: torch.device) -> Dict:
        z = lambda: torch.zeros(batch, self.hidden_size, device=device)
        return {
            'fast': (z(), z()),
            'mid':  (z(), z()),
            'slow': (z(), z()),
        }

    def forward(
        self,
        x: torch.Tensor,                              # (batch, seq_len, input_size)
        vol_indicator: Optional[torch.Tensor] = None, # (batch, seq_len, 1)
        init_hidden: Optional[Dict] = None,
    ):
        batch, seq_len, _ = x.shape
        device = x.device

        hidden = init_hidden if init_hidden is not None else self._init_hidden(batch, device)

        means, stds = [], []
        all_regime, all_scale_weights = [], []
        total_pred_err = torch.zeros(1, device=device)

        for t in range(seq_len):
            xt  = x[:, t, :]
            vt  = vol_indicator[:, t, :] if vol_indicator is not None else None

            # --- Three scale updates ---
            hs_f, hf_f, reg_f, pe_f = self.cell_fast(xt, *hidden['fast'], vt)
            hs_m, hf_m, reg_m, pe_m = self.cell_mid (xt, *hidden['mid'],  vt)
            hs_s, hf_s, reg_s, pe_s = self.cell_slow(xt, *hidden['slow'], vt)

            hidden['fast'] = (hs_f, hf_f)
            hidden['mid']  = (hs_m, hf_m)
            hidden['slow'] = (hs_s, hf_s)

            total_pred_err = total_pred_err + pe_f.mean() + pe_m.mean() + pe_s.mean()

            # --- Attention fusion ---
            concat = torch.cat([hf_f, hf_m, hf_s], dim=-1)       # (B, H*3)
            weights = torch.softmax(self.scale_attn(concat), dim=-1)  # (B, 3)
            fused = (weights[:, 0:1] * hf_f +
                     weights[:, 1:2] * hf_m +
                     weights[:, 2:3] * hf_s)
            fused = self.dropout(fused)

            # --- Memory bank read/write ---
            if self.use_memory:
                regime_mean = (reg_f + reg_m + reg_s) / 3.0
                self.memory.write(fused, regime_mean)
                fused = self.memory.read(fused)

            # --- Uncertainty head ---
            mean, std = self.head(fused)
            means.append(mean.unsqueeze(1))
            stds.append(std.unsqueeze(1))

            regime_stack = torch.stack([reg_f, reg_m, reg_s], dim=1)  # (B, 3)
            all_regime.append(regime_stack.unsqueeze(1))
            all_scale_weights.append(weights.unsqueeze(1))

        means  = torch.cat(means,  dim=1)            # (B, T, output_size)
        stds   = torch.cat(stds,   dim=1)            # (B, T, output_size)
        regime = torch.cat(all_regime, dim=1)        # (B, T, 3)
        scale_w = torch.cat(all_scale_weights, dim=1) # (B, T, 3)

        extra = {
            "pred_coding_loss": total_pred_err / (seq_len * 3),
            "regime_probs":     regime,
            "scale_weights":    scale_w,
            "final_hidden":     hidden,
        }

        return means, stds, extra
