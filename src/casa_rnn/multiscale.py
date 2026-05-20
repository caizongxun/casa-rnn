"""Multi-Scale CASA-RNN: three parallel temporal paths fused with attention."""
import torch
import torch.nn as nn
from typing import Optional, Dict
from .cell import CASARNNCell
from .memory import MemoryBank
from .heads import UncertaintyGatedHead
from .neuro_modules import (
    NeuromodulatorGating,
    ThalamicAttention,
    PrefrontalWorkingMemory,
    MetaLearningStrategyBank,
)


class MultiScaleCASARNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        dropout: float = 0.1,
        use_memory: bool = True,
        memory_slots: int = 32,
        regime_reset_threshold: float = 0.15,
        hidden_decay_rate: float = 0.3,
        use_bio: bool = True,
        context_size: int = 3,
    ):
        super().__init__()
        self.hidden_size            = hidden_size
        self.use_memory             = use_memory
        self.use_bio                = use_bio
        self.regime_reset_threshold = regime_reset_threshold
        self.hidden_decay_rate      = hidden_decay_rate

        self.cell_fast = CASARNNCell(input_size, hidden_size, dropout=dropout)
        self.cell_mid  = CASARNNCell(input_size, hidden_size, dropout=dropout)
        self.cell_slow = CASARNNCell(input_size, hidden_size, dropout=dropout)

        self.scale_attn = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

        if use_bio:
            self.thal_attn     = ThalamicAttention(hidden_size, context_size)
            self.neuro_gate    = NeuromodulatorGating(hidden_size, context_size)
            self.pfc_wm        = PrefrontalWorkingMemory(hidden_size, context_dim=hidden_size // 4)
            self.strategy_bank = MetaLearningStrategyBank(hidden_size, context_size, num_strategies=5)

        if use_memory:
            self.memory = MemoryBank(hidden_size, num_slots=memory_slots, write_threshold=0.12)

        self.head    = UncertaintyGatedHead(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.output_size = output_size
        self.register_buffer("regime_ema", torch.tensor(0.5))

    def _init_hidden(self, batch: int, device: torch.device) -> Dict:
        z = lambda: torch.zeros(batch, self.hidden_size, device=device)
        return {'fast': (z(), z()), 'mid': (z(), z()), 'slow': (z(), z())}

    def _soft_decay(self, h: torch.Tensor, delta: float) -> torch.Tensor:
        decay = min(delta, 1.0) * self.hidden_decay_rate
        return h * (1.0 - decay)

    def forward(
        self,
        x: torch.Tensor,
        vol_indicator: Optional[torch.Tensor] = None,
        init_hidden: Optional[Dict] = None,
    ):
        batch, seq_len, _ = x.shape
        device = x.device
        hidden = init_hidden if init_hidden is not None else self._init_hidden(batch, device)

        means, stds = [], []
        all_regime, all_scale_weights = [], []
        all_neuro_scalar = {"dopamine": [], "acetylcholine": [], "norepinephrine": [], "serotonin": []}
        # accumulate full tensors for bio constraint loss
        all_da, all_ne, all_vov, all_rpe_t = [], [], [], []
        all_strategy_w = []
        total_pred_err = torch.zeros(1, device=device)

        prev_log_vol = None
        for t in range(seq_len):
            xt = x[:, t, :]
            vt = vol_indicator[:, t, :] if vol_indicator is not None else None

            hs_f, hf_f, reg_f, pe_f = self.cell_fast(xt, *hidden['fast'], vt)
            hs_m, hf_m, reg_m, pe_m = self.cell_mid (xt, *hidden['mid'],  vt)
            hs_s, hf_s, reg_s, pe_s = self.cell_slow(xt, *hidden['slow'], vt)

            regime_mean = (reg_f + reg_m + reg_s) / 3.0
            r_now       = regime_mean.mean().detach()
            delta = (r_now - self.regime_ema).abs().item()
            self.regime_ema = 0.9 * self.regime_ema + 0.1 * r_now

            if delta > self.regime_reset_threshold:
                hf_f = self._soft_decay(hf_f, delta)
                hf_m = self._soft_decay(hf_m, delta)

            hidden['fast'] = (hs_f, hf_f)
            hidden['mid']  = (hs_m, hf_m)
            hidden['slow'] = (hs_s, hf_s)

            total_pred_err = total_pred_err + pe_f.mean() + pe_m.mean() + pe_s.mean()

            concat  = torch.cat([hf_f, hf_m, hf_s], dim=-1)
            weights = torch.softmax(self.scale_attn(concat), dim=-1)
            fused   = (weights[:, 0:1] * hf_f + weights[:, 1:2] * hf_m + weights[:, 2:3] * hf_s)
            fused   = self.dropout(fused)

            if self.use_bio:
                regime_t = regime_mean.unsqueeze(-1)           # (B,1)
                vol_t    = vt if vt is not None else torch.zeros_like(regime_t)
                log_vol  = torch.log1p(vol_t.abs())            # (B,1)
                if prev_log_vol is None:
                    vov = torch.zeros_like(log_vol)
                else:
                    vov = (log_vol - prev_log_vol).abs()
                prev_log_vol = log_vol.detach()

                ctx = torch.cat([regime_t, log_vol, vov], dim=-1).unsqueeze(1)  # (B,1,3)
                h_t = fused.unsqueeze(1)                                         # (B,1,H)
                rpe_t = ((pe_f + pe_m + pe_s) / 3.0).unsqueeze(1)               # (B,1,1)

                h_t = self.thal_attn(h_t, ctx)
                h_t, neuro_d = self.neuro_gate(h_t, ctx)
                h_t = self.pfc_wm(h_t, rpe=rpe_t)
                h_t, strategy_info = self.strategy_bank(h_t, ctx, rpe_t)

                fused = h_t.squeeze(1)

                for k in all_neuro_scalar:
                    all_neuro_scalar[k].append(neuro_d[k].mean().item())

                all_da.append(neuro_d["dopamine"])           # (B,1,1)
                all_ne.append(neuro_d["norepinephrine"])     # (B,1,1)
                all_vov.append(vov.unsqueeze(1))             # (B,1,1)
                all_rpe_t.append(rpe_t)                      # (B,1,1)
                all_strategy_w.append(strategy_info["strategy_w_tensor"])  # (B,1,S)

            if self.use_memory:
                self.memory.write(fused, regime_mean)
                fused = self.memory.read(fused, regime_mean)

            mean, std = self.head(fused)
            means.append(mean.unsqueeze(1))
            stds.append(std.unsqueeze(1))

            regime_stack = torch.stack([reg_f, reg_m, reg_s], dim=1)
            all_regime.append(regime_stack.unsqueeze(1))
            all_scale_weights.append(weights.unsqueeze(1))

        means   = torch.cat(means, dim=1)
        stds    = torch.cat(stds, dim=1)
        regime  = torch.cat(all_regime, dim=1)
        scale_w = torch.cat(all_scale_weights, dim=1)

        neuro_summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in all_neuro_scalar.items()}

        # concat tensors across time for bio constraint and strategy entropy
        if self.use_bio and all_da:
            neuro_tensors = {
                "dopamine":       torch.cat(all_da,  dim=1),  # (B,T,1)
                "norepinephrine": torch.cat(all_ne,  dim=1),
            }
            vov_full  = torch.cat(all_vov,   dim=1)  # (B,T,1)
            rpe_full  = torch.cat(all_rpe_t, dim=1)  # (B,T,1)
            strat_raw = torch.cat(all_strategy_w, dim=1)  # (B,T,S)
            sw_mean   = strat_raw.mean(dim=(0, 1)).detach().cpu().tolist()
            dom       = int(strat_raw.mean(dim=(0, 1)).argmax().item())
        else:
            neuro_tensors = {}
            vov_full = rpe_full = strat_raw = None
            sw_mean  = []
            dom      = -1

        extra = {
            "pred_coding_loss": total_pred_err / (seq_len * 3),
            "regime_probs":     regime,
            "scale_weights":    scale_w,
            "final_hidden":     hidden,
            "neuro":            neuro_summary,
            "neuro_tensors":    neuro_tensors,
            "vol_of_vol":       vov_full,
            "rpe_tensor":       rpe_full,
            "strategy_w_raw":   strat_raw,
            "strategy_weights": sw_mean,
            "dominant_strategy": dom,
        }
        return means, stds, extra
