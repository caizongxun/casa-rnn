"""
Multi-Scale CASA-RNN — full bio + topology pipeline.

Changelog:
  - Integrated DangerSignalDetector.selective_reset() on h_slow (Direction B)
  - SoftModuleRouter.regime_consistency_loss() added to extra dict for
    optional use in training loop (Direction A)
  - ContrastiveStateRegularizer added, hidden states collected for Direction C
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from .cell import CASARNNCell
from .memory import MemoryBank
from .heads import UncertaintyGatedHead
from .neuro_modules import (
    NeuromodulatorGating, ThalamicAttention, PrefrontalWorkingMemory,
    MetaLearningStrategyBank, RegimeTransitionDetector,
    CerebellarForwardModel, HomeostaticGainControl, SoftModuleRouter,
    ContrastiveStateRegularizer,
    MODULE_NAMES,
)
from .cpg import CPGEncoder
from .astrocyte import AstrocyteModulator
from .danger import DangerSignalDetector
from .tda_features import TDAFeatureExtractor


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
        cpg_periods: List[float] = (390.0, 1950.0, 8580.0),
        tda_window: int = 30,
    ):
        super().__init__()
        self.hidden_size            = hidden_size
        self.use_memory             = use_memory
        self.use_bio                = use_bio
        self.regime_reset_threshold = regime_reset_threshold
        self.hidden_decay_rate      = hidden_decay_rate
        self.input_size             = input_size

        from .cpg import CPGEncoder
        self.cpg         = CPGEncoder(cpg_periods)
        cpg_dim          = self.cpg.out_dim
        rnn_input_size   = input_size + cpg_dim

        self.cell_fast = CASARNNCell(rnn_input_size, hidden_size, dropout=dropout)
        self.cell_mid  = CASARNNCell(rnn_input_size, hidden_size, dropout=dropout)
        self.cell_slow = CASARNNCell(rnn_input_size, hidden_size, dropout=dropout)

        self.scale_attn = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

        if use_bio:
            self.thal_attn      = ThalamicAttention(hidden_size, context_size)
            self.neuro_gate     = NeuromodulatorGating(hidden_size, context_size)
            self.pfc_wm         = PrefrontalWorkingMemory(hidden_size, context_dim=hidden_size // 4)
            self.strategy_bank  = MetaLearningStrategyBank(hidden_size, context_size)
            self.transition_det = RegimeTransitionDetector(hidden_size)
            self.cerebellum     = CerebellarForwardModel(hidden_size, rnn_input_size, proj_size=hidden_size // 2)
            self.homeostasis    = HomeostaticGainControl()
            self.astrocyte      = AstrocyteModulator(hidden_size)
            self.danger_det     = DangerSignalDetector(hidden_size)
            self.tda            = TDAFeatureExtractor(input_size, window=tda_window)
            self.tda_proj       = nn.Linear(3, hidden_size)
            self.router         = SoftModuleRouter(hidden_size, num_modules=len(MODULE_NAMES))
            self.contrastive_reg = ContrastiveStateRegularizer(temperature=0.5)

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
        transition_label: float = 0.0,
        t_offset: int = 0,
        router_entropy_weight: float = 0.02,
    ):
        batch, seq_len, raw_feat = x.shape
        device = x.device
        hidden = init_hidden if init_hidden is not None else self._init_hidden(batch, device)

        cpg_feats = self.cpg(seq_len, batch, device, t_offset=t_offset)
        x_aug     = torch.cat([x, cpg_feats], dim=-1)

        tda_window    = min(seq_len, self.tda.window)
        tda_feats_seq = self.tda(x[:, :tda_window, :])
        tda_h_base    = self.tda_proj(tda_feats_seq)

        if self.use_bio:
            self.astrocyte.reset(batch, device)

        means, stds = [], []
        all_regime, all_scale_weights = [], []
        all_neuro_scalar = {k: [] for k in ["dopamine","acetylcholine","norepinephrine","serotonin"]}
        all_da, all_ne, all_sht = [], [], []
        all_vov, all_rpe_t = [], []
        all_strategy_w, all_trans_prob = [], []
        all_cereb_err   = []
        all_router_w    = []
        all_danger      = []
        all_fused       = []   # collected for contrastive loss
        all_regime_t    = []   # collected for contrastive loss
        total_pred_err  = torch.zeros(1, device=device)

        prev_log_vol = None
        prev_fused   = None

        slow_step_counter = 0
        slow_update_every = 1

        for t in range(seq_len):
            xt     = x[:, t, :]
            xt_aug = x_aug[:, t, :]
            vt     = vol_indicator[:, t, :] if vol_indicator is not None else None

            hs_f, hf_f, reg_f, pe_f = self.cell_fast(xt_aug, *hidden['fast'], vt)
            slow_step_counter += 1
            if slow_step_counter >= slow_update_every:
                hs_m, hf_m, reg_m, pe_m = self.cell_mid (xt_aug, *hidden['mid'],  vt)
                hs_s, hf_s, reg_s, pe_s = self.cell_slow(xt_aug, *hidden['slow'], vt)
                slow_step_counter = 0
            else:
                hs_m, hf_m = hidden['mid'];  reg_m = (hidden['mid'][1] * 0).mean(-1) + 0.5; pe_m = torch.zeros_like(pe_f)
                hs_s, hf_s = hidden['slow']; reg_s = reg_m; pe_s = pe_m

            regime_mean = (reg_f + reg_m + reg_s) / 3.0
            r_now  = regime_mean.mean().detach()
            delta  = (r_now - self.regime_ema).abs().item()
            self.regime_ema = 0.9 * self.regime_ema + 0.1 * r_now

            if delta > self.regime_reset_threshold:
                hf_f = self._soft_decay(hf_f, delta)
                hf_m = self._soft_decay(hf_m, delta)
                # Direction B: selective reset on h_slow via DangerSignalDetector
                # (applied below after danger score is computed)

            hidden['fast'] = (hs_f, hf_f)
            hidden['mid']  = (hs_m, hf_m)
            hidden['slow'] = (hs_s, hf_s)

            total_pred_err = total_pred_err + pe_f.mean() + pe_m.mean() + pe_s.mean()

            concat  = torch.cat([hf_f, hf_m, hf_s], dim=-1)
            weights = torch.softmax(self.scale_attn(concat), dim=-1)
            fused   = (weights[:, 0:1] * hf_f + weights[:, 1:2] * hf_m + weights[:, 2:3] * hf_s)
            fused   = self.dropout(fused)

            danger_score = torch.zeros(batch, 1, device=device)

            if self.use_bio:
                regime_t = regime_mean.unsqueeze(-1)
                vol_t    = vt if vt is not None else torch.zeros_like(regime_t)
                log_vol  = torch.log1p(vol_t.abs())
                if prev_log_vol is None:
                    vov = torch.zeros_like(log_vol)
                else:
                    vov = (log_vol - prev_log_vol).abs()
                prev_log_vol = log_vol.detach()

                self.danger_det.update(fused)
                danger_score = self.danger_det(fused)

                # Direction B: selective state reset on h_slow
                # Only apply when regime is actually shifting (delta > threshold)
                if delta > self.regime_reset_threshold:
                    hs_s_reset = self.danger_det.selective_reset(hidden['slow'][0], danger_score)
                    hf_s_reset = self.danger_det.selective_reset(hidden['slow'][1], danger_score)
                    hidden['slow'] = (hs_s_reset, hf_s_reset)

                mean_danger = danger_score.mean().item()
                slow_update_every = max(1, int(3.0 / (1.0 + 3.0 * mean_danger)))

                fused = fused + 0.1 * tda_h_base

                router_weights = self.router(
                    h=fused, regime=regime_t, danger=danger_score, vov=vov,
                )
                all_router_w.append(router_weights)

                w = router_weights

                ctx   = torch.cat([regime_t, log_vol, vov], dim=-1).unsqueeze(1)
                h_t   = fused.unsqueeze(1)
                rpe_t = ((pe_f + pe_m + pe_s) / 3.0).unsqueeze(1)

                cereb_err_val = 0.0
                if prev_fused is not None:
                    h_pred_cereb = self.cerebellum(prev_fused, xt_aug)
                    cereb_err    = self.cerebellum.cerebellar_error(h_pred_cereb, fused)
                    w_cereb      = w[:, MODULE_NAMES.index("cerebellum")].unsqueeze(-1).unsqueeze(-1)
                    rpe_t        = rpe_t + w_cereb * 0.3 * cereb_err.unsqueeze(1)
                    cereb_err_val = cereb_err.detach().mean().item()
                    all_cereb_err.append(cereb_err_val)
                prev_fused = fused.detach()

                self.transition_det.update_queue(delta)

                h_thal = self.thal_attn(h_t, ctx)
                w_thal = w[:, MODULE_NAMES.index("thalamic")].unsqueeze(-1).unsqueeze(-1)
                h_t    = (1 - w_thal) * h_t + w_thal * h_thal

                h_neuro, neuro_d = self.neuro_gate(h_t, ctx)
                w_neuro = w[:, MODULE_NAMES.index("neuromod")].unsqueeze(-1).unsqueeze(-1)
                h_t     = (1 - w_neuro) * h_t + w_neuro * h_neuro

                ne_boosted   = self.danger_det.ne_boost(danger_score, neuro_d["norepinephrine_raw"].squeeze(1))
                w_danger     = w[:, MODULE_NAMES.index("danger")].unsqueeze(-1)
                ne_effective = (1 - w_danger) * neuro_d["norepinephrine_raw"].squeeze(1) + w_danger * ne_boosted

                h_ast  = self.astrocyte(h_t.squeeze(1)).unsqueeze(1)
                w_ast  = w[:, MODULE_NAMES.index("astrocyte")].unsqueeze(-1).unsqueeze(-1)
                h_t    = (1 - w_ast) * h_t + w_ast * h_ast

                h_pfc  = self.pfc_wm(h_t, rpe=rpe_t)
                w_pfc  = w[:, MODULE_NAMES.index("pfc_wm")].unsqueeze(-1).unsqueeze(-1)
                h_t    = (1 - w_pfc) * h_t + w_pfc * h_pfc

                h_strat, strategy_info = self.strategy_bank(h_t, ctx, rpe_t)
                w_strat = w[:, MODULE_NAMES.index("strategy")].unsqueeze(-1).unsqueeze(-1)
                h_t     = (1 - w_strat) * h_t + w_strat * h_strat

                trans_prob = self.transition_det(h_t)
                all_trans_prob.append(trans_prob)

                w_tda  = w[:, MODULE_NAMES.index("tda")].unsqueeze(-1)
                fused  = h_t.squeeze(1) + w_tda * tda_h_base * 0.1
                h_t    = fused.unsqueeze(1)

                self.homeostasis.update(
                    da_mean=neuro_d["dopamine"].mean().item(),
                    ne_mean=ne_effective.mean().item(),
                    sht_mean=neuro_d["serotonin"].mean().item(),
                )

                fused = h_t.squeeze(1)

                # Collect hidden states for contrastive loss
                all_fused.append(fused.unsqueeze(1))
                all_regime_t.append(regime_t.unsqueeze(1))

                for k in all_neuro_scalar:
                    all_neuro_scalar[k].append(neuro_d[k].mean().item())
                all_da.append(neuro_d["dopamine_raw"])
                all_ne.append(neuro_d["norepinephrine_raw"])
                all_sht.append(neuro_d["serotonin_raw"])
                all_vov.append(vov.unsqueeze(1))
                all_rpe_t.append(rpe_t)
                all_strategy_w.append(strategy_info["strategy_w_tensor"])
                all_danger.append(danger_score)

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

        if all_router_w:
            router_w_full = torch.stack(all_router_w, dim=1)
            router_w_mean = router_w_full.mean(dim=(0, 1)).detach().cpu().tolist()
        else:
            router_w_full = None
            router_w_mean = []

        # Compute regime_consistency_loss for optional use in training loop
        regime_consist_loss = torch.tensor(0.0, device=device)
        if self.use_bio and router_w_full is not None and all_regime_t:
            regime_cat = torch.cat(all_regime_t, dim=1)  # (B, T, 1)
            regime_consist_loss = self.router.regime_consistency_loss(router_w_full, regime_cat)

        # Contrastive state regularizer loss
        contrastive_loss = torch.tensor(0.0, device=device)
        if self.use_bio and all_fused and all_regime_t:
            fused_cat  = torch.cat(all_fused, dim=1)    # (B, T, H)
            regime_cat = torch.cat(all_regime_t, dim=1) # (B, T, 1)
            contrastive_loss = self.contrastive_reg(fused_cat, regime_cat)

        if self.use_bio and all_da:
            neuro_tensors = {
                "dopamine":       torch.cat(all_da,  dim=1),
                "norepinephrine": torch.cat(all_ne,  dim=1),
                "serotonin":      torch.cat(all_sht, dim=1),
            }
            vov_full    = torch.cat(all_vov,        dim=1)
            rpe_full    = torch.cat(all_rpe_t,      dim=1)
            strat_raw   = torch.cat(all_strategy_w, dim=1)
            trans_stack = torch.cat(all_trans_prob, dim=1) if all_trans_prob else None
            sw_mean     = strat_raw.mean(dim=(0, 1)).detach().cpu().tolist()
            dom         = int(strat_raw.mean(dim=(0, 1)).argmax().item())
            avg_cereb   = sum(all_cereb_err) / len(all_cereb_err) if all_cereb_err else 0.0
            danger_full = torch.cat(all_danger, dim=0).mean().item() if all_danger else 0.0
        else:
            neuro_tensors = {}
            vov_full = rpe_full = strat_raw = trans_stack = None
            sw_mean  = []; dom = -1; avg_cereb = 0.0; danger_full = 0.0

        extra = {
            "pred_coding_loss":       total_pred_err / (seq_len * 3),
            "regime_probs":           regime,
            "scale_weights":          scale_w,
            "final_hidden":           hidden,
            "neuro":                  neuro_summary,
            "neuro_tensors":          neuro_tensors,
            "vol_of_vol":             vov_full,
            "rpe_tensor":             rpe_full,
            "strategy_w_raw":         strat_raw,
            "strategy_weights":       sw_mean,
            "dominant_strategy":      dom,
            "trans_prob":             trans_stack,
            "cereb_err":              avg_cereb,
            "transition_label":       transition_label,
            "router_weights":         router_w_full,
            "router_w_mean":          router_w_mean,
            "danger_score":           danger_full,
            "cpg_periods":            [o.period.item() for o in self.cpg.oscillators],
            "astrocyte_alpha":        self.astrocyte.alpha.item() if self.use_bio else None,
            "astrocyte_beta":         self.astrocyte.beta.item()  if self.use_bio else None,
            "router_temperature":     self.router.temperature.item() if self.use_bio else None,
            # New: auxiliary losses exposed for optional use in training loop
            "regime_consistency_loss": regime_consist_loss,
            "contrastive_loss":        contrastive_loss,
        }
        return means, stds, extra
