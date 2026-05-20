"""
Multi-Scale CASA-RNN -- full bio + topology + pattern pipeline.

Changelog:
  v0.8 - ctx source: (regime,log_vol,vov) -> h_fused stats (mean,std,norm)
         Neuromodulators now respond to actual hidden-state dynamics.
       - Learnable hyperparameters optimised by gradient:
           log_regime_thresh, log_decay_rate, log_tda_w, log_cereb_w,
           log_regime_loss_w  (all exposed via properties with clamping)
  v0.8.1 - NaN guard on danger_score before slow_update_every int cast
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from .cell import CASARNNCell
from .memory import MemoryBank
from .heads import UncertaintyGatedHead
from .pattern import PatternExtractor
from .neuro_modules import (
    NeuromodulatorGating, ThalamicAttention, PrefrontalWorkingMemory,
    MetaLearningStrategyBank, RegimeTransitionDetector,
    CerebellarForwardModel, HomeostaticGainControl, SoftModuleRouter,
    ContrastiveStateRegularizer, LatentExpertBank,
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
        cnn_mid_channels: int = 64,
        num_latent_experts: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_memory  = use_memory
        self.use_bio     = use_bio
        self.input_size  = input_size

        # --- Learnable hyperparameters (v0.8) ---
        # Stored in log-space so they stay positive; properties expose clamped values.
        self.log_regime_thresh  = nn.Parameter(torch.tensor(math.log(regime_reset_threshold)))
        self.log_decay_rate     = nn.Parameter(torch.tensor(math.log(hidden_decay_rate)))
        self.log_tda_w          = nn.Parameter(torch.tensor(math.log(0.1)))   # tda mix weight
        self.log_cereb_w        = nn.Parameter(torch.tensor(math.log(0.3)))   # cerebellum rpe weight
        self.log_regime_loss_w  = nn.Parameter(torch.tensor(math.log(1.0)))   # regime self-loss weight

        self.pattern_extractor = PatternExtractor(input_size, mid_channels=cnn_mid_channels)
        self.cpg               = CPGEncoder(cpg_periods)
        cpg_dim                = self.cpg.out_dim
        rnn_input_size         = input_size + cpg_dim

        self.cell_fast = CASARNNCell(rnn_input_size, hidden_size, dropout=dropout)
        self.cell_mid  = CASARNNCell(rnn_input_size, hidden_size, dropout=dropout)
        self.cell_slow = CASARNNCell(rnn_input_size, hidden_size, dropout=dropout)

        self.scale_attn = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

        if use_bio:
            self.thal_attn       = ThalamicAttention(hidden_size, context_size)
            self.neuro_gate      = NeuromodulatorGating(hidden_size, context_size)
            self.pfc_wm          = PrefrontalWorkingMemory(hidden_size, context_dim=hidden_size // 4)
            self.strategy_bank   = MetaLearningStrategyBank(hidden_size, context_size)
            self.transition_det  = RegimeTransitionDetector(hidden_size)
            self.cerebellum      = CerebellarForwardModel(hidden_size, rnn_input_size, proj_size=hidden_size // 2)
            self.homeostasis     = HomeostaticGainControl()
            self.astrocyte       = AstrocyteModulator(hidden_size)
            self.danger_det      = DangerSignalDetector(hidden_size)
            self.tda             = TDAFeatureExtractor(input_size, window=tda_window)
            self.tda_proj        = nn.Linear(3, hidden_size)
            self.router          = SoftModuleRouter(hidden_size, num_modules=len(MODULE_NAMES))
            self.contrastive_reg = ContrastiveStateRegularizer(temperature=0.5)
            self.latent_experts  = LatentExpertBank(hidden_size, num_experts=num_latent_experts)

        if use_memory:
            self.memory = MemoryBank(hidden_size, num_slots=memory_slots, write_threshold=0.12)

        self.head        = UncertaintyGatedHead(hidden_size, output_size)
        self.dropout     = nn.Dropout(dropout)
        self.output_size = output_size
        self.register_buffer("regime_ema", torch.tensor(0.5))

    # --- properties expose learnable params with safe ranges ---
    @property
    def regime_reset_threshold(self) -> float:
        return self.log_regime_thresh.exp().clamp(0.01, 0.5).item()

    @property
    def hidden_decay_rate(self) -> float:
        return self.log_decay_rate.exp().clamp(0.0, 1.0).item()

    @property
    def tda_w(self) -> torch.Tensor:
        return self.log_tda_w.exp().clamp(0.0, 1.0)

    @property
    def cereb_w(self) -> torch.Tensor:
        return self.log_cereb_w.exp().clamp(0.0, 1.0)

    @property
    def regime_loss_w(self) -> torch.Tensor:
        return self.log_regime_loss_w.exp().clamp(0.1, 5.0)

    def _init_hidden(self, batch: int, device: torch.device) -> Dict:
        z = lambda: torch.zeros(batch, self.hidden_size, device=device)
        return {'fast': (z(), z()), 'mid': (z(), z()), 'slow': (z(), z())}

    def _soft_decay(self, h: torch.Tensor, delta: float) -> torch.Tensor:
        rate = self.log_decay_rate.exp().clamp(0.0, 1.0)
        return h * (1.0 - min(delta, 1.0) * rate)

    @staticmethod
    def _h_ctx(h: torch.Tensor) -> torch.Tensor:
        """v0.8: build ctx from h statistics so neuro_gate has real variance."""
        h_mean = h.mean(dim=-1, keepdim=True)                          # (B,1)
        h_std  = h.std(dim=-1, keepdim=True).clamp(min=1e-6)          # (B,1)
        h_norm = h.norm(dim=-1, keepdim=True) / (h.shape[-1] ** 0.5)  # (B,1)
        return torch.cat([h_mean, h_std, h_norm], dim=-1).unsqueeze(1) # (B,1,3)

    @staticmethod
    def _safe_scalar(t: torch.Tensor, fallback: float = 0.0) -> float:
        """Return tensor mean as float; fall back to `fallback` if NaN/Inf."""
        val = t.mean().item()
        return fallback if (math.isnan(val) or math.isinf(val)) else val

    def forward(
        self,
        x: torch.Tensor,
        vol_indicator: Optional[torch.Tensor] = None,
        init_hidden: Optional[Dict] = None,
        transition_label: float = 0.0,
        t_offset: int = 0,
        router_entropy_weight: float = 0.02,
    ):
        batch, seq_len, _ = x.shape
        device = x.device
        hidden = init_hidden if init_hidden is not None else self._init_hidden(batch, device)

        x         = self.pattern_extractor(x)
        cpg_feats = self.cpg(seq_len, batch, device, t_offset=t_offset)
        x_aug     = torch.cat([x, cpg_feats], dim=-1)

        tda_window    = min(seq_len, self.tda.window)
        tda_feats_seq = self.tda(x[:, :tda_window, :])
        tda_h_base    = self.tda_proj(tda_feats_seq)

        if self.use_bio:
            self.astrocyte.reset(batch, device)
            expert_states = self.latent_experts.reset(batch, device)

        means, stds = [], []
        all_regime, all_scale_weights = [], []
        all_neuro_scalar = {k: [] for k in ["dopamine","acetylcholine","norepinephrine","serotonin"]}
        all_da, all_ne, all_sht = [], [], []
        all_vov, all_rpe_t      = [], []
        all_strategy_w          = []
        all_trans_prob          = []
        all_cereb_err           = []
        all_router_w            = []
        all_danger              = []
        all_fused               = []
        all_regime_t            = []
        all_latent_w            = []
        total_pred_err          = torch.zeros(1, device=device)
        total_regime_loss       = torch.zeros(1, device=device)
        max_danger              = torch.zeros(1, device=device)

        prev_log_vol      = None
        prev_fused        = None
        slow_step_counter = 0
        slow_update_every = 1
        thresh            = self.regime_reset_threshold  # snapshot once per forward

        for t in range(seq_len):
            xt     = x[:, t, :]
            xt_aug = x_aug[:, t, :]
            vt     = vol_indicator[:, t, :] if vol_indicator is not None else None

            hs_f, hf_f, reg_scalar_f, pe_f, rl_f = self.cell_fast(xt_aug, *hidden['fast'], vt)
            slow_step_counter += 1
            if slow_step_counter >= slow_update_every:
                hs_m, hf_m, reg_scalar_m, pe_m, rl_m = self.cell_mid (xt_aug, *hidden['mid'],  vt)
                hs_s, hf_s, reg_scalar_s, pe_s, rl_s = self.cell_slow(xt_aug, *hidden['slow'], vt)
                slow_step_counter = 0
            else:
                hs_m, hf_m = hidden['mid']
                reg_scalar_m = torch.full((batch,), 0.5, device=device)
                pe_m = torch.zeros_like(pe_f)
                rl_m = torch.zeros(1, device=device)
                hs_s, hf_s = hidden['slow']
                reg_scalar_s = reg_scalar_m.clone()
                pe_s = pe_m.clone()
                rl_s = rl_m.clone()

            # v0.8: scale regime_loss by learnable weight
            total_regime_loss = total_regime_loss + self.regime_loss_w * (rl_f + rl_m + rl_s)

            regime_scalar_mean = (reg_scalar_f + reg_scalar_m + reg_scalar_s) / 3.0

            r_now = regime_scalar_mean.mean().detach()
            delta = (r_now - self.regime_ema).abs().item()
            self.regime_ema = 0.9 * self.regime_ema + 0.1 * r_now

            if delta > thresh:
                hf_f = self._soft_decay(hf_f, delta)
                hf_m = self._soft_decay(hf_m, delta)

            hidden['fast'] = (hs_f, hf_f)
            hidden['mid']  = (hs_m, hf_m)
            hidden['slow'] = (hs_s, hf_s)

            total_pred_err = total_pred_err + pe_f.mean() + pe_m.mean() + pe_s.mean()

            concat  = torch.cat([hf_f, hf_m, hf_s], dim=-1)
            weights = torch.softmax(self.scale_attn(concat), dim=-1)
            fused   = (weights[:, 0:1]*hf_f + weights[:, 1:2]*hf_m + weights[:, 2:3]*hf_s)
            fused   = self.dropout(fused)

            danger_score = torch.zeros(batch, 1, device=device)

            if self.use_bio:
                vol_t   = vt if vt is not None else torch.zeros(batch, 1, device=device)
                log_vol = torch.log1p(vol_t.abs())
                vov     = (log_vol - prev_log_vol).abs() if prev_log_vol is not None else torch.zeros_like(log_vol)
                prev_log_vol = log_vol.detach()

                self.danger_det.update(fused)
                danger_score = self.danger_det(fused)
                # v0.8.1: NaN/Inf guard — if hidden state is polluted (e.g. early
                # NaN features), fall back to 0 so int() cast never fails.
                danger_mean = self._safe_scalar(danger_score, fallback=0.0)
                max_danger  = torch.max(max_danger, torch.tensor(danger_mean, device=device))

                if delta > thresh:
                    hidden['slow'] = (
                        self.danger_det.selective_reset(hidden['slow'][0], danger_score),
                        self.danger_det.selective_reset(hidden['slow'][1], danger_score),
                    )

                slow_update_every = max(1, int(3.0 / (1.0 + 3.0 * danger_mean)))

                # v0.8: tda mix uses learnable weight
                fused = fused + self.tda_w * tda_h_base

                router_weights = self.router(
                    h=fused,
                    regime=regime_scalar_mean.unsqueeze(-1),
                    danger=danger_score,
                    vov=vov,
                )
                all_router_w.append(router_weights)
                w = router_weights

                # v0.8: ctx from h_fused statistics instead of regime/vol/vov
                ctx = self._h_ctx(fused)  # (B,1,3): [h_mean, h_std, h_norm]

                h_t   = fused.unsqueeze(1)
                rpe_t = ((pe_f + pe_m + pe_s) / 3.0).unsqueeze(1)

                cereb_err_val = 0.0
                if prev_fused is not None:
                    h_pred_cereb = self.cerebellum(prev_fused, xt_aug)
                    cereb_err    = self.cerebellum.cerebellar_error(h_pred_cereb, fused)
                    w_cereb      = w[:, MODULE_NAMES.index("cerebellum")].unsqueeze(-1).unsqueeze(-1)
                    # v0.8: learnable cereb rpe weight
                    rpe_t        = rpe_t + w_cereb * self.cereb_w * cereb_err.unsqueeze(1)
                    cereb_err_val = cereb_err.detach().mean().item()
                    all_cereb_err.append(cereb_err_val)
                prev_fused = fused.detach()

                self.transition_det.update_queue(delta)

                h_thal = self.thal_attn(h_t, ctx)
                w_thal = w[:, MODULE_NAMES.index("thalamic")].unsqueeze(-1).unsqueeze(-1)
                h_t    = (1-w_thal)*h_t + w_thal*h_thal

                h_neuro, neuro_d = self.neuro_gate(h_t, ctx)
                w_neuro = w[:, MODULE_NAMES.index("neuromod")].unsqueeze(-1).unsqueeze(-1)
                h_t     = (1-w_neuro)*h_t + w_neuro*h_neuro

                ne_boosted   = self.danger_det.ne_boost(danger_score, neuro_d["norepinephrine_raw"].squeeze(1))
                w_danger     = w[:, MODULE_NAMES.index("danger")].unsqueeze(-1)
                ne_effective = (1-w_danger)*neuro_d["norepinephrine_raw"].squeeze(1) + w_danger*ne_boosted

                h_ast = self.astrocyte(h_t.squeeze(1)).unsqueeze(1)
                w_ast = w[:, MODULE_NAMES.index("astrocyte")].unsqueeze(-1).unsqueeze(-1)
                h_t   = (1-w_ast)*h_t + w_ast*h_ast

                h_pfc = self.pfc_wm(h_t, rpe=rpe_t)
                w_pfc = w[:, MODULE_NAMES.index("pfc_wm")].unsqueeze(-1).unsqueeze(-1)
                h_t   = (1-w_pfc)*h_t + w_pfc*h_pfc

                h_strat, strategy_info = self.strategy_bank(h_t, ctx, rpe_t)
                w_strat = w[:, MODULE_NAMES.index("strategy")].unsqueeze(-1).unsqueeze(-1)
                h_t     = (1-w_strat)*h_t + w_strat*h_strat

                trans_prob = self.transition_det(h_t)
                all_trans_prob.append(trans_prob)

                w_tda = w[:, MODULE_NAMES.index("tda")].unsqueeze(-1)
                fused = h_t.squeeze(1) + w_tda * tda_h_base * self.tda_w
                h_t   = fused.unsqueeze(1)

                fused_latent, expert_states, latent_w = self.latent_experts(h_t.squeeze(1), expert_states)
                h_t   = fused_latent.unsqueeze(1)
                fused = fused_latent
                all_latent_w.append(latent_w.detach().mean(dim=0).cpu())

                self.homeostasis.update(
                    da_mean=neuro_d["dopamine"].mean().item(),
                    ne_mean=ne_effective.mean().item(),
                    sht_mean=neuro_d["serotonin"].mean().item(),
                )
                fused = h_t.squeeze(1)

                all_fused.append(fused.unsqueeze(1))
                all_regime_t.append(regime_scalar_mean.unsqueeze(-1).unsqueeze(-1))

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
                self.memory.write(fused, regime_scalar_mean)
                fused = self.memory.read(fused, regime_scalar_mean)

            mean_out, std_out = self.head(fused)
            means.append(mean_out.unsqueeze(1))
            stds.append(std_out.unsqueeze(1))

            regime_stack = torch.stack([reg_scalar_f, reg_scalar_m, reg_scalar_s], dim=1).unsqueeze(1)
            all_regime.append(regime_stack)
            all_scale_weights.append(weights.unsqueeze(1))

        means   = torch.cat(means,  dim=1)
        stds    = torch.cat(stds,   dim=1)
        regime  = torch.cat(all_regime, dim=1)
        scale_w = torch.cat(all_scale_weights, dim=1)

        neuro_summary = {k: (sum(v)/len(v) if v else 0.0) for k, v in all_neuro_scalar.items()}

        if all_router_w:
            router_w_full = torch.stack(all_router_w, dim=1)
            router_w_mean = router_w_full.mean(dim=(0,1)).detach().cpu().tolist()
        else:
            router_w_full = None
            router_w_mean = []

        regime_consist_loss = torch.tensor(0.0, device=device)
        if self.use_bio and router_w_full is not None and all_regime_t:
            regime_cat = torch.cat(all_regime_t, dim=1)
            regime_consist_loss = self.router.regime_consistency_loss(router_w_full, regime_cat)

        contrastive_loss = torch.tensor(0.0, device=device)
        if self.use_bio and all_fused and all_regime_t:
            fused_cat  = torch.cat(all_fused, dim=1)
            regime_cat = torch.cat(all_regime_t, dim=1)
            contrastive_loss = self.contrastive_reg(fused_cat, regime_cat)

        if self.use_bio and all_da:
            neuro_tensors = {
                "dopamine":       torch.cat(all_da,  dim=1),
                "norepinephrine": torch.cat(all_ne,  dim=1),
                "serotonin":      torch.cat(all_sht, dim=1),
            }
            vov_full      = torch.cat(all_vov,        dim=1)
            rpe_full      = torch.cat(all_rpe_t,      dim=1)
            strat_raw     = torch.cat(all_strategy_w, dim=1)
            trans_stack   = torch.cat(all_trans_prob, dim=1) if all_trans_prob else None
            sw_mean       = strat_raw.mean(dim=(0,1)).detach().cpu().tolist()
            dom           = int(strat_raw.mean(dim=(0,1)).argmax().item())
            avg_cereb     = sum(all_cereb_err)/len(all_cereb_err) if all_cereb_err else 0.0
            danger_full   = torch.cat(all_danger, dim=0).mean().item() if all_danger else 0.0
            latent_w_mean = torch.stack(all_latent_w).mean(dim=0).tolist() if all_latent_w else []
        else:
            neuro_tensors = {}
            vov_full = rpe_full = strat_raw = trans_stack = None
            sw_mean = []; dom = -1; avg_cereb = 0.0; danger_full = 0.0
            latent_w_mean = []

        # learnable hyperparam values for logging
        learnable_params = {
            "regime_reset_threshold": self.log_regime_thresh.exp().item(),
            "hidden_decay_rate":      self.log_decay_rate.exp().item(),
            "tda_mix_weight":         self.log_tda_w.exp().item(),
            "cereb_rpe_weight":       self.log_cereb_w.exp().item(),
            "regime_loss_weight":     self.log_regime_loss_w.exp().item(),
        }

        extra = {
            "pred_coding_loss":        total_pred_err / (seq_len * 3),
            "regime_self_loss":        total_regime_loss / (seq_len * 3),
            "regime_probs":            regime,
            "scale_weights":           scale_w,
            "final_hidden":            hidden,
            "neuro":                   neuro_summary,
            "neuro_tensors":           neuro_tensors,
            "vol_of_vol":              vov_full,
            "rpe_tensor":              rpe_full,
            "strategy_w_raw":          strat_raw,
            "strategy_weights":        sw_mean,
            "dominant_strategy":       dom,
            "trans_prob":              trans_stack,
            "cereb_err":               avg_cereb,
            "transition_label":        transition_label,
            "router_weights":          router_w_full,
            "router_w_mean":           router_w_mean,
            "danger_score":            danger_full,
            "cpg_periods":             [o.period.item() for o in self.cpg.oscillators],
            "astrocyte_alpha":         self.astrocyte.alpha.item() if self.use_bio else None,
            "astrocyte_beta":          self.astrocyte.beta.item()  if self.use_bio else None,
            "router_temperature":      self.router.temperature.item() if self.use_bio else None,
            "regime_consistency_loss": regime_consist_loss,
            "contrastive_loss":        contrastive_loss,
            "latent_expert_weights":   latent_w_mean,
            "curriculum_weight":       max_danger.item(),
            "learnable_params":        learnable_params,
        }
        return means, stds, extra
