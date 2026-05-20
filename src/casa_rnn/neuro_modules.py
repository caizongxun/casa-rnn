"""
Bio-inspired neuromodulation + SoftModuleRouter.

Modules:
1. NeuromodulatorGating
2. ThalamicAttention
3. HippocampalReplayBuffer
4. PrefrontalWorkingMemory
5. MetaLearningStrategyBank
6. RegimeTransitionDetector
7. CerebellarForwardModel
8. HomeostaticGainControl
9. SoftModuleRouter  — learned mixture of all optional modules

Changelog:
  - SoftModuleRouter: added regime_consistency_loss() (Direction A)
  - ContrastiveStateRegularizer: new class (Direction C)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Optional, Tuple
import random


# ---------------------------------------------------------------------------
# 1. NeuromodulatorGating
# ---------------------------------------------------------------------------

class NeuromodulatorGating(nn.Module):
    def __init__(self, hidden_size: int, context_size: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.ctx_encoder = nn.Sequential(
            nn.Linear(context_size, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 4),
        )
        self.da_gate   = nn.Linear(hidden_size, hidden_size)
        self.ach_proj  = nn.Linear(hidden_size, hidden_size)
        self.ne_gain   = nn.Linear(hidden_size, hidden_size)
        self.sht_decay = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raw = self.ctx_encoder(context)
        da  = torch.sigmoid(raw[..., 0:1] * 2.5)
        ach = torch.sigmoid(raw[..., 1:2] * 2.0)
        ne  = torch.sigmoid(raw[..., 2:3] * 3.5)
        sht = torch.sigmoid(raw[..., 3:4] * 2.0)
        plastic  = torch.sigmoid(self.da_gate(h))
        h        = (1.0 - da) * h + da * plastic
        h_sharp  = torch.tanh(self.ach_proj(h))
        h        = h + ach * 1.5 * (h_sharp - h.mean(dim=-1, keepdim=True))
        gain     = 0.25 + 2.5 * ne
        h        = h * gain + 0.1 * ne * torch.tanh(self.ne_gain(h))
        smooth   = torch.tanh(self.sht_decay(h))
        h        = sht * h + (1.0 - sht) * smooth
        h        = self.norm(h)
        return h, {
            "dopamine":           da.detach(),
            "acetylcholine":      ach.detach(),
            "norepinephrine":     ne.detach(),
            "serotonin":          sht.detach(),
            "dopamine_raw":       da,
            "acetylcholine_raw":  ach,
            "norepinephrine_raw": ne,
            "serotonin_raw":      sht,
        }


# ---------------------------------------------------------------------------
# 2. ThalamicAttention
# ---------------------------------------------------------------------------

class ThalamicAttention(nn.Module):
    def __init__(self, hidden_size: int, context_size: int = 3):
        super().__init__()
        self.trn   = nn.Sequential(
            nn.Linear(context_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Sigmoid(),
        )
        self.relay = nn.Linear(hidden_size, hidden_size)
        self.norm  = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = self.trn(context)
        return self.norm(h + gate * torch.tanh(self.relay(h)))


# ---------------------------------------------------------------------------
# 3. HippocampalReplayBuffer
# ---------------------------------------------------------------------------

class HippocampalReplayBuffer:
    def __init__(self, capacity: int = 500, replay_every: int = 20):
        self.capacity     = capacity
        self.replay_every = replay_every
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, x: torch.Tensor, y: torch.Tensor, rpe: float) -> None:
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for i in range(x_cpu.shape[0]):
            self.buffer.append((x_cpu[i], y_cpu[i], rpe))

    def should_replay(self, step: int) -> bool:
        return len(self.buffer) >= 16 and step % self.replay_every == 0

    def sample_rpe_biased(self, n: int, temperature: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        buf   = list(self.buffer)
        rpes  = torch.tensor([b[2] for b in buf], dtype=torch.float32)
        probs = F.softmax(rpes * temperature, dim=0).numpy()
        idxs  = random.choices(range(len(buf)), weights=probs, k=n)
        xs    = torch.stack([buf[i][0] for i in idxs])
        ys    = torch.stack([buf[i][1] for i in idxs])
        return xs, ys

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 4. PrefrontalWorkingMemory
# ---------------------------------------------------------------------------

class PrefrontalWorkingMemory(nn.Module):
    def __init__(self, hidden_size: int, context_dim: int = 16):
        super().__init__()
        content_dim      = hidden_size - context_dim
        self.context_enc = nn.Linear(hidden_size, context_dim)
        self.content_enc = nn.Linear(hidden_size, content_dim)
        self.input_gate  = nn.Sequential(nn.Linear(hidden_size + 1, hidden_size), nn.Sigmoid())
        self.output_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.merge       = nn.Linear(hidden_size, hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor, rpe: Optional[torch.Tensor] = None) -> torch.Tensor:
        if rpe is None:
            rpe = torch.zeros(*h.shape[:2], 1, device=h.device)
        in_gate    = self.input_gate(torch.cat([h, rpe], dim=-1))
        h_ctx      = self.context_enc(h).detach() * 0.9 + self.context_enc(h) * 0.1
        h_content  = self.content_enc(h)
        h_combined = torch.cat([h_ctx, h_content], dim=-1)
        h_new      = in_gate * torch.tanh(self.merge(h_combined)) + (1 - in_gate) * h
        return self.norm(self.output_gate(h_new) * h_new)


# ---------------------------------------------------------------------------
# 5. MetaLearningStrategyBank
# ---------------------------------------------------------------------------

STRATEGY_NAMES = ["rehearsal", "chunking", "associative", "contrastive", "slow"]


class MetaLearningStrategyBank(nn.Module):
    def __init__(self, hidden_size: int, context_size: int = 3, num_strategies: int = 5):
        super().__init__()
        self.num_strategies = num_strategies
        self.selector    = nn.Sequential(
            nn.Linear(hidden_size + context_size + 1, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, num_strategies),
        )
        self.rehearsal   = nn.Linear(hidden_size, hidden_size)
        self.chunking    = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.associative = nn.Linear(hidden_size, hidden_size)
        self.contrastive = nn.Linear(hidden_size, hidden_size)
        self.slow        = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.norm        = nn.LayerNorm(hidden_size)

    def forward(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        rpe: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        sel_in = torch.cat([h, context, rpe], dim=-1)
        logits = self.selector(sel_in)
        w      = torch.softmax(logits, dim=-1)
        s0     = self.rehearsal(h)
        s1     = self.chunking(h)
        s2     = torch.tanh(self.associative(h))
        s3     = h - torch.tanh(self.contrastive(h))
        s4     = 0.7 * h + 0.3 * self.slow(h)
        stack  = torch.stack([s0, s1, s2, s3, s4], dim=-1)
        mixed  = (stack * w.unsqueeze(-2)).sum(dim=-1)
        out    = self.norm(h + mixed)
        w_mean = w.mean(dim=(0, 1))
        return out, {
            "strategy_weights":  w_mean.detach().cpu(),
            "dominant_strategy": int(w_mean.argmax().item()),
            "strategy_w_tensor": w,
        }

    def entropy_loss(self, w: torch.Tensor) -> torch.Tensor:
        w_mean = w.mean(dim=(0, 1)).clamp(min=1e-8)
        return -(w_mean * w_mean.log()).sum()


# ---------------------------------------------------------------------------
# 6. RegimeTransitionDetector
# ---------------------------------------------------------------------------

class RegimeTransitionDetector(nn.Module):
    def __init__(self, hidden_size: int, window: int = 8):
        super().__init__()
        self.window  = window
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size + window, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, 1), nn.Sigmoid(),
        )
        self.register_buffer("delta_queue", torch.zeros(window))

    def update_queue(self, regime_delta: float) -> None:
        self.delta_queue = torch.roll(self.delta_queue, -1)
        self.delta_queue[-1] = regime_delta

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_flat    = h.squeeze(1) if h.dim() == 3 else h
        B         = h_flat.shape[0]
        delta_feat = self.delta_queue.unsqueeze(0).expand(B, -1).to(h_flat.device)
        inp        = torch.cat([h_flat, delta_feat], dim=-1)
        return self.encoder(inp)

    def loss(self, trans_prob: torch.Tensor, label: float) -> torch.Tensor:
        target = torch.full_like(trans_prob, label)
        return F.binary_cross_entropy(trans_prob, target)


# ---------------------------------------------------------------------------
# 7. CerebellarForwardModel
# ---------------------------------------------------------------------------

class CerebellarForwardModel(nn.Module):
    def __init__(self, hidden_size: int, input_size: int, proj_size: int = 32):
        super().__init__()
        self.proj_size = proj_size
        self.h_proj    = nn.Linear(hidden_size, proj_size)
        self.x_proj    = nn.Linear(input_size,  proj_size)
        self.predict   = nn.Sequential(
            nn.Linear(proj_size * 2, proj_size), nn.Tanh(),
            nn.Linear(proj_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h_now: torch.Tensor, x_now: torch.Tensor) -> torch.Tensor:
        hp = torch.tanh(self.h_proj(h_now))
        xp = torch.tanh(self.x_proj(x_now))
        return self.norm(self.predict(torch.cat([hp, xp], dim=-1)))

    def cerebellar_error(self, h_pred: torch.Tensor, h_next: torch.Tensor) -> torch.Tensor:
        return (h_pred - h_next.detach()).pow(2).mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 8. HomeostaticGainControl
# ---------------------------------------------------------------------------

class HomeostaticGainControl(nn.Module):
    def __init__(self, da_set: float = 0.55, ne_set: float = 0.35, sht_set: float = 0.50,
                 ema_alpha: float = 0.02, penalty_weight: float = 0.03):
        super().__init__()
        self.da_set  = da_set
        self.ne_set  = ne_set
        self.sht_set = sht_set
        self.alpha   = ema_alpha
        self.weight  = penalty_weight
        self.register_buffer("da_ema",  torch.tensor(da_set))
        self.register_buffer("ne_ema",  torch.tensor(ne_set))
        self.register_buffer("sht_ema", torch.tensor(sht_set))

    @torch.no_grad()
    def update(self, da_mean: float, ne_mean: float, sht_mean: float) -> None:
        self.da_ema  = (1 - self.alpha) * self.da_ema  + self.alpha * da_mean
        self.ne_ema  = (1 - self.alpha) * self.ne_ema  + self.alpha * ne_mean
        self.sht_ema = (1 - self.alpha) * self.sht_ema + self.alpha * sht_mean

    def homeostatic_loss(
        self,
        da:  torch.Tensor,
        ne:  torch.Tensor,
        sht: torch.Tensor,
    ) -> torch.Tensor:
        l_da  = (da.mean()  - self.da_set).pow(2)
        l_ne  = (ne.mean()  - self.ne_set).pow(2)
        l_sht = (sht.mean() - self.sht_set).pow(2)
        return self.weight * (l_da + l_ne + l_sht)


# ---------------------------------------------------------------------------
# 9. SoftModuleRouter
# ---------------------------------------------------------------------------

MODULE_NAMES = ["thalamic", "neuromod", "pfc_wm", "strategy", "transition", "cerebellum", "astrocyte", "danger", "tda", "cpg"]


class SoftModuleRouter(nn.Module):
    """
    Learned soft gating over all optional bio/topology modules.

    Changelog:
      - regime_consistency_loss(): Direction A — same regime -> same routing.
        Penalises timesteps in the same regime whose router weights diverge,
        and rewards divergence between different regimes.
        This forces the router to specialise PER REGIME rather than averaging
        across regimes, which is why temperature was previously stuck at ~1.9.
    """
    def __init__(self, hidden_size: int, num_modules: int = len(MODULE_NAMES),
                 init_temperature: float = 2.0):
        super().__init__()
        self.num_modules = num_modules
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())
        self.router = nn.Sequential(
            nn.Linear(hidden_size + 3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_modules),
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.1, 5.0)

    def forward(
        self,
        h: torch.Tensor,       # (B, H)
        regime: torch.Tensor,  # (B, 1)
        danger: torch.Tensor,  # (B, 1)
        vov:    torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        ctx    = torch.cat([h, regime, danger, vov], dim=-1)
        logits = self.router(ctx)
        return torch.softmax(logits / self.temperature, dim=-1)

    def entropy_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Encourage diverse module usage during warmup."""
        w = weights.mean(dim=list(range(weights.dim() - 1))).clamp(min=1e-8)
        return -(w * w.log()).sum()

    def sparsity_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """After warmup: encourage sparse / specialised routing."""
        w = weights.mean(dim=list(range(weights.dim() - 1))).clamp(min=1e-8)
        return (w * w.log()).sum()

    def regime_consistency_loss(
        self,
        weights: torch.Tensor,   # (B, T, num_modules)
        regime:  torch.Tensor,   # (B, T, 1) or (B, T, 2) regime probs
    ) -> torch.Tensor:
        """
        Direction A: Regime-Consistent Router.

        For each pair of timesteps within the same batch:
          - same regime (regime diff < 0.2)  -> router weights should be similar
            (minimise cosine distance within regime)
          - different regime (diff > 0.5)    -> router weights should differ
            (maximise cosine distance across regimes)

        Uses a lightweight mean-per-regime approximation to avoid O(T^2) pairs:
          1. Segment sequence into low/high regime bins
          2. Compute mean router weight per bin
          3. Penalise intra-bin variance, reward inter-bin distance
        """
        # Use first channel of regime as regime scalar
        reg_scalar = regime[..., 0]  # (B, T)

        # Split into two regime bins: low (< 0.4) and high (> 0.6)
        low_mask  = (reg_scalar < 0.4)   # (B, T)
        high_mask = (reg_scalar > 0.6)   # (B, T)

        loss = torch.tensor(0.0, device=weights.device)
        n_terms = 0

        for mask in [low_mask, high_mask]:
            if mask.sum() < 2:
                continue
            # mean router weight within this regime bin
            w_in = weights[mask]          # (N, num_modules)
            w_mean = w_in.mean(dim=0)     # (num_modules,)
            # intra-regime variance: penalise spread
            intra_var = ((w_in - w_mean.unsqueeze(0)).pow(2)).mean()
            loss = loss + intra_var
            n_terms += 1

        # inter-regime: reward distance between low and high regime routing
        if low_mask.sum() > 0 and high_mask.sum() > 0:
            w_low  = weights[low_mask].mean(dim=0)
            w_high = weights[high_mask].mean(dim=0)
            sim    = F.cosine_similarity(w_low.unsqueeze(0), w_high.unsqueeze(0))
            # we want low similarity between regimes -> penalise high sim
            loss = loss + sim.squeeze() * 0.5
            n_terms += 1

        return loss / max(n_terms, 1)


# ---------------------------------------------------------------------------
# 10. ContrastiveStateRegularizer  (Direction C)
# ---------------------------------------------------------------------------

class ContrastiveStateRegularizer(nn.Module):
    """
    Direction C: Contrastive State Regularizer.

    Encourages hidden states to form regime-aware clusters:
      - Positive pairs:  same regime bin -> small cosine distance
      - Negative pairs:  different regime -> large cosine distance

    Uses NT-Xent (normalised temperature-scaled cross entropy) style loss
    on mean regime embeddings, which is O(B) not O(B^2).

    This is an AUXILIARY loss added on top of the task loss.
    Weight recommended: 0.01~0.05 (start small, real data may need tuning).
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        hidden: torch.Tensor,   # (B, T, H) fused hidden states
        regime: torch.Tensor,   # (B, T, 1) or (B, T, 2)
    ) -> torch.Tensor:
        reg_scalar = regime[..., 0]  # (B, T)
        low_mask   = (reg_scalar < 0.4)
        high_mask  = (reg_scalar > 0.6)

        if low_mask.sum() < 2 or high_mask.sum() < 2:
            return torch.tensor(0.0, device=hidden.device)

        # Flatten (B, T, H) -> (B*T, H) then mask
        h_flat  = hidden.reshape(-1, hidden.shape[-1])
        lm_flat = low_mask.reshape(-1)
        hm_flat = high_mask.reshape(-1)

        h_low  = F.normalize(h_flat[lm_flat], dim=-1)   # (N_low, H)
        h_high = F.normalize(h_flat[hm_flat], dim=-1)   # (N_high, H)

        # Use mean embeddings as anchors (avoids O(N^2) all-pairs)
        anchor_low  = h_low.mean(dim=0, keepdim=True)   # (1, H)
        anchor_high = h_high.mean(dim=0, keepdim=True)  # (1, H)

        # Positive sim: anchor_low with low samples
        sim_pos = (anchor_low * h_low).sum(dim=-1) / self.tau   # (N_low,)
        # Negative sim: anchor_low with high samples
        sim_neg = (anchor_low * h_high).sum(dim=-1) / self.tau  # (N_high,)

        # NT-Xent style: for each positive, contrast against all negatives
        pos_loss = -sim_pos.mean()
        neg_loss =  sim_neg.mean()

        # Symmetric: also do anchor_high vs its positives/negatives
        sim_pos2 = (anchor_high * h_high).sum(dim=-1) / self.tau
        sim_neg2 = (anchor_high * h_low).sum(dim=-1)  / self.tau
        pos_loss2 = -sim_pos2.mean()
        neg_loss2 =  sim_neg2.mean()

        return (pos_loss + neg_loss + pos_loss2 + neg_loss2) * 0.25
