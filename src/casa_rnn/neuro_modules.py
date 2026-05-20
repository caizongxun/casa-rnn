"""
Bio-inspired neuromodulation modules for CASA-RNN.

Extended with:
5. MetaLearningStrategyBank
   Learns multiple internal "study / memory strategies" and mixes them adaptively.
   Analogy: humans memorize via repetition, chunking, association, compression,
   contrastive rehearsal... the model should discover which internal learning mode
   works best under each regime.
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
    """
    Stronger dynamic range than previous version.

    context = [regime, log_vol, vol_of_vol]
    - DA: surprise / plasticity gate
    - ACh: sharpening / attention precision
    - NE: gain control, should rise strongly in volatile periods
    - 5HT: memory smoothing / decay control
    """
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
        # context shape: (B,T,3)
        raw = self.ctx_encoder(context)
        # widen dynamic range
        da  = torch.sigmoid(raw[..., 0:1] * 2.5)
        ach = torch.sigmoid(raw[..., 1:2] * 2.0)
        ne  = torch.sigmoid(raw[..., 2:3] * 3.5)
        sht = torch.sigmoid(raw[..., 3:4] * 2.0)

        # DA: interpolate between old hidden and plastic rewrite
        plastic = torch.sigmoid(self.da_gate(h))
        h = (1.0 - da) * h + da * plastic

        # ACh: sharpening contrast
        h_sharp = torch.tanh(self.ach_proj(h))
        h = h + ach * 1.5 * (h_sharp - h.mean(dim=-1, keepdim=True))

        # NE: stronger multiplicative gain, now 0.25x ~ 2.75x
        gain = 0.25 + 2.5 * ne
        h = h * gain + 0.1 * ne * torch.tanh(self.ne_gain(h))

        # 5HT: smoothing vs rewriting
        smooth = torch.tanh(self.sht_decay(h))
        h = sht * h + (1.0 - sht) * smooth

        h = self.norm(h)
        return h, {
            "dopamine":       da.mean().item(),
            "acetylcholine":  ach.mean().item(),
            "norepinephrine": ne.mean().item(),
            "serotonin":      sht.mean().item(),
        }


# ---------------------------------------------------------------------------
# 2. ThalamicAttention
# ---------------------------------------------------------------------------

class ThalamicAttention(nn.Module):
    def __init__(self, hidden_size: int, context_size: int = 3):
        super().__init__()
        self.trn = nn.Sequential(
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
        xs = torch.stack([buf[i][0] for i in idxs])
        ys = torch.stack([buf[i][1] for i in idxs])
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

class MetaLearningStrategyBank(nn.Module):
    """
    Internal "learning methods" / memory strategies.

    The model chooses a mixture of strategies depending on context and surprise:
      - rehearsal: repeat / reinforce current representation
      - chunking: compress nearby dimensions into stable summary
      - associative: bind to transformed representation
      - contrastive: emphasize deviations and edges
      - slow_consolidation: push toward low-frequency stable memory

    This is a meta-learner over internal learning styles, not over optimizer steps.
    """
    def __init__(self, hidden_size: int, context_size: int = 3, num_strategies: int = 5):
        super().__init__()
        self.num_strategies = num_strategies
        self.selector = nn.Sequential(
            nn.Linear(hidden_size + context_size + 1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_strategies),
        )

        self.rehearsal = nn.Linear(hidden_size, hidden_size)
        self.chunking = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.associative = nn.Linear(hidden_size, hidden_size)
        self.contrastive = nn.Linear(hidden_size, hidden_size)
        self.slow = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor, context: torch.Tensor, rpe: torch.Tensor):
        # h:(B,T,H), context:(B,T,C), rpe:(B,T,1)
        sel_in = torch.cat([h, context, rpe], dim=-1)
        logits = self.selector(sel_in)
        w = torch.softmax(logits, dim=-1)

        s0 = self.rehearsal(h)
        s1 = self.chunking(h)
        s2 = torch.tanh(self.associative(h))
        s3 = h - torch.tanh(self.contrastive(h))
        s4 = 0.7 * h + 0.3 * self.slow(h)

        stack = torch.stack([s0, s1, s2, s3, s4], dim=-1)  # (B,T,H,S)
        mixed = (stack * w.unsqueeze(-2)).sum(dim=-1)
        out = self.norm(h + mixed)

        return out, {
            "strategy_weights": w.mean(dim=(0,1)).detach().cpu(),
            "dominant_strategy": int(w.mean(dim=(0,1)).argmax().item()),
        }
