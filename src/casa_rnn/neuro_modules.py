"""
Bio-inspired neuromodulation modules for CASA-RNN.

Based on neuroscience research (2024-2025):

1. NeuromodulatorGating
   Dopamine / Acetylcholine / Norepinephrine / Serotonin gating.
   Papers: Three-Factor Learning in SNNs (arXiv 2504.05341),
           Computational Models of Neuromodulation (Frontiers 2026).

2. ThalamicAttention
   Corticothalamic selective relay (TRN suppression).
   Papers: Corticothalamic Synaptic Noise as Selective Attention (Frontiers 2015),
           Neural Circuits That Mediate Selective Attention (PMC 2018).

3. HippocampalReplayBuffer
   RPE-biased sharp-wave ripple replay.
   Papers: Post-learning replay biased by RPE (Nature Comms 2025),
           Brain-Like Replay Naturally Emerges in RL (arXiv 2402.01467).

4. PrefrontalWorkingMemory
   PFC orthogonal context/content subspaces + BG gating.
   Papers: Adaptive chunking in PFC-BG circuit (eLife 2025),
           Compositional architecture in PFC (biorxiv 2025).
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
    def __init__(self, hidden_size: int, context_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.ctx_encoder = nn.Sequential(
            nn.Linear(context_size, 32), nn.Tanh(),
            nn.Linear(32, 4), nn.Sigmoid(),
        )
        self.da_gate   = nn.Linear(hidden_size, hidden_size)
        self.ach_proj  = nn.Linear(hidden_size, hidden_size)
        self.ne_gain   = nn.Linear(hidden_size, hidden_size)
        self.sht_decay = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        neuro = self.ctx_encoder(context)
        da, ach, ne, sht = [v.unsqueeze(-1) for v in neuro.unbind(dim=-1)]

        h = da * torch.sigmoid(self.da_gate(h)) + (1 - da) * h
        h_sharp = torch.tanh(self.ach_proj(h))
        h = h + ach * (h_sharp - h.mean(dim=-1, keepdim=True))
        h = h * (0.5 + 1.5 * ne)
        h = sht * h + (1 - sht) * torch.tanh(self.sht_decay(h))
        h = self.norm(h)

        return h, {
            "dopamine":       da.squeeze(-1).mean().item(),
            "acetylcholine":  ach.squeeze(-1).mean().item(),
            "norepinephrine": ne.squeeze(-1).mean().item(),
            "serotonin":      sht.squeeze(-1).mean().item(),
        }


# ---------------------------------------------------------------------------
# 2. ThalamicAttention
# ---------------------------------------------------------------------------

class ThalamicAttention(nn.Module):
    def __init__(self, hidden_size: int, context_size: int = 2):
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
    """
    RPE-biased experience replay.

    BUG FIX vs previous version:
      OLD: push() stored (B,T,D) tensor -> sample stacked them -> (n,B,T,D) = 4D -> crash
      NEW: push() stores INDIVIDUAL samples (T,D) from the batch
           sample_rpe_biased() stacks (T,D) tensors -> (n,T,D) = 3D = correct model input

    Each call to push() with a batch of size B adds B individual entries.
    """

    def __init__(self, capacity: int = 500, replay_every: int = 20):
        self.capacity     = capacity
        self.replay_every = replay_every
        # Each entry: (x: Tensor(T,D), y: Tensor(T,1), rpe: float)
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, x: torch.Tensor, y: torch.Tensor, rpe: float) -> None:
        """x: (B,T,D), y: (B,T,1) - stores each sample individually."""
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()
        for i in range(x_cpu.shape[0]):
            self.buffer.append((x_cpu[i], y_cpu[i], rpe))

    def should_replay(self, step: int) -> bool:
        return len(self.buffer) >= 16 and step % self.replay_every == 0

    def sample_rpe_biased(self, n: int, temperature: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (n,T,D) and (n,T,1) tensors biased toward high-RPE events."""
        buf   = list(self.buffer)
        rpes  = torch.tensor([b[2] for b in buf], dtype=torch.float32)
        probs = F.softmax(rpes * temperature, dim=0).numpy()
        idxs  = random.choices(range(len(buf)), weights=probs, k=n)
        xs = torch.stack([buf[i][0] for i in idxs])  # (n, T, D)
        ys = torch.stack([buf[i][1] for i in idxs])  # (n, T, 1)
        return xs, ys

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 4. PrefrontalWorkingMemory
# ---------------------------------------------------------------------------

class PrefrontalWorkingMemory(nn.Module):
    """
    PFC-BG working memory with orthogonal context/content subspaces.
    Input gate (D1 striatum): RPE-controlled - only surprising events update WM.
    Output gate (D2 striatum): task-demand controlled readout.
    """

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
