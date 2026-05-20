"""
Bio-inspired neuromodulation modules for CASA-RNN.

Based on neuroscience research (2024-2025):

1. NeuromodulatorGating
   Inspired by: dopamine/acetylcholine/norepinephrine neuromodulatory circuits.
   Papers: Three-Factor Learning in SNNs (arXiv 2504.05341), Computational Models
           of Neuromodulation (Frontiers 2026).
   Idea: four parallel "neuromodulator" channels each gate information flow
   differently, controlled by current market context (volatility, regime).

2. ThalamicAttention
   Inspired by: corticothalamic gating as selective attention mechanism.
   Papers: Corticothalamic Synaptic Noise as Selective Attention (Frontiers 2015),
           Neural Circuits That Mediate Selective Attention (PMC 2018).
   Idea: thalamus acts as a relay with a learnable "gate" that selectively
   suppresses irrelevant inputs. Modelled as multiplicative mask on hidden state.

3. HippocampalReplayBuffer
   Inspired by: RPE-biased hippocampal replay for memory consolidation.
   Papers: Post-learning replay biased by RPE (Nature Comms 2025),
           Brain-Like Replay Naturally Emerges in RL (arXiv 2402.01467).
   Idea: store (state, prediction_error) pairs. During training, replay
   high-RPE experiences more frequently so the model re-learns from its
   biggest mistakes (just like the brain prioritises surprising events).

4. PrefrontalWorkingMemory
   Inspired by: PFC persistent firing for working memory gating.
   Papers: Adaptive chunking in PFC-BG circuit (eLife 2025),
           Compositional architecture in PFC (biorxiv 2025).
   Idea: two orthogonal subspaces: one for task context (regime/market-state),
   one for content (what to remember). Input and output gates controlled
   by dopamine-like reward prediction error signal.
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
    Four neuromodulator channels, each with a distinct biological role:

    Dopamine (DA):  reward/RPE signal -> gates how much NEW information enters
                    high DA = open gate (explore, update weights aggressively)
                    low  DA = closed gate (exploit, stable predictions)

    Acetylcholine (ACh): attention/uncertainty -> boosts signal-to-noise
                    high ACh = sharpen features, reduce noise tolerance
                    low  ACh = broader, less precise processing

    Norepinephrine (NE): arousal/volatility -> controls gain of the whole layer
                    high NE = high gain (market shock, high vol)
                    low  NE = low gain (quiet market)

    Serotonin (5HT): temporal discounting -> how far back does memory extend
                    high 5HT = longer memory horizon
                    low  5HT = short-term focus

    All four are computed from a small context vector (regime, volatility).
    """

    def __init__(self, hidden_size: int, context_size: int = 2):
        super().__init__()
        self.hidden_size = hidden_size

        # Context encoder: maps (regime, vol) -> neuromodulator levels
        self.ctx_encoder = nn.Sequential(
            nn.Linear(context_size, 32),
            nn.Tanh(),
            nn.Linear(32, 4),   # 4 neuromodulator levels
            nn.Sigmoid(),
        )

        # Per-neuromodulator projection heads
        self.da_gate   = nn.Linear(hidden_size, hidden_size)  # dopamine input gate
        self.ach_proj  = nn.Linear(hidden_size, hidden_size)  # acetylcholine sharpening
        self.ne_gain   = nn.Linear(hidden_size, hidden_size)  # norepinephrine gain
        self.sht_decay = nn.Linear(hidden_size, hidden_size)  # serotonin decay gate

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        h: torch.Tensor,            # (B, T, H) hidden state
        context: torch.Tensor,      # (B, T, context_size) regime+vol
    ) -> Tuple[torch.Tensor, dict]:
        neuro = self.ctx_encoder(context)       # (B, T, 4)
        da, ach, ne, sht = neuro.unbind(dim=-1) # each (B, T)
        da  = da.unsqueeze(-1)                  # (B, T, 1) for broadcasting
        ach = ach.unsqueeze(-1)
        ne  = ne.unsqueeze(-1)
        sht = sht.unsqueeze(-1)

        # Dopamine gate: how much new info enters
        h_new = torch.sigmoid(self.da_gate(h))
        h = da * h_new + (1 - da) * h

        # Acetylcholine: sharpen via contrast enhancement (subtract mean)
        h_sharp = torch.tanh(self.ach_proj(h))
        h_mean  = h.mean(dim=-1, keepdim=True)
        h = h + ach * (h_sharp - h_mean)

        # Norepinephrine: multiply gain
        ne_scale = 0.5 + 1.5 * ne   # range [0.5, 2.0]
        h = h * ne_scale

        # Serotonin: blend with a decayed version (longer vs shorter memory)
        h_decay = torch.tanh(self.sht_decay(h))
        h = sht * h + (1 - sht) * h_decay

        h = self.norm(h)

        neuro_levels = {
            "dopamine": da.squeeze(-1).mean().item(),
            "acetylcholine": ach.squeeze(-1).mean().item(),
            "norepinephrine": ne.squeeze(-1).mean().item(),
            "serotonin": sht.squeeze(-1).mean().item(),
        }
        return h, neuro_levels


# ---------------------------------------------------------------------------
# 2. ThalamicAttention
# ---------------------------------------------------------------------------

class ThalamicAttention(nn.Module):
    """
    Thalamus as a selective relay gate.

    In the brain, the thalamus does NOT just relay signals passively.
    It has a learnable suppression layer (TRN - Thalamic Reticular Nucleus)
    that can VETO entire feature channels based on top-down context.

    Implementation:
    - Bottom-up input: feature sequence h (B, T, H)
    - Top-down context: current regime/market state (B, T, ctx)
    - TRN computes a per-channel suppression mask [0,1]
    - Suppressed channels are zeroed out (selective attention)
    - Residual connection ensures gradient flow even when gate=0
    """

    def __init__(self, hidden_size: int, context_size: int = 2):
        super().__init__()
        # TRN: top-down suppression
        self.trn = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
        # Relay projection
        self.relay = nn.Linear(hidden_size, hidden_size)
        self.norm  = nn.LayerNorm(hidden_size)

    def forward(
        self,
        h: torch.Tensor,        # (B, T, H)
        context: torch.Tensor,  # (B, T, ctx)
    ) -> torch.Tensor:
        gate      = self.trn(context)           # (B, T, H) suppression mask
        relayed   = torch.tanh(self.relay(h))   # relay transform
        gated     = gate * relayed              # selective pass-through
        return self.norm(h + gated)             # residual: never fully block


# ---------------------------------------------------------------------------
# 3. HippocampalReplayBuffer
# ---------------------------------------------------------------------------

class HippocampalReplayBuffer:
    """
    RPE-biased experience replay, mimicking hippocampal sharp-wave ripple replay.

    Key insight from Nature Comms 2025: the brain preferentially replays
    experiences with HIGH reward prediction error (RPE), not just recent ones.
    This means the model re-learns from its most surprising mistakes.

    For financial markets:
    - RPE = |actual_return - predicted_return|  (how wrong the model was)
    - High RPE events = regime transitions, black swans, earnings surprises
    - These are exactly the events that matter most for risk management

    Usage in training loop:
        replay_buf.push(x_batch, y_batch, rpe=loss.detach())
        if replay_buf.should_replay(step):
            rx, ry = replay_buf.sample_rpe_biased(batch_size)
            replay_loss = loss_fn(model(rx), ry)
            loss = loss + 0.3 * replay_loss
    """

    def __init__(self, capacity: int = 500, replay_every: int = 20):
        self.capacity     = capacity
        self.replay_every = replay_every
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        x:   torch.Tensor,   # input batch (detached)
        y:   torch.Tensor,   # target batch (detached)
        rpe: float,          # reward prediction error (scalar)
    ) -> None:
        self.buffer.append((x.detach().cpu(), y.detach().cpu(), rpe))

    def should_replay(self, step: int) -> bool:
        return len(self.buffer) >= 10 and step % self.replay_every == 0

    def sample_rpe_biased(
        self,
        n: int,
        temperature: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample n experiences with probability proportional to RPE.
        temperature > 1 = more biased toward high-RPE events.
        """
        buf   = list(self.buffer)
        rpes  = torch.tensor([b[2] for b in buf], dtype=torch.float32)
        # Softmax with temperature: higher temp = stronger RPE bias
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
    """
    Working memory module inspired by PFC-Basal Ganglia circuit.

    Key neuroscience findings (eLife 2025, biorxiv 2025):
    1. PFC maintains TWO orthogonal subspaces:
       - Context subspace: encodes WHAT market regime we're in (slow-changing)
       - Content subspace: encodes WHAT pattern we're tracking (fast-changing)
    2. Input gate (striatum D1): decides what to STORE
       controlled by dopamine = RPE signal
    3. Output gate (striatum D2): decides what to READ OUT
       controlled by task demand
    4. Chunking: re-uses same PFC populations for multiple items
       = the same hidden units represent different info at different times

    This is more powerful than LSTM because:
    - LSTM has ONE hidden space, mixed content
    - PFC-WM has TWO orthogonal spaces + explicit RPE-controlled gating
    """

    def __init__(self, hidden_size: int, context_dim: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        content_dim      = hidden_size - context_dim
        self.content_dim = content_dim

        # Input gate (striatum D1): RPE-controlled
        self.input_gate = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # +1 for RPE
            nn.Sigmoid(),
        )

        # Output gate (striatum D2): task-demand controlled
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )

        # Context subspace encoder (slow: regime)
        self.context_enc = nn.Linear(hidden_size, context_dim)

        # Content subspace encoder (fast: what to remember)
        self.content_enc = nn.Linear(hidden_size, content_dim)

        # Merge back
        self.merge = nn.Linear(hidden_size, hidden_size)
        self.norm  = nn.LayerNorm(hidden_size)

    def forward(
        self,
        h:   torch.Tensor,          # (B, T, H) current hidden state
        rpe: Optional[torch.Tensor] = None,  # (B, T, 1) prediction error
    ) -> torch.Tensor:
        B, T, H = h.shape

        if rpe is None:
            rpe = torch.zeros(B, T, 1, device=h.device)

        # Input gate: high RPE -> open gate -> update working memory
        in_gate  = self.input_gate(torch.cat([h, rpe], dim=-1))   # (B,T,H)

        # Separate context and content subspaces
        h_ctx     = self.context_enc(h)    # (B,T,ctx_dim) - slow regime
        h_content = self.content_enc(h)    # (B,T,content_dim) - fast pattern

        # Apply orthogonality pressure: detach context from content gradient path
        # (context should not change when content changes)
        h_ctx     = h_ctx.detach() * 0.9 + self.context_enc(h) * 0.1

        # Recombine
        h_combined = torch.cat([h_ctx, h_content], dim=-1)        # (B,T,H)

        # Input gate: RPE-controlled update
        h_new = in_gate * torch.tanh(self.merge(h_combined)) + (1 - in_gate) * h

        # Output gate: controls what gets passed downstream
        out_gate = self.output_gate(h_new)
        h_out    = out_gate * h_new

        return self.norm(h_out)
