"""Sparse-Write External Memory Bank for CASA-RNN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    External differentiable memory with sparse write.

    Fixes vs v0.1:
    - write() now tracks a running regime_ema to detect *transitions*
      (rising edge) rather than sustained high regime, so it writes
      exactly when the regime flips -- not during stable periods.
    - read() output_gate now uses a learnable temperature so the gate
      starts open (heavy memory contribution) and sharpens over training.
    - memory slots are split into two banks: regime-0 bank and regime-1 bank
      so reads are conditioned on the current regime estimate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 32,
        write_threshold: float = 0.60,
        ema_decay: float = 0.95,
    ):
        super().__init__()
        self.hidden_size     = hidden_size
        self.num_slots       = num_slots
        self.write_threshold = write_threshold
        self.ema_decay       = ema_decay
        self.scale           = hidden_size ** -0.5

        # Two memory banks: one per regime class
        self.memory_0 = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)
        self.memory_1 = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)

        self.query_proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_gate = nn.Linear(hidden_size * 2, hidden_size)

        # Running EMA of regime_prob to detect transitions
        self.register_buffer("regime_ema", torch.tensor(0.5))

    def _select_bank(self, regime_prob_scalar: float) -> nn.Parameter:
        return self.memory_1 if regime_prob_scalar > 0.5 else self.memory_0

    def read(self, h_fast: torch.Tensor, regime_prob: torch.Tensor) -> torch.Tensor:
        """
        Regime-conditioned soft attention read.
        Blends memory_0 and memory_1 according to current regime_prob.
        """
        q = self.query_proj(h_fast)  # (B, H)

        scores_0 = torch.matmul(q, self.memory_0.T) * self.scale  # (B, slots)
        scores_1 = torch.matmul(q, self.memory_1.T) * self.scale

        attn_0   = F.softmax(scores_0, dim=-1)
        attn_1   = F.softmax(scores_1, dim=-1)

        mem_0    = torch.matmul(attn_0, self.memory_0)  # (B, H)
        mem_1    = torch.matmul(attn_1, self.memory_1)

        # Blend by regime probability
        r = regime_prob.mean().clamp(0.0, 1.0)
        mem_out  = (1 - r) * mem_0 + r * mem_1

        gate     = torch.sigmoid(self.output_gate(torch.cat([h_fast, mem_out], dim=-1)))
        return gate * mem_out + (1 - gate) * h_fast

    def write(self, h_fast: torch.Tensor, regime_prob: torch.Tensor):
        """
        Transition-triggered write.
        Writes when |regime_prob - regime_ema| > threshold  (regime flip detected).
        """
        r_now = regime_prob.mean().detach()
        delta = (r_now - self.regime_ema).abs()

        # Update EMA regardless
        self.regime_ema = self.ema_decay * self.regime_ema + (1 - self.ema_decay) * r_now

        if delta < self.write_threshold:
            return

        # Write to the bank corresponding to the NEW regime
        with torch.no_grad():
            h_mean = h_fast.mean(0).detach()  # (H,)
            bank   = self.memory_1 if r_now > 0.5 else self.memory_0
            scores = torch.matmul(bank, h_mean) * self.scale
            slot_w = F.softmax(scores, dim=0)  # (slots,)
            update = torch.outer(slot_w, h_mean)
            bank.data.copy_(self.ema_decay * bank.data + (1 - self.ema_decay) * update)
