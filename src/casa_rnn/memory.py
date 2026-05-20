"""Sparse-Write External Memory Bank for CASA-RNN."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """
    External differentiable memory with sparse write.

    Write condition: regime_prob > threshold  (only write on structural breaks)
    Read: soft attention over all memory slots using current h_fast as query

    This gives the model access to historical regime states -- e.g. it can
    "remember" what the last crisis looked like and compare to now.

    Args:
        hidden_size: dimension of hidden state and memory slots
        num_slots:   number of memory slots (default 32)
        write_threshold: regime probability above which a slot is written
    """

    def __init__(self, hidden_size: int, num_slots: int = 32, write_threshold: float = 0.65):
        super().__init__()
        self.hidden_size     = hidden_size
        self.num_slots       = num_slots
        self.write_threshold = write_threshold
        self.scale           = hidden_size ** -0.5

        # Persistent memory slots (learned initialization)
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_size) * 0.02)

        # Query projection
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Output gate: blend memory read with h_fast
        self.output_gate = nn.Linear(hidden_size * 2, hidden_size)

    def read(self, h_fast: torch.Tensor) -> torch.Tensor:
        """
        Soft attention read.
        Args:
            h_fast: (batch, hidden_size)
        Returns:
            mem_out: (batch, hidden_size)
        """
        q      = self.query_proj(h_fast)                      # (B, H)
        scores = torch.matmul(q, self.memory.T) * self.scale  # (B, num_slots)
        attn   = F.softmax(scores, dim=-1)                    # (B, num_slots)
        mem_out = torch.matmul(attn, self.memory)             # (B, H)

        # Blend: gate controls how much memory vs. h_fast to use
        gate    = torch.sigmoid(self.output_gate(torch.cat([h_fast, mem_out], dim=-1)))
        return gate * mem_out + (1 - gate) * h_fast

    def write(self, h_fast: torch.Tensor, regime_prob: torch.Tensor):
        """
        Sparse write: only update a memory slot when regime_prob > threshold.
        Uses a soft weighted update to stay differentiable.

        Args:
            h_fast:      (batch, hidden_size)
            regime_prob: (batch,)  scalar regime probability per sample
        """
        should_write = (regime_prob.mean() > self.write_threshold)
        if not should_write:
            return

        with torch.no_grad():
            # Write to the slot most similar to current h_fast (soft nearest)
            h_mean  = h_fast.mean(0).detach()                 # (H,)
            scores  = torch.matmul(self.memory, h_mean)       # (num_slots,)
            slot_w  = F.softmax(scores / self.scale, dim=0)   # (num_slots,)
            # Exponential moving average update
            update  = torch.outer(slot_w, h_mean)             # (num_slots, H)
            self.memory.data = 0.95 * self.memory.data + 0.05 * update
