"""
Astrocyte slow-wave modulator.

Astrocytes are non-neuronal glial cells that modulate synaptic gain
on timescales of seconds-to-minutes — slower than NE/DA spikes but
faster than hormonal changes. In finance this maps to hour-scale
liquidity cycles.

Mechanism:
  a_t = alpha * a_{t-1} + (1-alpha) * tanh(W_a h_t)   [slow EMA]
  h_out = h * (1 + beta * a_t)                         [multiplicative gain]

beta and alpha are LEARNABLE so the model decides how much astrocyte
modulation to apply (alpha near 1 -> very slow; alpha near 0 -> off).
"""
import torch
import torch.nn as nn


class AstrocyteModulator(nn.Module):
    def __init__(self, hidden_size: int, init_alpha: float = 0.995, init_beta: float = 0.1):
        super().__init__()
        self.proj  = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm  = nn.LayerNorm(hidden_size)
        # learnable scalars (unconstrained; sigmoid/clamp applied in forward)
        self.log_alpha = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_alpha)).item()))
        self.log_beta  = nn.Parameter(torch.tensor(torch.log(torch.tensor(init_beta)).item()))
        self.register_buffer("state", torch.zeros(1, hidden_size))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)          # (0,1)

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp().clamp(0.0, 1.0)   # (0,1]

    def reset(self, batch: int, device: torch.device) -> None:
        self.state = torch.zeros(batch, self.state.shape[-1], device=device)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, hidden_size)  — called per timestep inside the RNN loop.
        Returns modulated h of same shape.
        """
        B = h.shape[0]
        if self.state.shape[0] != B:
            self.reset(B, h.device)
        self.state = self.alpha * self.state.detach() + (1 - self.alpha) * torch.tanh(self.proj(h))
        h_mod = h * (1.0 + self.beta * self.state)
        return self.norm(h_mod)
