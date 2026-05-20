"""CASARNNModel: full pipeline with optional FeatureGenome."""
import torch
import torch.nn as nn
from typing import Optional
from .genome import FeatureGenome
from .multiscale import MultiScaleCASARNN
from .loss import CounterfactualLoss


class CASARNNModel(nn.Module):
    """
    Full CASA-RNN pipeline:

        raw_input
            |
        FeatureGenome  (auto feature evolution)
            |
        MultiScaleCASARNN
          |- CASARNNCell (fast/mid/slow)
          |- Scale fusion
          |- ThalamicAttention    [NEW: deep integration]
          |- NeuromodulatorGating [NEW: deep integration]
          |- PrefrontalWorkingMemory [NEW: deep integration]
          |- MemoryBank read
          |- UncertaintyGatedHead
            |
        (mean, std, extra)

    Two-stage regime loop:
    - Stage 1: forward pass with uniform regime
    - Stage 2: genome conditioned on Stage-1 regime signal
    """

    def __init__(
        self,
        raw_size:      int,
        hidden_size:   int   = 64,
        output_size:   int   = 1,
        feat_size:     int   = 16,
        use_genome:    bool  = True,
        top_k_pairs:   int   = 8,
        dropout:       float = 0.1,
        use_memory:    bool  = True,
        memory_slots:  int   = 32,
        use_bio:       bool  = True,    # wire bio modules into RNN
    ):
        super().__init__()
        self.use_genome = use_genome

        if use_genome:
            self.genome = FeatureGenome(raw_size, feat_size, top_k_pairs)
            rnn_input = feat_size
        else:
            rnn_input = raw_size

        self.rnn = MultiScaleCASARNN(
            input_size=rnn_input,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            use_memory=use_memory,
            memory_slots=memory_slots,
            use_bio=use_bio,
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        vol_indicator: Optional[torch.Tensor] = None,
    ):
        if self.use_genome:
            with torch.no_grad():
                z0 = self.genome(x_raw, regime=None)
                _, _, extra0 = self.rnn(z0, vol_indicator)
                regime_signal = extra0["regime_probs"].mean(-1, keepdim=True)
            z = self.genome(x_raw, regime=regime_signal.detach())
        else:
            z = x_raw

        return self.rnn(z, vol_indicator)

    def get_feature_report(self) -> dict:
        if not self.use_genome:
            return {"genome": "disabled"}
        return self.genome.get_evolved_feature_report()
