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
        FeatureGenome  (optional, auto feature evolution)
            |
        MultiScaleCASARNN
            |
        (mean, std, extra)

    Two-stage regime loop:
    - Stage 1: forward pass with uniform regime (no prior)
    - Stage 2: use Stage-1 regime signal to condition TemporalGenome
    This allows the genome's lag selection to be regime-aware without
    requiring a separate pretrained regime predictor.
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
        )

    def forward(
        self,
        x_raw: torch.Tensor,
        vol_indicator: Optional[torch.Tensor] = None,
    ):
        """
        Two-stage forward:
        Stage 1 — RNN with raw/uniform features to get initial regime
        Stage 2 — Genome uses Stage-1 regime to condition lag selection
        """
        if self.use_genome:
            # Stage 1: quick pass to get regime signal
            with torch.no_grad():
                z0 = self.genome(x_raw, regime=None)
                _, _, extra0 = self.rnn(z0, vol_indicator)
                regime_signal = extra0["regime_probs"].mean(-1, keepdim=True)  # (B,T,1)

            # Stage 2: genome conditioned on regime
            z = self.genome(x_raw, regime=regime_signal.detach())
        else:
            z = x_raw

        return self.rnn(z, vol_indicator)

    def get_feature_report(self) -> dict:
        """What features has the genome evolved? Call after training."""
        if not self.use_genome:
            return {"genome": "disabled"}
        return self.genome.get_evolved_feature_report()
