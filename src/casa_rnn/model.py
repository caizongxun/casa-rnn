"""CASARNNModel: full pipeline with optional FeatureGenome + ConformalWrapper."""
import torch
import torch.nn as nn
from typing import Optional
from .genome import FeatureGenome
from .multiscale import MultiScaleCASARNN
from .loss import CounterfactualLoss
from .conformal import ConformalWrapper


class CASARNNModel(nn.Module):
    """
    Full CASA-RNN pipeline:

        raw_input
            |
        FeatureGenome
            |
        CPGEncoder (augments input with oscillator features)
            |
        MultiScaleCASARNN
          |- CASARNNCell (fast/mid/slow) with adaptive stride
          |- SoftModuleRouter  <- learns which bio modules to use
          |    |- ThalamicAttention
          |    |- NeuromodulatorGating + HomeostaticGainControl
          |    |- PrefrontalWorkingMemory
          |    |- MetaLearningStrategyBank
          |    |- RegimeTransitionDetector
          |    |- CerebellarForwardModel
          |    |- AstrocyteModulator
          |    |- DangerSignalDetector
          |    |- TDAFeatureExtractor (Betti numbers)
          |    |- CPG (already in input, weight signals importance)
          |- MemoryBank
          |- UncertaintyGatedHead
            |
        ConformalWrapper (post-hoc calibration)
            |
        (mean, std, extra)
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
        use_bio:       bool  = True,
    ):
        super().__init__()
        self.use_genome = use_genome

        if use_genome:
            self.genome   = FeatureGenome(raw_size, feat_size, top_k_pairs)
            rnn_input     = feat_size
        else:
            rnn_input = raw_size

        self.rnn      = MultiScaleCASARNN(
            input_size=rnn_input,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout=dropout,
            use_memory=use_memory,
            memory_slots=memory_slots,
            use_bio=use_bio,
        )
        self.conformal = ConformalWrapper(coverage=0.90)

    def forward(
        self,
        x_raw: torch.Tensor,
        vol_indicator: Optional[torch.Tensor] = None,
        transition_label: float = 0.0,
        t_offset: int = 0,
        router_entropy_weight: float = 0.02,
    ):
        if self.use_genome:
            with torch.no_grad():
                z0 = self.genome(x_raw, regime=None)
                _, _, extra0 = self.rnn(z0, vol_indicator)
                regime_signal = extra0["regime_probs"].mean(-1, keepdim=True)
            z = self.genome(x_raw, regime=regime_signal.detach())
        else:
            z = x_raw

        return self.rnn(
            z, vol_indicator,
            transition_label=transition_label,
            t_offset=t_offset,
            router_entropy_weight=router_entropy_weight,
        )

    def calibrate_conformal(self, x_cal, vol_cal, y_cal) -> float:
        """Run once after training on a held-out calibration set."""
        self.eval()
        with torch.no_grad():
            means, stds, _ = self.forward(x_cal, vol_cal)
        return self.conformal.calibrate(
            means[:, -1, :], stds[:, -1, :], y_cal[:, -1, :]
        )

    def predict_with_interval(self, x, vol=None):
        """Returns (mean, std, lower_bound, upper_bound) with conformal guarantee."""
        self.eval()
        with torch.no_grad():
            means, stds, extra = self.forward(x, vol)
        lower, upper = self.conformal.predict_interval(means, stds)
        return means, stds, lower, upper, extra

    def get_feature_report(self) -> dict:
        if not self.use_genome:
            return {"genome": "disabled"}
        return self.genome.get_evolved_feature_report()
