"""CASA-RNN: Causal Adaptive State Autoregressive RNN for financial markets."""
from .model import CASARNNModel
from .loss import CounterfactualLoss
from .conformal import ConformalWrapper
from .cpg import CPGEncoder
from .astrocyte import AstrocyteModulator
from .danger import DangerSignalDetector
from .tda_features import TDAFeatureExtractor
from .neuro_modules import (
    NeuromodulatorGating,
    ThalamicAttention,
    HippocampalReplayBuffer,
    PrefrontalWorkingMemory,
    MetaLearningStrategyBank,
    RegimeTransitionDetector,
    CerebellarForwardModel,
    HomeostaticGainControl,
    SoftModuleRouter,
    MODULE_NAMES,
    STRATEGY_NAMES,
)

__all__ = [
    "CASARNNModel",
    "CounterfactualLoss",
    "ConformalWrapper",
    "CPGEncoder",
    "AstrocyteModulator",
    "DangerSignalDetector",
    "TDAFeatureExtractor",
    "NeuromodulatorGating",
    "ThalamicAttention",
    "HippocampalReplayBuffer",
    "PrefrontalWorkingMemory",
    "MetaLearningStrategyBank",
    "RegimeTransitionDetector",
    "CerebellarForwardModel",
    "HomeostaticGainControl",
    "SoftModuleRouter",
    "MODULE_NAMES",
    "STRATEGY_NAMES",
]
