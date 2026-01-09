"""
Time Series Layers Module.

This module exports a comprehensive suite of specialized layers and blocks
for time series forecasting, signal processing, and sequence modeling.
It includes implementations of state-of-the-art architectures (N-BEATS, xLSTM, DeepAR),
probabilistic output heads, and scientific forecasting utilities.
"""

# ---------------------------------------------------------------------
# Architecture Blocks
# ---------------------------------------------------------------------

from .mixed_sequential_block import MixedSequentialBlock
from .temporal_convolutional_network import TemporalConvNet, TemporalBlock
from .prism_blocks import (
    PRISMLayer,
    PRISMTimeTree,
    PRISMNode,
    FrequencyBandRouter,
    FrequencyBandStatistics
)

# ---------------------------------------------------------------------
# N-BEATS Family
# ---------------------------------------------------------------------

from .nbeats_blocks import (
    NBeatsBlock,
    GenericBlock,
    TrendBlock,
    SeasonalityBlock
)
from .nbeatsx_blocks import ExogenousBlock

# ---------------------------------------------------------------------
# xLSTM Family
# ---------------------------------------------------------------------

from .xlstm_blocks import (
    sLSTMCell,
    sLSTMLayer,
    sLSTMBlock,
    mLSTMCell,
    mLSTMLayer,
    mLSTMBlock
)

# ---------------------------------------------------------------------
# Forecasting Heads & Fusion
# ---------------------------------------------------------------------

from .adaptive_lag_attention import AdaptiveLagAttentionLayer
from .temporal_fusion import TemporalFusionLayer
from .quantile_head_fixed_io import QuantileHead
from .quantile_head_variable_io import QuantileSequenceHead
from .deepar_blocks import (
    GaussianLikelihoodHead,
    NegativeBinomialLikelihoodHead,
    DeepARCell,
    ScaleLayer
)

# ---------------------------------------------------------------------
# Scientific Forecasting & Signal Processing
# ---------------------------------------------------------------------

from .forecasting_layers import (
    NaiveResidual,
    ForecastabilityGate,
    ConformalQuantileHead,
    create_manokhin_compliant_model
)
from .ema_layer import (
    ExponentialMovingAverage,
    EMASlopeFilter
)


# ---------------------------------------------------------------------
# Export public interface
# ---------------------------------------------------------------------

__all__ = [
    # Architecture Blocks
    "MixedSequentialBlock",
    "TemporalConvNet",
    "TemporalBlock",
    "PRISMLayer",
    "PRISMTimeTree",
    "PRISMNode",
    "FrequencyBandRouter",
    "FrequencyBandStatistics",

    # N-BEATS Family
    "NBeatsBlock",
    "GenericBlock",
    "TrendBlock",
    "SeasonalityBlock",
    "ExogenousBlock",

    # xLSTM Family
    "sLSTMCell",
    "sLSTMLayer",
    "sLSTMBlock",
    "mLSTMCell",
    "mLSTMLayer",
    "mLSTMBlock",

    # Forecasting Heads & Fusion
    "AdaptiveLagAttentionLayer",
    "TemporalFusionLayer",
    "QuantileHead",
    "QuantileSequenceHead",
    "GaussianLikelihoodHead",
    "NegativeBinomialLikelihoodHead",
    "DeepARCell",
    "ScaleLayer",

    # Scientific & Signal Processing
    "NaiveResidual",
    "ForecastabilityGate",
    "ConformalQuantileHead",
    "create_manokhin_compliant_model",
    "ExponentialMovingAverage",
    "EMASlopeFilter",
]