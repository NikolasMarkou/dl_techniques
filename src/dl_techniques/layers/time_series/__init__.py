"""
Time series layers.

This package holds the layers used to build forecasting and sequence models.
It covers four families: N-BEATS blocks, xLSTM cells and blocks, the PRISM
wavelet tree, and the DeepAR scale layer and cell. Around those sit fusion and
lag-attention layers, and a few signal-processing utilities. The forecasting
heads (quantile and likelihood heads) live in ``layers/heads/time_series``.

There is no factory here. Unlike ``layers/ffn`` and ``layers/attention``, this
package has no ``create_*_layer`` dispatcher, so import the class you want by
name.

The public surface is 29 names. ``__all__`` at the bottom of this file is the
list of record; read it there rather than from a copy that can drift. The
package README says what each name is for and shows runnable examples.

Everything exported is a Keras layer except one: ``create_manokhin_compliant_model``
is a builder that returns a full ``keras.Model``.
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
# Fusion & DeepAR Blocks
# ---------------------------------------------------------------------

from .adaptive_lag_attention import AdaptiveLagAttentionLayer
from .temporal_fusion import TemporalFusionLayer
from .deepar_blocks import (
    DeepARCell,
    ScaleLayer
)
from .residual_projection_block import ResidualProjectionBlock

# ---------------------------------------------------------------------
# Scientific Forecasting & Signal Processing
# ---------------------------------------------------------------------

from .forecasting_layers import (
    NaiveResidual,
    ForecastabilityGate,
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

    # Fusion & DeepAR Blocks
    "AdaptiveLagAttentionLayer",
    "TemporalFusionLayer",
    "DeepARCell",
    "ScaleLayer",
    "ResidualProjectionBlock",

    # Scientific & Signal Processing
    "NaiveResidual",
    "ForecastabilityGate",
    "create_manokhin_compliant_model",
    "ExponentialMovingAverage",
    "EMASlopeFilter",
]