"""Time-series models — public API re-exports.

A family of independent forecasting architectures, one subpackage each:
`deepar/` (autoregressive probabilistic), `nbeats/` (basis expansion, plus the
exogenous `nbeatsx` variant), `prism/`, `tirex/`, `mdn/` (mixture density
heads), `xlstm/` (extended LSTM, with a dedicated forecaster wrapper) and
`adaptive_ema/`. They share the `ForecastMixin` in `forecast.py` but no
backbone, so there is no single variant table across the family.
"""
from .adaptive_ema.model import (
    AdaptiveEMASlopeFilterModel,
    create_adaptive_ema_slope_filter,
)
from .deepar.model import DeepAR, create_deepar
from .mdn.model import MDNModel, create_mdn_model
from .nbeats.nbeats import NBeatsNet, create_nbeats_model
from .nbeats.nbeatsx import NBeatsXNet, create_nbeatsx_model
from .prism.model import PRISMModel, create_prism_model
from .tirex.model import TiRexCore, create_tirex_by_variant, create_tirex_model
from .xlstm.forecaster import xLSTMForecaster, create_xlstm_forecaster
from .xlstm.model import xLSTM, create_xlstm

__all__ = [
    "AdaptiveEMASlopeFilterModel",
    "DeepAR",
    "MDNModel",
    "NBeatsNet",
    "NBeatsXNet",
    "PRISMModel",
    "TiRexCore",
    "create_adaptive_ema_slope_filter",
    "create_deepar",
    "create_mdn_model",
    "create_nbeats_model",
    "create_nbeatsx_model",
    "create_prism_model",
    "create_tirex_by_variant",
    "create_tirex_model",
    "create_xlstm",
    "create_xlstm_forecaster",
    "xLSTM",
    "xLSTMForecaster",
]
