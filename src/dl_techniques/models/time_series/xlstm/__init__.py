"""xLSTM model package (language model + continuous forecaster).

Public API mirrors the ``models/time_series/adaptive_ema`` template: the model
classes and their factory functions are re-exported. Internal callers may still
import from the submodules (``.model`` / ``.forecaster``) directly.

The package exposes two DISTINCT contracts:
- :class:`xLSTM` — language model over integer token inputs ``[B, T]``.
- :class:`xLSTMForecaster` — continuous time-series forecaster ``[B, T, F]``.
"""

from .model import xLSTM, create_xlstm
from .forecaster import xLSTMForecaster, create_xlstm_forecaster

__all__ = [
    "xLSTM",
    "create_xlstm",
    "xLSTMForecaster",
    "create_xlstm_forecaster",
]
