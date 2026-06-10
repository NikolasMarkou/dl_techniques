"""Adaptive EMA slope-filter model package.

Public API mirrors the ``models/vit`` template: the model class and the
factory function are re-exported. Internal callers may still import from
``dl_techniques.models.adaptive_ema.model`` directly.
"""

from .model import (
    AdaptiveEMASlopeFilterModel,
    create_adaptive_ema_slope_filter,
)

__all__ = [
    "AdaptiveEMASlopeFilterModel",
    "create_adaptive_ema_slope_filter",
]
