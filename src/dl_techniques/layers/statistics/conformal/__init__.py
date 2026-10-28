"""Split-conformal layers: one frozen quantile ``q_hat``, applied three ways.

``ConformalLayer`` is the shared base (the frozen weight, ``set_q_hat``, ``calibrate``).
``ConformalInterval`` widens a point estimate, ``ConformalQuantileHead`` widens learned
quantiles, ``ConformalPredictionSet`` turns classifier logits into a prediction set.
"""

from .base import ConformalLayer
from .interval import ConformalInterval
from .quantile_head import ConformalQuantileHead
from .prediction_set import ConformalPredictionSet, attach_conformal_sets

__all__ = [
    "ConformalLayer",
    "ConformalInterval",
    "ConformalQuantileHead",
    "ConformalPredictionSet",
    "attach_conformal_sets",
]
