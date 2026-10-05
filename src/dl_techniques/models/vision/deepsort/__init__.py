"""DeepSORT multi-object tracker — public API re-exports.

The learned appearance embedding (:class:`DeepSortAppearanceNet`) plus the
plain-NumPy tracking runtime (Kalman filter, cascade/IoU matching, track
lifecycle, gallery metric, multi-target orchestration).
"""

from .appearance import (
    CosineClassifier,
    DeepSortAppearanceNet,
    DeepSortResidualBlock,
    create_deepsort_embedding,
)
from .kalman import CHI2INV95, KalmanFilter
from .matching import (
    INFTY_COST,
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    iou,
    iou_cost,
    matching_cascade,
    min_cost_matching,
)
from .state import Detection, Track, TrackState
from .tracker import DeepSortTracker

__all__ = [
    "CosineClassifier",
    "DeepSortAppearanceNet",
    "DeepSortResidualBlock",
    "create_deepsort_embedding",
    "CHI2INV95",
    "KalmanFilter",
    "INFTY_COST",
    "NearestNeighborDistanceMetric",
    "gate_cost_matrix",
    "iou",
    "iou_cost",
    "matching_cascade",
    "min_cost_matching",
    "Detection",
    "Track",
    "TrackState",
    "DeepSortTracker",
]
