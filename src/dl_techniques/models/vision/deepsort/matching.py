"""Association machinery: costs, cascade, gating, gallery metric.

Transcribed from ``linear_assignment.py``, ``iou_matching.py`` and
``nn_matching.py`` in https://github.com/nwojke/deep_sort. The cascade
prefers recently-seen tracks level by level; infeasible Mahalanobis pairs
are gated to :data:`INFTY_COST`; the appearance gallery keeps the last
``budget`` embeddings per identity and scores the nearest neighbor.

Plain NumPy/SciPy runtime code (``scipy.optimize.linear_sum_assignment``),
no Keras involved.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Callable, Dict, List, Optional, Tuple

from .kalman import CHI2INV95, KalmanFilter

#: Cost assigned to infeasible associations.
INFTY_COST = 1e5

#: Default cosine-distance acceptance threshold (ref app default).
DEFAULT_MATCHING_THRESHOLD = 0.2

#: Default per-identity gallery budget (paper value).
DEFAULT_NN_BUDGET = 100


def min_cost_matching(
    distance_metric: Callable,
    max_distance: float,
    tracks: list,
    detections: list,
    track_indices: Optional[np.ndarray] = None,
    detection_indices: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Solve one linear assignment problem with gating.

    :param distance_metric: Callable ``(tracks, detections, track_rows,
        detection_cols) -> (N, M)`` cost matrix.
    :type distance_metric: callable
    :param max_distance: Associations above this cost are disregarded.
    :type max_distance: float
    :param tracks: Predicted tracks.
    :type tracks: list
    :param detections: Current detections.
    :type detections: list
    :param track_indices: Rows under consideration (defaults to all).
    :type track_indices: numpy.ndarray or None
    :param detection_indices: Columns under consideration (defaults to all).
    :type detection_indices: numpy.ndarray or None
    :return: ``(matches, unmatched_tracks, unmatched_detections)`` with
        global indices.
    :rtype: tuple
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], list(track_indices), list(detection_indices)

    cost_matrix = np.asarray(
        distance_metric(tracks, detections, track_indices, detection_indices),
        dtype=np.float64,
    )
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = np.asarray(linear_sum_assignment(cost_matrix)).T

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(int(detection_idx))
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(int(track_idx))
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(int(track_idx))
            unmatched_detections.append(int(detection_idx))
        else:
            matches.append((int(track_idx), int(detection_idx)))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric: Callable,
    max_distance: float,
    cascade_depth: int,
    tracks: list,
    detections: list,
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match tracks by increasing age, most-recently-seen first.

    Level ``l`` matches tracks with ``time_since_update == 1 + l`` against
    the still-unmatched detections, so a recently-seen track wins a contested
    detection over a long-lost one.

    :param distance_metric: Cost callable (see :func:`min_cost_matching`).
    :type distance_metric: callable
    :param max_distance: Gating threshold.
    :type max_distance: float
    :param cascade_depth: Levels to run; set to the maximum track age.
    :type cascade_depth: int
    :param tracks: Predicted tracks.
    :type tracks: list
    :param detections: Current detections.
    :type detections: list
    :param track_indices: Global track rows under consideration.
    :type track_indices: list or None
    :param detection_indices: Global detection columns under consideration.
    :type detection_indices: list or None
    :return: ``(matches, unmatched_tracks, unmatched_detections)``.
    :rtype: tuple
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = list(detection_indices)
    matches: List[Tuple[int, int]] = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:
            continue
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections,
        )
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    kf: KalmanFilter,
    cost_matrix: np.ndarray,
    tracks: list,
    detections: list,
    track_indices: list,
    detection_indices: list,
    gated_cost: float = INFTY_COST,
    only_position: bool = False,
) -> np.ndarray:
    """Set Mahalanobis-infeasible cost entries to ``gated_cost``.

    :param kf: The Kalman filter.
    :type kf: KalmanFilter
    :param cost_matrix: ``(N, M)`` cost matrix, modified in place.
    :type cost_matrix: numpy.ndarray
    :param tracks: Predicted tracks.
    :type tracks: list
    :param detections: Current detections.
    :type detections: list
    :param track_indices: Global track rows.
    :type track_indices: list
    :param detection_indices: Global detection columns.
    :type detection_indices: list
    :param gated_cost: Replacement cost for infeasible pairs.
    :type gated_cost: float
    :param only_position: Gate on center position only (2 dof, else 4).
    :type only_position: bool
    :return: The modified cost matrix.
    :rtype: numpy.ndarray
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = CHI2INV95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices]
    )
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


def iou(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Intersection over union of one ``(x, y, w, h)`` box vs candidates.

    :param bbox: Single box ``(top-left x, top-left y, width, height)``.
    :type bbox: numpy.ndarray
    :param candidates: Array ``(M, 4)`` in the same format.
    :type candidates: numpy.ndarray
    :return: Array ``(M,)`` of IoU scores in [0, 1].
    :rtype: numpy.ndarray
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]
    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
    ]
    br = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
    ]
    wh = np.maximum(0.0, br - tl)
    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(
    tracks: list,
    detections: list,
    track_indices: Optional[np.ndarray] = None,
    detection_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``1 - IoU`` cost matrix; stale tracks (``time_since_update > 1``) cost inf.

    :param tracks: Predicted tracks.
    :type tracks: list
    :param detections: Current detections.
    :type detections: list
    :param track_indices: Rows under consideration.
    :type track_indices: numpy.ndarray or None
    :param detection_indices: Columns under consideration.
    :type detection_indices: numpy.ndarray or None
    :return: Cost matrix ``(N, M)``.
    :rtype: numpy.ndarray
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = INFTY_COST
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix


def _nn_cosine_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Min cosine distance from each query row of ``y`` to gallery ``x``."""
    distances = 1.0 - np.dot(
        np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True),
        (np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)).T,
    )
    return distances.min(axis=0)


def _nn_euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Min squared Euclidean distance from each query row of ``y`` to ``x``."""
    x, y = np.asarray(x), np.asarray(y)
    r2 = (
        -2.0 * np.dot(x, y.T)
        + np.square(x).sum(axis=1)[:, None]
        + np.square(y).sum(axis=1)[None, :]
    )
    return np.maximum(0.0, r2).min(axis=0)


class NearestNeighborDistanceMetric:
    """Per-identity gallery returning nearest-neighbor appearance distances.

    :param metric: ``"cosine"`` or ``"euclidean"``.
    :type metric: str
    :param matching_threshold: Distances above this are invalid matches.
    :type matching_threshold: float
    :param budget: Max samples kept per identity (oldest evicted); None
        keeps everything.
    :type budget: int or None
    :raises ValueError: On an unknown metric name.
    """

    def __init__(
        self,
        metric: str,
        matching_threshold: float,
        budget: Optional[int] = None,
    ) -> None:
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples: Dict[int, list] = {}

    def partial_fit(
        self, features: np.ndarray, targets: np.ndarray, active_targets: List[int]
    ) -> None:
        """Add observations and drop identities no longer tracked.

        :param features: Array ``(N, M)`` of new embeddings.
        :type features: numpy.ndarray
        :param targets: Identity per row of ``features``.
        :type targets: numpy.ndarray
        :param active_targets: Identities to keep; the rest are forgotten.
        :type active_targets: list
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(int(target), []).append(feature)
            if self.budget is not None:
                self.samples[int(target)] = self.samples[int(target)][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features: np.ndarray, targets: List[int]) -> np.ndarray:
        """Cost matrix ``(len(targets), len(features))`` of nearest distances.

        :param features: Array ``(N, M)`` of query embeddings.
        :type features: numpy.ndarray
        :param targets: Identities to match against (rows).
        :type targets: list
        :return: Cost matrix.
        :rtype: numpy.ndarray
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
