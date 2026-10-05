"""Multi-target tracker: cascade + IoU association with lifecycle management.

Transcribed from ``tracker.py`` in https://github.com/nwojke/deep_sort.
Each step predicts all tracks, associates confirmed tracks by gated
appearance through the matching cascade, associates the leftovers (plus
unconfirmed tracks) by IoU, initiates tracks from orphan detections, and
refreshes the appearance gallery from confirmed tracks. One deliberate,
documented difference from the paper text: the association cost is
appearance-only with Mahalanobis gating (what the reference code does),
not the ``lambda``-blended motion+appearance cost the paper describes.

Plain runtime orchestration over NumPy state -- no Keras, no weights.
"""

import numpy as np
from typing import List, Tuple

from .kalman import KalmanFilter
from .matching import (
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    iou_cost,
    matching_cascade,
    min_cost_matching,
)
from .state import Detection, Track

#: Defaults transcribed from the reference tracker + demo app.
DEFAULT_MAX_IOU_DISTANCE = 0.7
DEFAULT_MAX_AGE = 30
DEFAULT_N_INIT = 3


class DeepSortTracker:
    """Online multi-target tracker over appearance-bearing detections.

    :param metric: Gallery appearance metric (cosine, threshold 0.2,
        budget 100 in the reference setup).
    :type metric: NearestNeighborDistanceMetric
    :param max_iou_distance: IoU-fallback acceptance threshold.
    :type max_iou_distance: float
    :param max_age: Consecutive misses before a confirmed track dies.
    :type max_age: int
    :param n_init: Consecutive hits before a tentative track confirms.
    :type n_init: int
    """

    def __init__(
        self,
        metric: NearestNeighborDistanceMetric,
        max_iou_distance: float = DEFAULT_MAX_IOU_DISTANCE,
        max_age: int = DEFAULT_MAX_AGE,
        n_init: int = DEFAULT_N_INIT,
    ) -> None:
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1

    def predict(self) -> None:
        """Propagate every track one step. Call before :meth:`update`."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections: List[Detection]) -> None:
        """Associate detections, manage the track set, refresh the gallery.

        :param detections: Current-frame detections.
        :type detections: list of Detection
        """
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(
        self, detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Cascade confirmed tracks by appearance, leftovers by IoU."""
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices,
            )

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection: Detection) -> None:
        """Start a tentative track from an orphan detection."""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean, covariance, self._next_id, self.n_init, self.max_age,
                detection.feature,
            )
        )
        self._next_id += 1
