"""Track lifecycle state: detections, tracks and their states.

Transcribed from ``detection.py`` and ``track.py`` in
https://github.com/nwojke/deep_sort. Plain data holders; the Kalman math
lives in :mod:`kalman`.
"""

import numpy as np
from typing import Optional


class TrackState:
    """Track lifecycle states (int enum, reference values kept)."""

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Detection:
    """One bounding-box detection with confidence and appearance feature.

    :param tlwh: Box ``(top-left x, top-left y, width, height)``.
    :type tlwh: array-like
    :param confidence: Detector confidence.
    :type confidence: float
    :param feature: Appearance descriptor (any width).
    :type feature: array-like
    """

    def __init__(self, tlwh, confidence: float, feature) -> None:
        self.tlwh = np.asarray(tlwh, dtype=np.float64)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self) -> np.ndarray:
        """Box as ``(min x, min y, max x, max y)``."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Box as ``(center x, center y, aspect, height)``."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class Track:
    """Single-target track: Kalman state plus lifecycle and feature cache.

    :param mean: Initial state mean ``(8,)``.
    :type mean: numpy.ndarray
    :param covariance: Initial state covariance ``(8, 8)``.
    :type covariance: numpy.ndarray
    :param track_id: Unique identifier.
    :type track_id: int
    :param n_init: Consecutive hits before confirmation; a miss while
        tentative deletes the track.
    :type n_init: int
    :param max_age: Consecutive misses before a confirmed track is deleted.
    :type max_age: int
    :param feature: First observed feature (cached when given).
    :type feature: numpy.ndarray or None
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        max_age: int,
        feature: Optional[np.ndarray] = None,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)
        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self) -> np.ndarray:
        """Current box as ``(top-left x, top-left y, width, height)``."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self) -> np.ndarray:
        """Current box as ``(min x, min y, max x, max y)``."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf) -> None:
        """Advance the state distribution one step.

        :param kf: The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection: Detection) -> None:
        """Correct with an associated detection and cache its feature.

        Confirms the track once ``hits >= n_init``.

        :param kf: The Kalman filter.
        :param detection: The associated detection.
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self) -> None:
        """Age a missed track; tentative misses and over-age tracks die."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self) -> bool:
        """Whether the track is unconfirmed."""
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        """Whether the track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self) -> bool:
        """Whether the track is dead and should be removed."""
        return self.state == TrackState.Deleted
