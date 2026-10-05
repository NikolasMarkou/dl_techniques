"""Kalman filter for image-space bounding-box tracking (DeepSORT transcription).

An 8-dim state ``(x, y, a, h, vx, vy, va, vh)`` -- center, aspect, height
plus velocities -- follows a constant-velocity model with the box location
as direct linear observation. Motion/observation uncertainty scales with the
current height estimate. Transcribed from ``deep_sort/kalman_filter.py`` in
https://github.com/nwojke/deep_sort, including the chi-square gating table.

This is plain NumPy/SciPy runtime code, not a Keras layer: there is nothing
to learn and nothing to build.
"""

import numpy as np
import scipy.linalg
from typing import Tuple

#: 0.95 quantiles of the chi-square distribution (MATLAB/Octave chi2inv),
#: used as Mahalanobis gating thresholds.
CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """Constant-velocity Kalman filter over ``(x, y, a, h)`` box states.

    Uncertainty weights are relative to the current height estimate (the
    reference calls this hacky; it is transcribed unchanged).
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create a track from an unassociated measurement.

        :param measurement: Box ``(x, y, a, h)``.
        :type measurement: numpy.ndarray
        :return: ``(mean (8,), covariance (8, 8))``; velocities start at 0.
        :rtype: tuple
        """
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict one step forward.

        :param mean: State mean ``(8,)``.
        :type mean: numpy.ndarray
        :param covariance: State covariance ``(8, 8)``.
        :type covariance: numpy.ndarray
        :return: Predicted ``(mean, covariance)``.
        :rtype: tuple
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project the state distribution to measurement space.

        :param mean: State mean ``(8,)``.
        :type mean: numpy.ndarray
        :param covariance: State covariance ``(8, 8)``.
        :type covariance: numpy.ndarray
        :return: Projected ``(mean (4,), covariance (4, 4))``.
        :rtype: tuple
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the prediction with a measurement.

        :param mean: Predicted mean ``(8,)``.
        :type mean: numpy.ndarray
        :param covariance: Predicted covariance ``(8, 8)``.
        :type covariance: numpy.ndarray
        :param measurement: Box ``(x, y, a, h)``.
        :type measurement: numpy.ndarray
        :return: Corrected ``(mean, covariance)``.
        :rtype: tuple
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = np.asarray(measurement, dtype=np.float64) - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Squared Mahalanobis distances to N measurements.

        :param mean: State mean ``(8,)``.
        :type mean: numpy.ndarray
        :param covariance: State covariance ``(8, 8)``.
        :type covariance: numpy.ndarray
        :param measurements: Array ``(N, 4)`` of ``(x, y, a, h)`` boxes.
        :type measurements: numpy.ndarray
        :param only_position: Gate on center position only (2 dof).
        :type only_position: bool
        :return: Array ``(N,)`` of squared distances.
        :rtype: numpy.ndarray
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = np.asarray(measurements)[:, :2]
        cholesky_factor = np.linalg.cholesky(covariance)
        d = np.asarray(measurements, dtype=np.float64) - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        return np.sum(z * z, axis=0)
