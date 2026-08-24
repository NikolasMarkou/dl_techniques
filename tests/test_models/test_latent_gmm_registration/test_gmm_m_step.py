"""The GMM M-step must return component means, not means scaled by N.

Guard for C-40 (plan-2026-08-14T233721-d4f9beb2, step 31). Before the fix,
``compute_gmm_params`` divided the responsibility-weighted sum by ``pi``, the
per-component MEAN responsibility ``(1/N)*sum_i gamma_ik``, where the M-step (and
the function's own docstring) require the SUM ``sum_i gamma_ik``. Every returned
``mu`` was therefore exactly ``num_points`` times too large, and
``compute_rigid_transform``'s translation -- ``t = c_target - R*c_source`` --
inherited the same factor, so supervised training chased a ``loss_t`` target it
could not reach.

Expected values here are derived from the M-step definition
(``mu_k = sum_i gamma_ik x_i / sum_i gamma_ik``), never read back out of the
implementation.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.latent_gmm_registration.model import (
    compute_gmm_params,
    compute_rigid_transform,
)


def _t(arr):
    return keras.ops.convert_to_tensor(np.asarray(arr, dtype="float32"))


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


class TestComputeGmmParams:
    def test_uniform_responsibility_gives_the_plain_centroid(self):
        """gamma == 1/K for every point: each component mean IS the centroid.

        At HEAD this returned N * centroid (N = 64 here).
        """
        rng = np.random.default_rng(0)
        num_points, num_gaussians = 64, 4
        points = rng.normal(size=(1, num_points, 3)).astype("float32")
        gamma = np.full((1, num_points, num_gaussians), 1.0 / num_gaussians, "float32")

        pi, mu = compute_gmm_params(_t(points), _t(gamma))

        centroid = points[0].mean(axis=0)  # M-step value for a uniform gamma
        mu_np = _np(mu)
        assert mu_np.shape == (1, num_gaussians, 3)
        for k in range(num_gaussians):
            np.testing.assert_allclose(mu_np[0, k], centroid, rtol=1e-4, atol=1e-5)

        # pi is the MIXING coefficient and is deliberately left as the mean.
        np.testing.assert_allclose(
            _np(pi)[0], np.full(num_gaussians, 1.0 / num_gaussians), rtol=1e-5, atol=1e-6
        )

    def test_hard_split_gives_each_halfs_own_centroid(self):
        """Distinct per-component means, so a global rescale cannot hide."""
        rng = np.random.default_rng(1)
        num_points = 32
        points = rng.normal(size=(1, num_points, 3)).astype("float32")
        gamma = np.zeros((1, num_points, 2), "float32")
        gamma[0, : num_points // 2, 0] = 1.0
        gamma[0, num_points // 2 :, 1] = 1.0

        _pi, mu = compute_gmm_params(_t(points), _t(gamma))
        mu_np = _np(mu)

        np.testing.assert_allclose(
            mu_np[0, 0], points[0, : num_points // 2].mean(axis=0), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            mu_np[0, 1], points[0, num_points // 2 :].mean(axis=0), rtol=1e-4, atol=1e-5
        )

    def test_mu_is_scale_free_in_the_point_count(self):
        """Anti-vacuity for the N-factor specifically: duplicating a cloud leaves
        every component mean unchanged. The pre-fix code scaled with N, so this
        alone separates the two implementations."""
        rng = np.random.default_rng(2)
        points = rng.normal(size=(1, 16, 3)).astype("float32")
        doubled = np.concatenate([points, points], axis=1)

        gamma_a = np.full((1, 16, 3), 1.0 / 3.0, "float32")
        gamma_b = np.full((1, 32, 3), 1.0 / 3.0, "float32")

        _, mu_a = compute_gmm_params(_t(points), _t(gamma_a))
        _, mu_b = compute_gmm_params(_t(doubled), _t(gamma_b))

        np.testing.assert_allclose(_np(mu_a), _np(mu_b), rtol=1e-4, atol=1e-5)


class TestRigidTransformInheritsTheFix:
    def test_pure_translation_is_recovered_at_its_true_magnitude(self):
        """target = source + t, matched components: t_est must equal t.

        At HEAD, mu was N x too large on both sides, so t_est came back at
        N * t (N = 24 here) -- the quantity supervised training could not reach.
        """
        rng = np.random.default_rng(3)
        num_points, num_gaussians = 24, 6
        source = rng.normal(size=(1, num_points, 3)).astype("float32")
        translation = np.array([[0.3, -0.7, 1.25]], dtype="float32")
        target = source + translation[:, None, :]

        # A non-degenerate gamma: soft but component-distinguishing.
        logits = rng.normal(size=(1, num_points, num_gaussians)).astype("float32") * 3.0
        gamma = np.exp(logits - logits.max(axis=-1, keepdims=True))
        gamma /= gamma.sum(axis=-1, keepdims=True)
        gamma = gamma.astype("float32")

        pi_x, mu_x = compute_gmm_params(_t(source), _t(gamma))
        pi_y, mu_y = compute_gmm_params(_t(target), _t(gamma))

        R, t = compute_rigid_transform(mu_x, pi_x, mu_y, pi_y)

        np.testing.assert_allclose(_np(R)[0], np.eye(3), rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(_np(t)[0], translation[0], rtol=1e-3, atol=1e-3)
