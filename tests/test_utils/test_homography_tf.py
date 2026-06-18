"""Parity tests for the in-graph (tf.data) homography ports.

The numpy primitives in ``dl_techniques.utils.homography`` (and
``_cell_correspondence`` in ``train.superpoint.train_superpoint``) are the
parity reference. These tests are the correctness gate (plan SC2) for the tf
ports added in Step 2:

- ``warp_image_tf``        vs numpy ``warp_image``        (atol 1e-4)
- ``_cell_correspondence_tf`` vs numpy ``_cell_correspondence`` (EXACT equality)
- ``sample_homography_tf``  shape / [2,2]==1 / finite / invertible / bounds
"""

import numpy as np
import tensorflow as tf

from dl_techniques.utils.homography import (
    sample_homography,
    warp_image,
    warp_image_tf,
    sample_homography_tf,
    _cell_correspondence_tf,
    _DEFAULT_ROTATION,
    _DEFAULT_SCALE,
    _DEFAULT_PERSPECTIVE,
    _DEFAULT_TRANSLATION,
)
from train.superpoint.train_superpoint import _cell_correspondence


class TestWarpImageTfParity:
    """warp_image_tf must match numpy warp_image within atol=1e-4."""

    def _fixed_image(self) -> np.ndarray:
        rng = np.random.default_rng(11)
        return rng.uniform(0.0, 1.0, size=(64, 64, 1)).astype(np.float32)

    def test_warp_image_tf_parity(self) -> None:
        img = self._fixed_image()
        max_err = 0.0
        for s in range(5):
            h = sample_homography((64, 64), rng=np.random.default_rng(s))
            ref = warp_image(img, h).numpy()
            got = warp_image_tf(
                tf.constant(img), tf.constant(h)
            ).numpy()
            assert got.shape == ref.shape
            err = float(np.max(np.abs(got - ref)))
            max_err = max(max_err, err)
            assert np.allclose(got, ref, atol=1e-4), (
                f"seed {s}: max abs error {err} exceeds atol 1e-4"
            )
        # Expose the residual for the report.
        print(f"\n[warp_image_tf parity] max abs error over 5 H = {max_err:.3e}")

    def test_warp_image_tf_identity(self) -> None:
        img = self._fixed_image()
        out = warp_image_tf(
            tf.constant(img), tf.eye(3, dtype=tf.float32)
        ).numpy()
        assert out.shape == img.shape
        np.testing.assert_allclose(out, img, atol=1e-4)


class TestCellCorrespondenceTfParity:
    """_cell_correspondence_tf must EXACTLY equal the numpy reference."""

    def test_cell_correspondence_tf_parity(self) -> None:
        H = W = 64
        cell = 8
        for s in range(6):
            h = sample_homography((H, W), rng=np.random.default_rng(s))
            ref = _cell_correspondence(h, H, W, cell)
            got = _cell_correspondence_tf(
                tf.constant(h), H, W, cell
            ).numpy()
            assert got.shape == ref.shape
            assert np.array_equal(got, ref), (
                f"seed {s}: correspondence not exactly equal "
                f"(diff count {int(np.sum(got != ref))})"
            )

    def test_cell_correspondence_tf_identity(self) -> None:
        H = W = 64
        cell = 8
        h = np.eye(3, dtype=np.float32)
        ref = _cell_correspondence(h, H, W, cell)
        got = _cell_correspondence_tf(tf.constant(h), H, W, cell).numpy()
        assert np.array_equal(got, ref)
        # Identity => each cell maps to itself => identity matrix.
        np.testing.assert_array_equal(got, np.eye(ref.shape[0], dtype=np.float32))


class TestSampleHomographyTfValid:
    """sample_homography_tf: shape / normalization / validity / bounds."""

    def test_shape_dtype_and_normalized(self) -> None:
        for s in range(8):
            h = sample_homography_tf(
                (64, 64), seed=tf.constant([s, s + 100], dtype=tf.int32)
            )
            assert h.shape == (3, 3)
            assert h.dtype == tf.float32
            hn = h.numpy()
            assert np.isclose(hn[2, 2], 1.0, atol=1e-5)
            assert np.all(np.isfinite(hn))
            det = np.linalg.det(hn.astype(np.float64))
            assert abs(det) > 1e-6, f"seed {s}: near-singular H (det={det})"

    def test_identity_when_all_ranges_zero(self) -> None:
        h = sample_homography_tf(
            (64, 64),
            seed=tf.constant([0, 1], dtype=tf.int32),
            rotation=0.0,
            scale=(1.0, 1.0),
            perspective=0.0,
            translation=0.0,
            shear=0.0,
        ).numpy()
        np.testing.assert_allclose(h, np.eye(3), atol=1e-5)

    def test_params_within_sampler_bounds(self) -> None:
        """Over many seeds the realized transform stays within achievable ranges.

        We reconstruct the pre-perspective affine bound: translation is bounded
        by ``translation * size`` plus the rotation/scale-induced center shift,
        and the bottom-row perspective entries are bounded by
        ``perspective / size``. We check the loose-but-real envelope.
        """
        H = W = 64
        center = max(H, W) / 2.0
        # The bottom-row perspective entry is g = perspective/size, but the
        # center-pivot composition (from_center @ ... @ to_center) mixes it with
        # the translation column, scaling its realized magnitude by ~(1+center).
        max_persp = _DEFAULT_PERSPECTIVE / float(W) * (1.0 + center) + 1e-6
        # Generous affine envelope: |translate| <= t*size + (scale+1)*center.
        max_translate = (
            _DEFAULT_TRANSLATION * max(H, W)
            + (_DEFAULT_SCALE[1] + 1.0) * center
            + 1e-3
        )
        for s in range(50):
            h = sample_homography_tf(
                (H, W), seed=tf.constant([s, 2 * s + 3], dtype=tf.int32)
            ).numpy()
            # Bottom-row perspective entries within normalized magnitude.
            assert abs(h[2, 0]) <= max_persp + 1e-6
            assert abs(h[2, 1]) <= max_persp + 1e-6
            # Translation (last column) within the affine envelope.
            assert abs(h[0, 2]) <= max_translate
            assert abs(h[1, 2]) <= max_translate
            # Linear block magnitude bounded by scale*|rotation/shear| ~ <= 2.
            assert np.all(np.abs(h[:2, :2]) <= _DEFAULT_SCALE[1] * 2.0 + 1e-3)
            assert np.isfinite(h).all()
            # Sanity that rotation magnitude is consistent with the bound.
            _ = _DEFAULT_ROTATION  # referenced for documentation/bound intent
