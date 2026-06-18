"""Tests for the homography sampling and warping data-pipeline utilities."""

import numpy as np
import pytest

from dl_techniques.utils.homography import (
    sample_homography,
    warp_image,
    warp_points,
)


class TestSampleHomography:
    """Sampler shape, dtype, determinism, and normalization."""

    def test_shape_and_dtype(self) -> None:
        h = sample_homography((64, 80), rng=np.random.default_rng(0))
        assert h.shape == (3, 3)
        assert h.dtype == np.float32

    def test_bottom_right_normalized(self) -> None:
        for s in range(5):
            h = sample_homography((64, 64), rng=np.random.default_rng(s))
            assert np.isclose(h[2, 2], 1.0, atol=1e-5)

    def test_deterministic_same_seed(self) -> None:
        h1 = sample_homography((64, 64), rng=np.random.default_rng(0))
        h2 = sample_homography((64, 64), rng=np.random.default_rng(0))
        np.testing.assert_allclose(h1, h2, rtol=0, atol=0)

    def test_deterministic_seed_kwarg(self) -> None:
        h1 = sample_homography((48, 96), seed=123)
        h2 = sample_homography((48, 96), seed=123)
        np.testing.assert_allclose(h1, h2, rtol=0, atol=0)

    def test_different_seeds_differ(self) -> None:
        h1 = sample_homography((64, 64), seed=1)
        h2 = sample_homography((64, 64), seed=2)
        assert not np.allclose(h1, h2)

    def test_identity_when_all_ranges_zero(self) -> None:
        # No rotation/scale/translation/perspective/shear => identity.
        h = sample_homography(
            (64, 64),
            seed=0,
            rotation=0.0,
            scale=(1.0, 1.0),
            perspective=0.0,
            translation=0.0,
            shear=0.0,
        )
        np.testing.assert_allclose(h, np.eye(3), atol=1e-5)


class TestWarpImage:
    """Image warp identity, shape, and batch behavior."""

    @pytest.fixture
    def image(self) -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.uniform(0.0, 1.0, size=(32, 40, 3)).astype(np.float32)

    def test_identity_unchanged(self, image: np.ndarray) -> None:
        out = warp_image(image, np.eye(3, dtype=np.float32)).numpy()
        assert out.shape == image.shape
        np.testing.assert_allclose(out, image, atol=1e-4)

    def test_output_size(self, image: np.ndarray) -> None:
        out = warp_image(
            image, np.eye(3, dtype=np.float32), output_size=(16, 20)
        ).numpy()
        assert out.shape == (16, 20, 3)

    def test_batch_path(self, image: np.ndarray) -> None:
        batch = np.stack([image, image * 0.5], axis=0)
        h = np.stack([np.eye(3), np.eye(3)], axis=0).astype(np.float32)
        out = warp_image(batch, h).numpy()
        assert out.shape == batch.shape
        np.testing.assert_allclose(out, batch, atol=1e-4)

    def test_invalid_rank_raises(self) -> None:
        with pytest.raises(ValueError):
            warp_image(np.zeros((5,), dtype=np.float32), np.eye(3))


class TestWarpPoints:
    """Point warp correctness and cross-consistency with warp_image."""

    def test_identity_unchanged(self) -> None:
        pts = np.array([[0.0, 0.0], [10.0, 5.0], [31.0, 31.0]], dtype=np.float32)
        out = warp_points(pts, np.eye(3, dtype=np.float32))
        np.testing.assert_allclose(out, pts, atol=1e-5)

    def test_known_translation(self) -> None:
        # Forward translation H: x' = x + 3, y' = y + 2.
        h = np.array(
            [[1.0, 0.0, 3.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        pts = np.array([[5.0, 5.0], [0.0, 0.0]], dtype=np.float32)
        out = warp_points(pts, h)
        expected = np.array([[8.0, 7.0], [3.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_single_point(self) -> None:
        h = np.array(
            [[1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        out = warp_points(np.array([2.0, 3.0], dtype=np.float32), h)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [6.0, 3.0], atol=1e-5)

    def test_image_point_consistency(self) -> None:
        """A bright pixel must land where warp_points predicts (within 1px).

        This is the convention cross-check: warp_image (which inverts H) and
        warp_points (forward H) must agree on direction.
        """
        h_size = (48, 48)
        img = np.zeros((*h_size, 1), dtype=np.float32)
        # Bright pixel at (x=10, y=12) i.e. row=12, col=10.
        src_x, src_y = 10, 12
        img[src_y, src_x, 0] = 1.0

        # Forward translation: shift right 5, down 7.
        h = np.array(
            [[1.0, 0.0, 5.0], [0.0, 1.0, 7.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        warped = warp_image(img, h, interpolation="nearest").numpy()[..., 0]
        predicted = warp_points(
            np.array([[src_x, src_y]], dtype=np.float32), h
        )[0]
        pred_x, pred_y = predicted[0], predicted[1]

        # Brightest pixel in the warped image.
        max_idx = np.unravel_index(np.argmax(warped), warped.shape)
        got_y, got_x = max_idx[0], max_idx[1]

        assert warped[got_y, got_x] > 0.5, "bright pixel vanished after warp"
        assert abs(got_x - pred_x) <= 1.0, f"x off: got {got_x}, pred {pred_x}"
        assert abs(got_y - pred_y) <= 1.0, f"y off: got {got_y}, pred {pred_y}"

    def test_consistency_under_random_homography(self) -> None:
        """Same cross-check under a sampled (rotation+scale+translation) H."""
        h_size = (64, 64)
        h = sample_homography(h_size, seed=3)

        # Use an interior point so it stays in-frame after warping.
        src_x, src_y = 30, 34
        img = np.zeros((*h_size, 1), dtype=np.float32)
        img[src_y, src_x, 0] = 1.0

        warped = warp_image(img, h, interpolation="nearest").numpy()[..., 0]
        predicted = warp_points(
            np.array([[src_x, src_y]], dtype=np.float32), h
        )[0]
        pred_x, pred_y = predicted[0], predicted[1]

        max_idx = np.unravel_index(np.argmax(warped), warped.shape)
        got_y, got_x = max_idx[0], max_idx[1]

        assert warped[got_y, got_x] > 0.3
        # nearest-interp on a non-integer mapping: allow 1.5px slack.
        assert abs(got_x - pred_x) <= 1.5, f"x off: got {got_x}, pred {pred_x}"
        assert abs(got_y - pred_y) <= 1.5, f"y off: got {got_y}, pred {pred_y}"
