"""Tests for the synthetic-shapes renderer (MagicPoint stage-1 data)."""

import numpy as np
import pytest

from dl_techniques.datasets.synthetic_shapes import (
    DUSTBIN_CLASS,
    SHAPE_GENERATORS,
    draw_checkerboard,
    draw_cube,
    draw_ellipses,
    draw_lines,
    draw_polygons,
    draw_stars,
    generate_synthetic_sample,
    keypoints_to_grid_labels,
    synthetic_shapes_generator,
)

H, W = 128, 128


class TestShapeGenerators:
    """Each draw_* returns a valid image + in-bounds keypoints."""

    @pytest.mark.parametrize(
        "fn",
        [draw_checkerboard, draw_lines, draw_polygons, draw_stars, draw_cube, draw_ellipses],
    )
    def test_output_contract(self, fn):
        rng = np.random.default_rng(7)
        img, kps = fn(H, W, rng)

        assert img.shape == (H, W)
        assert img.dtype == np.float32
        assert img.min() >= 0.0 and img.max() <= 1.0

        assert kps.ndim == 2 and kps.shape[1] == 2
        assert kps.dtype == np.float32
        if kps.shape[0] > 0:
            assert np.all(kps[:, 0] >= 0) and np.all(kps[:, 0] <= W - 1)
            assert np.all(kps[:, 1] >= 0) and np.all(kps[:, 1] <= H - 1)

    def test_ellipses_have_no_keypoints(self):
        rng = np.random.default_rng(3)
        _, kps = draw_ellipses(H, W, rng)
        assert kps.shape == (0, 2)

    def test_checkerboard_has_internal_corners(self):
        rng = np.random.default_rng(1)
        _, kps = draw_checkerboard(H, W, rng)
        assert kps.shape[0] > 0

    def test_registry_matches_callables(self):
        assert set(SHAPE_GENERATORS) == {
            "checkerboard", "lines", "polygons", "stars", "cube", "ellipses",
        }


class TestGenerateSample:
    """generate_synthetic_sample shape + determinism."""

    def test_output_shape(self):
        rng = np.random.default_rng(0)
        img, kps = generate_synthetic_sample(H, W, rng)
        assert img.shape == (H, W, 1)
        assert img.dtype == np.float32
        assert img.min() >= 0.0 and img.max() <= 1.0
        assert kps.ndim == 2 and kps.shape[1] == 2

    def test_seeded_determinism(self):
        img_a, kp_a = generate_synthetic_sample(H, W, np.random.default_rng(42))
        img_b, kp_b = generate_synthetic_sample(H, W, np.random.default_rng(42))
        np.testing.assert_array_equal(img_a, img_b)
        np.testing.assert_array_equal(kp_a, kp_b)

    def test_different_seeds_differ(self):
        img_a, _ = generate_synthetic_sample(H, W, np.random.default_rng(1))
        img_b, _ = generate_synthetic_sample(H, W, np.random.default_rng(2))
        assert not np.array_equal(img_a, img_b)

    def test_invalid_shape_types_raises(self):
        with pytest.raises(ValueError):
            generate_synthetic_sample(H, W, np.random.default_rng(0), shape_types=["nope"])

    def test_shape_types_subset(self):
        rng = np.random.default_rng(0)
        img, _ = generate_synthetic_sample(H, W, rng, shape_types=["ellipses"])
        assert img.shape == (H, W, 1)


class TestGridLabels:
    """keypoints_to_grid_labels: range, mapping, dustbin, shape."""

    def test_shape_and_dtype(self):
        labels = keypoints_to_grid_labels(np.zeros((0, 2), np.float32), H, W, cell=8)
        assert labels.shape == (H // 8, W // 8)
        assert labels.dtype == np.int32

    def test_empty_all_dustbin(self):
        labels = keypoints_to_grid_labels(np.zeros((0, 2), np.float32), H, W, cell=8)
        assert np.all(labels == DUSTBIN_CLASS)
        assert DUSTBIN_CLASS == 64

    def test_value_range(self):
        rng = np.random.default_rng(5)
        kps = rng.uniform(0, H - 1, size=(50, 2)).astype(np.float32)
        labels = keypoints_to_grid_labels(kps, H, W, cell=8)
        assert labels.min() >= 0 and labels.max() <= 64

    def test_known_mapping(self):
        # Keypoint at (x=11, y=3): cell (cx=1, cy=0); within-cell col=3, row=3.
        # flattened index = row*8 + col = 3*8 + 3 = 27 at grid (cy=0, cx=1).
        kps = np.array([[11.0, 3.0]], dtype=np.float32)
        labels = keypoints_to_grid_labels(kps, H, W, cell=8)
        assert labels[0, 1] == 27
        # All other cells dustbin.
        mask = np.ones_like(labels, dtype=bool)
        mask[0, 1] = False
        assert np.all(labels[mask] == 64)

    def test_corner_origin_maps_to_zero(self):
        kps = np.array([[0.0, 0.0]], dtype=np.float32)
        labels = keypoints_to_grid_labels(kps, H, W, cell=8)
        assert labels[0, 0] == 0

    def test_nearest_center_chosen_on_collision(self):
        # Two points in cell (0,0): (0,0) far from center, (4,4) near center=3.5.
        kps = np.array([[0.0, 0.0], [4.0, 4.0]], dtype=np.float32)
        labels = keypoints_to_grid_labels(kps, H, W, cell=8)
        # (4,4) -> row*8+col = 4*8+4 = 36 wins over (0,0)->0.
        assert labels[0, 0] == 36

    def test_out_of_bounds_ignored(self):
        kps = np.array([[1000.0, 1000.0], [-5.0, -5.0]], dtype=np.float32)
        labels = keypoints_to_grid_labels(kps, H, W, cell=8)
        assert np.all(labels == 64)

    def test_custom_cell(self):
        labels = keypoints_to_grid_labels(np.zeros((0, 2), np.float32), 64, 64, cell=16)
        assert labels.shape == (4, 4)
        assert np.all(labels == 16 * 16)


class TestGenerator:
    """synthetic_shapes_generator yields the (image, grid_label) contract."""

    def test_yields_finite_count(self):
        gen = synthetic_shapes_generator(H, W, seed=0, n_samples=3)
        items = list(gen)
        assert len(items) == 3
        img, labels = items[0]
        assert img.shape == (H, W, 1)
        assert labels.shape == (H // 8, W // 8)
        assert labels.dtype == np.int32
        assert labels.min() >= 0 and labels.max() <= 64

    def test_seeded_determinism(self):
        a = list(synthetic_shapes_generator(H, W, seed=11, n_samples=2))
        b = list(synthetic_shapes_generator(H, W, seed=11, n_samples=2))
        for (ia, la), (ib, lb) in zip(a, b):
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_array_equal(la, lb)
