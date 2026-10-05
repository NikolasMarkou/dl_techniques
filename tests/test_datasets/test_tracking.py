"""Tests for the tracking pair pipeline (synthetic only; COCO needs TFDS data)."""

import numpy as np
import pytest

from dl_techniques.datasets.vision.tracking import (
    exemplar_side_for_box,
    mean_pad_crop,
    crop_pair,
    box_iou,
    match_rpn_targets,
    flat_to_grid,
    synthetic_tracking_generator,
    build_siamfc_example,
    build_rpn_example,
)
from dl_techniques.models.vision.dasiamrpn.model import generate_dasiamrpn_anchors


class TestGeometry:
    def test_exemplar_side_paper_value(self):
        # w = h = 100, c = 0.5 -> sqrt(200 * 200) = 200.
        assert exemplar_side_for_box(100.0, 100.0) == pytest.approx(200.0)

    def test_exemplar_side_rejects_degenerate(self):
        with pytest.raises(ValueError):
            exemplar_side_for_box(0.0, 10.0)

    def test_mean_pad_crop_in_bounds_is_identity(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64, 3), dtype=np.float32)
        out = mean_pad_crop(image, 32.0, 32.0, 16)
        np.testing.assert_array_equal(out, image[24:40, 24:40, :])

    def test_mean_pad_crop_out_of_frame_uses_channel_mean(self):
        image = np.ones((16, 16, 3), dtype=np.float32) * 0.25
        out = mean_pad_crop(image, 0.0, 0.0, 16)
        assert out.shape == (16, 16, 3)
        np.testing.assert_allclose(out[0, 0, :], 0.25, atol=0, rtol=0)

    def test_crop_pair_shapes_and_centered_gt(self):
        rng = np.random.default_rng(1)
        image = rng.random((512, 512, 3), dtype=np.float32)
        z, x, gt = crop_pair(image, np.array([256, 256, 100, 80], dtype=np.float32), 127, 255)
        assert z.shape == (127, 127, 3)
        assert x.shape == (255, 255, 3)
        # Anchor frame: centered target decodes to the origin.
        np.testing.assert_allclose(gt[:2], [0.0, 0.0], atol=0, rtol=0)
        assert gt[2] > 0 and gt[3] > 0


class TestMatching:
    def test_perfect_overlap_is_positive_far_is_negative(self):
        anchors = np.array(
            [[0, 0, 10, 10], [100, 100, 10, 10], [3, 0, 10, 10]], dtype=np.float32
        )
        labels, cls_w, deltas, reg_w = match_rpn_targets(
            anchors, np.array([0, 0, 10, 10], dtype=np.float32)
        )
        assert labels[0] == 1 and reg_w[0] == 1.0
        assert labels[1] == 0 and reg_w[1] == 0.0
        # IoU of [3,0,10,10] vs [0,0,10,10] = 70/130 ~ 0.538 -> ignored band.
        assert labels[2] == -1 and cls_w[2] == 0.0

    def test_identity_deltas_are_zero(self):
        anchors = np.array([[5, 5, 20, 20]], dtype=np.float32)
        _, _, deltas, _ = match_rpn_targets(anchors, np.array([5, 5, 20, 20]))
        np.testing.assert_allclose(deltas, 0.0, atol=1e-6, rtol=0)

    def test_unordered_thresholds_raise(self):
        with pytest.raises(ValueError):
            match_rpn_targets(np.zeros((2, 4)), np.zeros(4), pos_iou=0.3, neg_iou=0.6)

    def test_flat_to_grid_anchor_major_layout(self):
        flat = np.arange(2 * 3 * 3).reshape(-1, 1).astype(np.float32)
        grid = flat_to_grid(flat, anchor_num=2, score_size=3)
        assert grid.shape == (3, 3, 2, 1)
        # Anchor-major: rows 0..8 belong to anchor 0.
        np.testing.assert_array_equal(grid[:, :, 0, 0].reshape(-1), np.arange(9))
        np.testing.assert_array_equal(grid[0, 0, :, 0], [0, 9])
        with pytest.raises(ValueError):
            flat_to_grid(np.zeros((10, 1)), anchor_num=2, score_size=3)


class TestBuilders:
    def test_synthetic_generator_deterministic_shapes(self):
        gen_a = synthetic_tracking_generator(4, seed=7)
        gen_b = synthetic_tracking_generator(4, seed=7)
        for (img_a, box_a), (img_b, box_b) in zip(gen_a, gen_b):
            assert img_a.shape == (512, 512, 3)
            np.testing.assert_array_equal(img_a, img_b)
            np.testing.assert_array_equal(box_a, box_b)

    def test_build_siamfc_example(self):
        image, box = next(synthetic_tracking_generator(2, seed=3))
        (z, x), label = build_siamfc_example(
            image, box, score_size=17, augment=False
        )
        assert z.shape == (127, 127, 3) and x.shape == (255, 255, 3)
        assert label.shape == (17, 17, 2)
        assert label[..., 1].sum() > 0  # some valid pixels

    def test_build_rpn_example_has_positives(self):
        image, box = next(synthetic_tracking_generator(4, seed=5))
        anchors = generate_dasiamrpn_anchors(19)
        (z, x), targets = build_rpn_example(
            image, box, anchors, anchor_num=5, score_size=19, augment=False
        )
        assert z.shape == (127, 127, 3) and x.shape == (271, 271, 3)
        assert targets["cls"].shape == (19, 19, 5, 2)
        assert targets["reg"].shape == (19, 19, 5, 5)
        labels = targets["cls"][..., 0].reshape(-1)
        assert int((labels == 1).sum()) > 0  # centered GT matches anchors
        assert int((labels == 0).sum()) > 0
