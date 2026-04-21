"""Tests for yolo_decode — DFL decode, anchor generation, per-class NMS."""

import numpy as np
import pytest

from dl_techniques.utils.yolo_decode import (
    decode_dfl_logits,
    decode_predictions,
    dist_to_xyxy,
    make_anchors_np,
    nms_per_class,
)


# ---------------------------------------------------------------------------
# make_anchors_np
# ---------------------------------------------------------------------------


class TestMakeAnchors:
    def test_shape_and_stride_count(self):
        anchors, strides = make_anchors_np((256, 256), [8, 16, 32])
        assert anchors.shape == (32 * 32 + 16 * 16 + 8 * 8, 2)  # 1344
        assert strides.shape == (1344, 1)

    def test_anchors_are_in_pixel_coords(self):
        """First anchor for stride=8: ((0.5*8), (0.5*8)) = (4, 4)."""
        anchors, strides = make_anchors_np((64, 64), [8])
        assert np.isclose(anchors[0, 0], 4.0)
        assert np.isclose(anchors[0, 1], 4.0)
        assert np.isclose(strides[0, 0], 8.0)

    def test_degenerate_stride_raises(self):
        with pytest.raises(ValueError, match="too large"):
            make_anchors_np((16, 16), [32])


# ---------------------------------------------------------------------------
# decode_dfl_logits
# ---------------------------------------------------------------------------


class TestDFL:
    def test_shape(self):
        reg_logits = np.random.randn(2, 100, 4 * 16).astype(np.float32)
        dist = decode_dfl_logits(reg_logits, reg_max=16)
        assert dist.shape == (2, 100, 4)

    def test_sharply_peaked_at_bin_k(self):
        """Putting a massive positive logit on bin k should give expectation ≈ k."""
        reg_max = 8
        reg_logits = np.full((1, 1, 4 * reg_max), -100.0, dtype=np.float32)
        # Set bin 3 to a huge value for each of the 4 edges.
        for edge in range(4):
            reg_logits[0, 0, edge * reg_max + 3] = 100.0
        dist = decode_dfl_logits(reg_logits, reg_max)
        assert np.allclose(dist, 3.0, atol=1e-3)

    def test_uniform_logits_gives_mean_bin(self):
        reg_max = 8
        reg_logits = np.zeros((1, 1, 4 * reg_max), dtype=np.float32)
        dist = decode_dfl_logits(reg_logits, reg_max)
        # E[U{0..7}] = 3.5
        assert np.allclose(dist, 3.5)


# ---------------------------------------------------------------------------
# dist_to_xyxy
# ---------------------------------------------------------------------------


class TestDistToXyxy:
    def test_zero_distance_gives_point_box(self):
        anchors = np.array([[10.0, 20.0]], dtype=np.float32)
        distances = np.zeros((1, 4), dtype=np.float32)
        xyxy = dist_to_xyxy(distances, anchors)
        assert np.allclose(xyxy, [10.0, 20.0, 10.0, 20.0])

    def test_known_distances(self):
        """anchor (50, 50), dists (l=10, t=20, r=30, b=40) → xyxy (40, 30, 80, 90)."""
        anchors = np.array([[50.0, 50.0]], dtype=np.float32)
        distances = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
        xyxy = dist_to_xyxy(distances, anchors)
        assert np.allclose(xyxy, [40.0, 30.0, 80.0, 90.0])


# ---------------------------------------------------------------------------
# decode_predictions — round-trip sanity
# ---------------------------------------------------------------------------


class TestDecodePredictions:
    def test_round_trip_of_known_box(self):
        """Encode a synthetic one-box prediction and verify we recover it after decode."""
        input_shape = (64, 64)
        strides = [8, 16, 32]
        reg_max = 8
        num_classes = 3
        anchors, stride_t = make_anchors_np(input_shape, strides)
        A = anchors.shape[0]

        # Target: box xyxy (12, 12, 28, 28) assigned to anchor closest to center (20, 20).
        anchor_xy = np.array([20.0, 20.0], dtype=np.float32)
        d2 = ((anchors - anchor_xy) ** 2).sum(-1)
        ai = int(np.argmin(d2))
        s = float(stride_t[ai, 0])

        # Distances (in stride units) from this anchor center to (12, 12, 28, 28):
        # l = (ax - 12) / s, t = (ay - 12) / s, r = (28 - ax) / s, b = (28 - ay) / s
        ax, ay = anchors[ai]
        target_dist_units = np.array(
            [(ax - 12) / s, (ay - 12) / s, (28 - ax) / s, (28 - ay) / s],
            dtype=np.float32,
        )
        # Construct logits: one-hot on bins closest to target_dist_units.
        y_pred = np.full((1, A, 4 * reg_max + num_classes), -50.0, dtype=np.float32)
        for edge in range(4):
            bin_k = int(round(target_dist_units[edge]))
            bin_k = max(0, min(bin_k, reg_max - 1))
            y_pred[0, ai, edge * reg_max + bin_k] = 50.0
        # Class 1 positive, others negative.
        y_pred[0, ai, 4 * reg_max + 1] = 10.0

        out = decode_predictions(
            y_pred, anchors, stride_t, reg_max=reg_max,
            num_classes=num_classes, score_threshold=0.5,
        )
        assert len(out) == 1
        boxes, scores, classes = out[0]
        assert boxes.shape[0] >= 1
        # Find the detection on the targeted anchor.
        # The box should be close to (12, 12, 28, 28) within ~1/2 stride rounding error.
        best_box = boxes[np.argmax(scores)]
        assert np.allclose(best_box, [12.0, 12.0, 28.0, 28.0], atol=s)
        assert classes[np.argmax(scores)] == 1

    def test_score_threshold_filters(self):
        input_shape = (32, 32)
        strides = [8]
        reg_max = 4
        num_classes = 2
        anchors, stride_t = make_anchors_np(input_shape, strides)
        A = anchors.shape[0]

        # All logits very negative for cls → no detection passes threshold.
        y_pred = np.full((1, A, 4 * reg_max + num_classes), -50.0, dtype=np.float32)
        out = decode_predictions(
            y_pred, anchors, stride_t, reg_max=reg_max,
            num_classes=num_classes, score_threshold=0.5,
        )
        boxes, scores, classes = out[0]
        assert boxes.shape[0] == 0

    def test_wrong_output_dim_raises(self):
        anchors, stride_t = make_anchors_np((32, 32), [8])
        A = anchors.shape[0]
        # Deliberately wrong last dim.
        y_pred = np.zeros((1, A, 99), dtype=np.float32)
        with pytest.raises(ValueError, match="last dim"):
            decode_predictions(
                y_pred, anchors, stride_t, reg_max=8, num_classes=3,
            )


# ---------------------------------------------------------------------------
# nms_per_class
# ---------------------------------------------------------------------------


class TestNMS:
    def test_empty_input(self):
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.int32)
        kb, ks, kc = nms_per_class(boxes, scores, classes)
        assert kb.shape == (0, 4)
        assert ks.shape == (0,)
        assert kc.shape == (0,)

    def test_duplicate_suppressed(self):
        # Two identical boxes of the same class; NMS keeps only the higher-score.
        boxes = np.array(
            [[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.float32
        )
        scores = np.array([0.9, 0.5], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)
        kb, ks, kc = nms_per_class(boxes, scores, classes, iou_threshold=0.5)
        assert kb.shape == (1, 4)
        assert ks[0] == pytest.approx(0.9)

    def test_different_classes_not_suppressed(self):
        """Same box in different classes should both survive."""
        boxes = np.array(
            [[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 1], dtype=np.int32)
        kb, ks, kc = nms_per_class(boxes, scores, classes, iou_threshold=0.5)
        assert kb.shape == (2, 4)
        assert set(kc.tolist()) == {0, 1}

    def test_non_overlapping_kept(self):
        boxes = np.array(
            [[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32
        )
        scores = np.array([0.9, 0.8], dtype=np.float32)
        classes = np.array([0, 0], dtype=np.int32)
        kb, ks, kc = nms_per_class(boxes, scores, classes, iou_threshold=0.5)
        assert kb.shape == (2, 4)

    def test_max_total_truncates(self):
        rng = np.random.default_rng(0)
        N = 500
        # Random non-overlapping boxes across 10 classes.
        boxes = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            x, y = (i * 13) % 1000, (i * 17) % 1000
            boxes[i] = [x, y, x + 5, y + 5]
        scores = rng.uniform(0.5, 1.0, size=N).astype(np.float32)
        classes = rng.integers(0, 10, size=N).astype(np.int32)
        kb, ks, kc = nms_per_class(
            boxes, scores, classes, max_per_class=1000, max_total=50,
        )
        assert kb.shape == (50, 4)
