"""Tests for the Siamese tracking losses."""

import keras
import numpy as np
import pytest

from dl_techniques.losses import (
    SiamFCLogisticLoss,
    DaSiamRPNClsLoss,
    DaSiamRPNRegLoss,
    create_siamfc_label,
)


class TestCreateSiamfcLabel:
    def test_shape_and_dtypes(self):
        label = create_siamfc_label(17)
        assert label.shape == (17, 17, 2)
        assert label.dtype == np.float32

    def test_center_positive_far_negative_ring_ignored(self):
        label = create_siamfc_label(17, pos_radius_px=25.0, neg_radius_px=50.0)
        # Center: dist 0 -> positive.
        assert label[8, 8, 0] == 1.0 and label[8, 8, 1] == 1.0
        # (8, 12): dist 32px -> ignored ring.
        assert label[8, 12, 1] == 0.0
        # Corner (0, 0): dist ~90.5px -> negative.
        assert label[0, 0, 0] == 0.0 and label[0, 0, 1] == 1.0

    def test_matches_reference_convention(self):
        # Inline transcription of the reference create_BCELogit_loss_label.
        size, pos_thr, neg_thr, stride = 17, 25.0, 50.0, 8
        pos_thr, neg_thr = pos_thr / stride, neg_thr / stride
        center = (size - 1) / 2
        line = np.arange(0, size) - center
        line = line**2
        line = np.expand_dims(line, axis=0)
        dist_map = line + line.transpose()
        ref = np.zeros([size, size, 2]).astype(np.float32)
        ref[:, :, 0] = dist_map <= pos_thr**2
        ref[:, :, 1] = (dist_map <= pos_thr**2) | (dist_map > neg_thr**2)
        np.testing.assert_array_equal(create_siamfc_label(size), ref)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            create_siamfc_label(0)
        with pytest.raises(ValueError):
            create_siamfc_label(17, pos_radius_px=50.0, neg_radius_px=25.0)


class TestSiamFCLogisticLoss:
    def test_per_sample_shape_not_scalar(self):
        loss_fn = SiamFCLogisticLoss()
        y_true = np.stack([create_siamfc_label(17)] * 3, axis=0)
        y_pred = np.zeros((3, 17, 17, 1), dtype="float32")
        out = loss_fn.call(y_true, y_pred)
        assert tuple(out.shape) == (3,)

    def test_confident_correct_is_small_and_wrong_is_large(self):
        loss_fn = SiamFCLogisticLoss()
        label = create_siamfc_label(17)
        y_true = np.stack([label] * 2, axis=0)
        signed = (2.0 * label[..., 0:1] - 1.0).astype("float32")
        good = loss_fn.call(y_true, 5.0 * signed[np.newaxis].repeat(2, axis=0))
        bad = loss_fn.call(y_true, -5.0 * signed[np.newaxis].repeat(2, axis=0))
        assert bool(np.all(np.asarray(good) < 0.1))
        assert bool(np.all(np.asarray(bad) > 4.0))

    def test_fully_ignored_sample_is_zero_not_nan(self):
        loss_fn = SiamFCLogisticLoss()
        y_true = np.zeros((2, 17, 17, 2), dtype="float32")
        y_pred = np.zeros((2, 17, 17, 1), dtype="float32")
        out = np.asarray(loss_fn.call(y_true, y_pred))
        assert bool(np.all(out == 0.0))

    def test_masked_row_does_not_leak_into_sibling(self):
        loss_fn = SiamFCLogisticLoss()
        label = create_siamfc_label(17)
        y_true = np.stack([label, label], axis=0)
        y_true[0, :, :, 1] = 0.0  # ignore everything in row 0
        rng = np.random.RandomState(0)
        y_pred = rng.randn(2, 17, 17, 1).astype("float32")
        out = np.asarray(loss_fn.call(y_true, y_pred))
        assert out[0] == 0.0
        solo = np.asarray(
            loss_fn.call(y_true[1:2], y_pred[1:2])
        )
        # Pure absolute bound (rtol=0): batch-of-2 vs batch-of-1 summation
        # order jitters at ~1e-7; the claim is row independence, not bitwise
        # identity across batch sizes.
        np.testing.assert_allclose(out[1], solo[0], atol=1e-6, rtol=0)

    def test_config_round_trip(self):
        loss_fn = SiamFCLogisticLoss()
        rebuilt = SiamFCLogisticLoss.from_config(loss_fn.get_config())
        assert isinstance(rebuilt, SiamFCLogisticLoss)


class TestDaSiamRPNClsLoss:
    def _packed(self, batch=2, score=5, anchors=3, seed=0):
        rng = np.random.RandomState(seed)
        labels = rng.randint(0, 2, (batch, score, score, anchors)).astype("float32")
        weight = np.ones((batch, score, score, anchors), dtype="float32")
        y_true = np.stack([labels, weight], axis=-1)
        y_pred = rng.randn(batch, score, score, anchors, 2).astype("float32")
        return y_true, y_pred

    def test_per_sample_shape(self):
        y_true, y_pred = self._packed()
        out = DaSiamRPNClsLoss().call(y_true, y_pred)
        assert tuple(out.shape) == (2,)

    def test_perfect_logits_near_zero(self):
        y_true, _ = self._packed()
        labels = y_true[..., 0]
        # class-1 logit high where label==1 and vice versa.
        logits = np.where(
            labels[..., np.newaxis] == 1,
            np.array([-10.0, 10.0], dtype="float32"),
            np.array([10.0, -10.0], dtype="float32"),
        )
        out = np.asarray(DaSiamRPNClsLoss().call(y_true, logits.astype("float32")))
        assert bool(np.all(out < 1e-3))

    def test_ignored_anchors_do_not_contribute(self):
        y_true, y_pred = self._packed()
        y_true[..., 1] = 0.0  # ignore everything
        out = np.asarray(DaSiamRPNClsLoss().call(y_true, y_pred))
        assert bool(np.all(out == 0.0))

    def test_ignored_minus_one_labels_do_not_nan(self):
        # -1 labels must never reach the sparse lookup (graph-mode NaN).
        y_true, y_pred = self._packed()
        y_true[..., 0] = -1.0
        y_true[..., 1] = 0.0
        out = np.asarray(DaSiamRPNClsLoss().call(y_true, y_pred))
        assert bool(np.all(out == 0.0))

    def test_flat_model_layout_matches_grouped(self):
        y_true, y_pred = self._packed()
        flat = y_pred.reshape(y_pred.shape[0], 5, 5, 3 * 2)
        grouped = np.asarray(DaSiamRPNClsLoss().call(y_true, y_pred))
        from_flat = np.asarray(DaSiamRPNClsLoss().call(y_true, flat))
        np.testing.assert_allclose(from_flat, grouped, atol=1e-6, rtol=0)

    def test_config_round_trip(self):
        rebuilt = DaSiamRPNClsLoss.from_config(DaSiamRPNClsLoss().get_config())
        assert isinstance(rebuilt, DaSiamRPNClsLoss)


class TestDaSiamRPNRegLoss:
    def _packed(self, batch=2, score=5, anchors=3, seed=1):
        rng = np.random.RandomState(seed)
        deltas = rng.randn(batch, score, score, anchors, 4).astype("float32")
        weight = (rng.rand(batch, score, score, anchors) > 0.5).astype("float32")
        y_true = np.concatenate([deltas, weight[..., np.newaxis]], axis=-1)
        y_pred = rng.randn(batch, score, score, anchors, 4).astype("float32")
        return y_true, y_pred

    def test_per_sample_shape(self):
        y_true, y_pred = self._packed()
        out = DaSiamRPNRegLoss().call(y_true, y_pred)
        assert tuple(out.shape) == (2,)

    def test_perfect_deltas_are_zero(self):
        y_true, _ = self._packed()
        out = np.asarray(DaSiamRPNRegLoss().call(y_true, y_true[..., :4]))
        assert bool(np.all(out == 0.0))

    def test_negative_anchors_with_garbage_deltas_are_free(self):
        y_true, y_pred = self._packed()
        y_true[..., 4] = 0.0  # no positives anywhere
        y_pred = 1e6 * np.ones_like(y_pred)
        out = np.asarray(DaSiamRPNRegLoss().call(y_true, y_pred))
        assert bool(np.all(out == 0.0))

    def test_flat_model_layout_matches_grouped(self):
        y_true, y_pred = self._packed()
        flat = y_pred.reshape(y_pred.shape[0], 5, 5, 3 * 4)
        grouped = np.asarray(DaSiamRPNRegLoss().call(y_true, y_pred))
        from_flat = np.asarray(DaSiamRPNRegLoss().call(y_true, flat))
        np.testing.assert_allclose(from_flat, grouped, atol=1e-6, rtol=0)

    def test_huber_knee(self):
        loss_fn = DaSiamRPNRegLoss(huber_delta=1.0)
        y_true = np.zeros((1, 1, 1, 1, 5), dtype="float32")
        y_true[..., 4] = 1.0
        y_true[..., 0] = 0.5  # |d| = 0.5 target vs 0 pred -> 0.5*d^2 mean
        out = float(loss_fn.call(y_true, np.zeros((1, 1, 1, 1, 4), dtype="float32"))[0])
        np.testing.assert_allclose(out, 0.5 * 0.25 / 4.0, atol=1e-6, rtol=0)

    def test_config_round_trip_keeps_delta(self):
        rebuilt = DaSiamRPNRegLoss.from_config(
            DaSiamRPNRegLoss(huber_delta=2.0).get_config()
        )
        assert rebuilt.huber_delta == 2.0
