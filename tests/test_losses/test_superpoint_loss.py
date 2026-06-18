"""Tests for SuperPoint detector + descriptor losses."""

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.losses.superpoint_loss import (
    SuperPointDetectorLoss,
    SuperPointDescriptorLoss,
)


# ---------------------------------------------------------------------
# Detector loss
# ---------------------------------------------------------------------


class TestDetectorLoss:
    @pytest.fixture
    def labels(self):
        rng = np.random.default_rng(0)
        return ops.convert_to_tensor(
            rng.integers(0, 65, size=(2, 8, 8)).astype("int32")
        )

    @pytest.fixture
    def logits(self):
        return keras.random.normal((2, 8, 8, 65), seed=1)

    def test_finite_scalar(self, labels, logits):
        loss = SuperPointDetectorLoss()
        value = loss(labels, logits)
        value_np = ops.convert_to_numpy(value)
        assert value_np.shape == ()
        assert np.isfinite(value_np)
        assert value_np > 0.0

    def test_gradient_flows(self, labels):
        import tensorflow as tf

        loss = SuperPointDetectorLoss()
        logits = tf.Variable(keras.random.normal((2, 8, 8, 65), seed=2))
        with tf.GradientTape() as tape:
            value = loss(labels, logits)
        grad = tape.gradient(value, logits)
        assert grad is not None
        assert np.isfinite(ops.convert_to_numpy(grad)).all()
        assert np.abs(ops.convert_to_numpy(grad)).sum() > 0.0

    def test_perfect_prediction_near_zero(self, labels):
        # One-hot huge logit at the true class -> near-zero CE.
        labels_np = ops.convert_to_numpy(labels)
        one_hot = np.eye(65, dtype="float32")[labels_np]  # (2,8,8,65)
        logits = ops.convert_to_tensor(one_hot * 50.0)
        loss = SuperPointDetectorLoss()
        value = ops.convert_to_numpy(loss(labels, logits))
        assert value < 1e-3

    def test_config_round_trip(self):
        loss = SuperPointDetectorLoss(name="det")
        config = loss.get_config()
        rebuilt = SuperPointDetectorLoss.from_config(config)
        assert rebuilt.name == "det"
        # functional equivalence on a toy input
        labels = ops.convert_to_tensor(
            np.random.default_rng(3).integers(0, 65, size=(2, 8, 8)).astype("int32")
        )
        logits = keras.random.normal((2, 8, 8, 65), seed=4)
        a = ops.convert_to_numpy(loss(labels, logits))
        b = ops.convert_to_numpy(rebuilt(labels, logits))
        assert np.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------
# Descriptor loss
# ---------------------------------------------------------------------


def _l2norm(x):
    return x / (ops.sqrt(ops.sum(ops.square(x), axis=-1, keepdims=True)) + 1e-8)


class TestDescriptorLoss:
    @pytest.fixture
    def desc(self):
        d1 = _l2norm(keras.random.normal((2, 8, 8, 16), seed=10))
        d2 = _l2norm(keras.random.normal((2, 8, 8, 16), seed=11))
        return d1, d2

    @pytest.fixture
    def diagonal_corr(self):
        n = 64
        eye = ops.eye(n, dtype="float32")
        return ops.broadcast_to(ops.expand_dims(eye, 0), (2, n, n))

    def test_compute_finite_scalar(self, desc, diagonal_corr):
        d1, d2 = desc
        loss = SuperPointDescriptorLoss()
        value = ops.convert_to_numpy(loss.compute(d1, d2, diagonal_corr))
        assert value.shape == ()
        assert np.isfinite(value)
        assert value >= 0.0

    def test_identical_descriptors_low_loss(self, diagonal_corr):
        # Identical desc1 == desc2 with diagonal correspondence: positive pairs
        # have similarity 1.0 == positive_margin -> hinge(1.0 - 1.0) = 0.
        d = _l2norm(keras.random.normal((2, 8, 8, 16), seed=12))
        loss = SuperPointDescriptorLoss()
        same = ops.convert_to_numpy(loss.compute(d, d, diagonal_corr))
        # Compare against a mismatched pair, which should be larger.
        d_other = _l2norm(keras.random.normal((2, 8, 8, 16), seed=13))
        diff = ops.convert_to_numpy(loss.compute(d, d_other, diagonal_corr))
        assert same < diff
        # Positive-pair term should be ~0 for identical descriptors; remaining
        # loss is only the (small) negative term over off-diagonal pairs.
        assert same < 0.5

    def test_gradient_flows(self, diagonal_corr):
        import tensorflow as tf

        loss = SuperPointDescriptorLoss()
        d1 = tf.Variable(_l2norm(keras.random.normal((2, 8, 8, 16), seed=14)))
        d2 = tf.Variable(_l2norm(keras.random.normal((2, 8, 8, 16), seed=15)))
        with tf.GradientTape() as tape:
            value = loss.compute(d1, d2, diagonal_corr)
        grads = tape.gradient(value, [d1, d2])
        assert all(g is not None for g in grads)
        for g in grads:
            assert np.isfinite(ops.convert_to_numpy(g)).all()

    def test_call_default_path(self, desc):
        d1, d2 = desc
        loss = SuperPointDescriptorLoss()
        value = ops.convert_to_numpy(loss(d2, d1))  # call(y_true=d2, y_pred=d1)
        assert value.shape == ()
        assert np.isfinite(value)
        assert value >= 0.0

    def test_config_round_trip(self):
        loss = SuperPointDescriptorLoss(
            positive_margin=0.9, negative_margin=0.1, lambda_d=2.0, name="desc"
        )
        config = loss.get_config()
        rebuilt = SuperPointDescriptorLoss.from_config(config)
        assert rebuilt.positive_margin == 0.9
        assert rebuilt.negative_margin == 0.1
        assert rebuilt.lambda_d == 2.0
        assert rebuilt.name == "desc"

    def test_lambda_d_negative_raises(self):
        with pytest.raises(ValueError):
            SuperPointDescriptorLoss(lambda_d=-1.0)
