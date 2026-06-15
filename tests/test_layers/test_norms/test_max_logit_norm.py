"""
Test suite for the MaxLogit normalization family: MaxLogitNorm, DecoupledMaxLogit,
DMLPlus (focal + center).

Primary regression target: the `ops.squeeze(ops.max(norm, axis=-1), axis=-1)` bug
(plan_2026-06-15_2485b951 / B1). `ops.max` over a keepdims=True norm already drops
the reduced axis, so the extra squeeze(axis=-1) targeted the batch dimension and
crashed for any batch > 1. These tests exercise batch > 1 explicitly so the bug
cannot regress silently.

Output contracts:
- MaxLogitNorm:        single tensor, shape == input (L2-normalized logits).
- DecoupledMaxLogit:   tuple (combined, max_cosine, max_norm), each shape == input[:-1].
- DMLPlus(focal):      single tensor, shape == input[:-1] (max cosine).
- DMLPlus(center):     tuple (max_norm, norm) -> shapes (input[:-1], input[:-1] + (1,)).
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.norms.max_logit_norm import (
    MaxLogitNorm,
    DecoupledMaxLogit,
    DMLPlus,
)


@pytest.fixture
def logits_2d() -> np.ndarray:
    """Batch of 8 (>1, to expose the squeeze bug) with 10 classes."""
    np.random.seed(0)
    return np.random.randn(8, 10).astype(np.float32)


class TestB1SqueezeRegression:
    """Pins the B1 squeeze crash: forward must succeed for batch > 1."""

    def test_maxlogitnorm_forward(self, logits_2d):
        y = MaxLogitNorm()(ops.convert_to_tensor(logits_2d))
        assert tuple(y.shape) == (8, 10)

    def test_decoupled_max_logit_forward(self, logits_2d):
        combined, max_cos, max_norm = DecoupledMaxLogit()(
            ops.convert_to_tensor(logits_2d)
        )
        # B1: each is per-sample scalar -> shape (8,), NOT a crash.
        assert tuple(combined.shape) == (8,)
        assert tuple(max_cos.shape) == (8,)
        assert tuple(max_norm.shape) == (8,)

    def test_dmlplus_focal_forward(self, logits_2d):
        y = DMLPlus(model_type="focal")(ops.convert_to_tensor(logits_2d))
        assert tuple(y.shape) == (8,)

    def test_dmlplus_center_forward(self, logits_2d):
        max_norm, norm = DMLPlus(model_type="center")(
            ops.convert_to_tensor(logits_2d)
        )
        assert tuple(max_norm.shape) == (8,)
        assert tuple(norm.shape) == (8, 1)

    def test_max_norm_value_matches_l2_norm(self, logits_2d):
        """max_norm should equal the per-sample L2 norm (single class axis)."""
        _, _, max_norm = DecoupledMaxLogit(epsilon=1e-12)(
            ops.convert_to_tensor(logits_2d)
        )
        expected = np.sqrt((logits_2d ** 2).sum(axis=-1))
        np.testing.assert_allclose(
            ops.convert_to_numpy(max_norm), expected, atol=1e-4
        )


class TestSerialization:
    """get_config / from_config and full .keras model round-trip."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: MaxLogitNorm(epsilon=1e-6),
            lambda: DecoupledMaxLogit(constant=0.8),
            lambda: DMLPlus(model_type="focal"),
            lambda: DMLPlus(model_type="center"),
        ],
    )
    def test_config_roundtrip(self, factory):
        layer = factory()
        rebuilt = type(layer).from_config(layer.get_config())
        assert rebuilt.get_config() == layer.get_config()

    def test_keras_model_roundtrip(self, logits_2d):
        x = ops.convert_to_tensor(logits_2d)
        inp = keras.Input(shape=(10,))
        out = DecoupledMaxLogit()(inp)
        model = keras.Model(inp, out)
        path = os.path.join(tempfile.mkdtemp(), "dml.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        for a, b in zip(model(x), reloaded(x)):
            np.testing.assert_allclose(
                ops.convert_to_numpy(a), ops.convert_to_numpy(b), atol=1e-6
            )


class TestGraphSafety:
    """No eager ops: layers must trace under @tf.function with symbolic shape."""

    def test_decoupled_graph_trace(self):
        layer = DecoupledMaxLogit()

        @tf.function(input_signature=[tf.TensorSpec([None, 10], tf.float32)])
        def fn(x):
            return layer(x)

        out = fn(tf.random.normal((5, 10)))
        assert tuple(out[0].shape) == (5,)

    def test_dmlplus_center_graph_trace(self):
        layer = DMLPlus(model_type="center")

        @tf.function(input_signature=[tf.TensorSpec([None, 10], tf.float32)])
        def fn(x):
            return layer(x)

        max_norm, norm = fn(tf.random.normal((7, 10)))
        assert tuple(max_norm.shape) == (7,)
