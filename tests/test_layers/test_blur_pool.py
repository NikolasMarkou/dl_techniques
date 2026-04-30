"""Tests for :class:`dl_techniques.layers.blur_pool.BlurPool2D`."""

from __future__ import annotations

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.blur_pool import BlurPool2D


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def sample_4d() -> tf.Tensor:
    return tf.random.uniform((2, 16, 16, 8), seed=0)


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------


def test_default_init():
    layer = BlurPool2D()
    assert layer.strides == 2
    assert layer.padding == "same"


def test_custom_init():
    layer = BlurPool2D(strides=4, padding="valid")
    assert layer.strides == 4
    assert layer.padding == "valid"


@pytest.mark.parametrize("bad_strides", [0, -1, 1.5, "2"])
def test_invalid_strides_raises(bad_strides):
    with pytest.raises(ValueError):
        BlurPool2D(strides=bad_strides)


def test_invalid_padding_raises():
    with pytest.raises(ValueError):
        BlurPool2D(padding="reflect")


# ---------------------------------------------------------------------
# Build / kernel correctness
# ---------------------------------------------------------------------


def test_kernel_normalised_to_one(sample_4d):
    layer = BlurPool2D(strides=2)
    _ = layer(sample_4d)  # triggers build
    k = keras.ops.convert_to_numpy(layer.kernel)
    assert k.shape == (4, 4, sample_4d.shape[-1], 1)
    # Each per-channel slice sums to 1.
    for c in range(sample_4d.shape[-1]):
        assert np.isclose(k[:, :, c, 0].sum(), 1.0, atol=1e-6)


def test_kernel_is_binomial(sample_4d):
    """Per-channel kernel must equal outer([1,3,3,1]) / 64."""
    layer = BlurPool2D(strides=2)
    _ = layer(sample_4d)
    k = keras.ops.convert_to_numpy(layer.kernel)
    expected = np.outer([1.0, 3.0, 3.0, 1.0], [1.0, 3.0, 3.0, 1.0]) / 64.0
    for c in range(sample_4d.shape[-1]):
        np.testing.assert_allclose(k[:, :, c, 0], expected, atol=1e-6)


def test_kernel_not_trainable(sample_4d):
    layer = BlurPool2D(strides=2)
    _ = layer(sample_4d)
    assert layer.kernel.trainable is False
    assert all(not v.trainable for v in layer.weights)


# ---------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------


@pytest.mark.parametrize("strides,expected_hw", [(1, 16), (2, 8), (4, 4)])
def test_output_shape_same_padding(strides, expected_hw):
    x = tf.random.uniform((2, 16, 16, 6), seed=0)
    layer = BlurPool2D(strides=strides, padding="same")
    y = layer(x)
    assert y.shape == (2, expected_hw, expected_hw, 6)


def test_compute_output_shape_matches_call(sample_4d):
    layer = BlurPool2D(strides=2, padding="same")
    y = layer(sample_4d)
    inferred = layer.compute_output_shape(tuple(sample_4d.shape))
    assert tuple(y.shape) == inferred


def test_channel_preservation():
    x = tf.random.uniform((1, 32, 32, 17), seed=0)
    y = BlurPool2D(strides=2)(x)
    assert y.shape[-1] == 17


# ---------------------------------------------------------------------
# Constant input gives constant output (low-pass property)
# ---------------------------------------------------------------------


def test_constant_interior_unchanged():
    """Kernel sums to 1, so interior (non-border) samples must equal input."""
    x = tf.ones((1, 16, 16, 4), dtype=tf.float32) * 0.7
    y = keras.ops.convert_to_numpy(BlurPool2D(strides=2)(x))
    # 16x16 -> 8x8 with stride=2; border-2 cells in each dim are zero-padded.
    interior = y[:, 2:-2, 2:-2, :]
    np.testing.assert_allclose(
        interior, np.full_like(interior, 0.7), atol=1e-6
    )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_get_config_round_trip():
    layer = BlurPool2D(strides=3, padding="valid")
    cfg = layer.get_config()
    rebuilt = BlurPool2D.from_config(cfg)
    assert rebuilt.strides == 3
    assert rebuilt.padding == "valid"


def test_save_load_in_model(tmp_path):
    inp = keras.layers.Input(shape=(16, 16, 4))
    out = BlurPool2D(strides=2, name="blur")(inp)
    model = keras.Model(inp, out)

    x = np.random.RandomState(0).randn(2, 16, 16, 4).astype(np.float32)
    pre = model.predict(x, verbose=0)

    path = os.path.join(tmp_path, "blur_model.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    post = loaded.predict(x, verbose=0)
    np.testing.assert_allclose(pre, post, atol=1e-6)
