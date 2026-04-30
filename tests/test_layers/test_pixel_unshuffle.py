"""Tests for :class:`dl_techniques.layers.pixel_unshuffle.PixelUnshuffle2D`."""

from __future__ import annotations

import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.pixel_unshuffle import PixelUnshuffle2D


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------


def test_default_init():
    layer = PixelUnshuffle2D()
    assert layer.scale == 2
    assert layer.out_channels is None
    assert layer.proj is None


def test_with_projection_init():
    layer = PixelUnshuffle2D(scale=2, out_channels=96)
    assert layer.scale == 2
    assert layer.out_channels == 96
    assert layer.proj is not None


@pytest.mark.parametrize("bad", [0, -1, 2.5, "2"])
def test_invalid_scale(bad):
    with pytest.raises(ValueError):
        PixelUnshuffle2D(scale=bad)


@pytest.mark.parametrize("bad", [0, -3, 1.5])
def test_invalid_out_channels(bad):
    with pytest.raises(ValueError):
        PixelUnshuffle2D(out_channels=bad)


# ---------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------


def test_unshuffle_shape_no_proj():
    x = tf.random.uniform((2, 16, 16, 6), seed=0)
    y = PixelUnshuffle2D(scale=2)(x)
    assert tuple(y.shape) == (2, 8, 8, 24)


def test_unshuffle_shape_with_proj():
    x = tf.random.uniform((2, 16, 16, 6), seed=0)
    y = PixelUnshuffle2D(scale=2, out_channels=12)(x)
    assert tuple(y.shape) == (2, 8, 8, 12)


@pytest.mark.parametrize("scale,h_out", [(1, 16), (2, 8), (4, 4)])
def test_scale_factor(scale, h_out):
    x = tf.random.uniform((1, 16, 16, 4), seed=0)
    y = PixelUnshuffle2D(scale=scale)(x)
    assert tuple(y.shape) == (1, h_out, h_out, 4 * scale ** 2)


def test_compute_output_shape_matches_call():
    layer = PixelUnshuffle2D(scale=2, out_channels=24)
    x = tf.random.uniform((2, 16, 16, 6), seed=0)
    y = layer(x)
    inferred = layer.compute_output_shape(tuple(x.shape))
    assert tuple(y.shape) == inferred


# ---------------------------------------------------------------------
# Lossless property: pair with PixelShuffle (depth-to-space) inverse
# ---------------------------------------------------------------------


def test_inverse_via_depth_to_space():
    """Unshuffle then depth-to-space (tf.nn.depth_to_space) recovers input."""
    x = tf.random.uniform((2, 8, 8, 3), seed=42, dtype=tf.float32)
    layer = PixelUnshuffle2D(scale=2)
    y = layer(x)  # (2, 4, 4, 12)
    x_recon = tf.nn.depth_to_space(y, block_size=2)  # (2, 8, 8, 3)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(x),
        keras.ops.convert_to_numpy(x_recon),
        atol=1e-6,
    )


def test_unshuffle_known_pattern():
    """Manual check: scale=2 must arrange (i,j)-block pixels along channel axis."""
    # Single 2x2 image with a single channel
    x = tf.constant(
        [[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=tf.float32
    )  # shape (1, 2, 2, 1)
    y = PixelUnshuffle2D(scale=2)(x)  # (1, 1, 1, 4)
    arr = keras.ops.convert_to_numpy(y)[0, 0, 0, :]
    # Expected order from reshape((1, 1, 2, 1, 2, 1)) -> transpose (0,1,3,2,4,5):
    # row-major over (h_block=0..1, w_block=0..1, c=0): 1, 2, 3, 4
    np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_get_config_round_trip_no_proj():
    layer = PixelUnshuffle2D(scale=2)
    cfg = layer.get_config()
    rebuilt = PixelUnshuffle2D.from_config(cfg)
    assert rebuilt.scale == 2
    assert rebuilt.out_channels is None


def test_get_config_round_trip_with_proj():
    layer = PixelUnshuffle2D(scale=2, out_channels=24, use_bias=False)
    cfg = layer.get_config()
    rebuilt = PixelUnshuffle2D.from_config(cfg)
    assert rebuilt.scale == 2
    assert rebuilt.out_channels == 24
    assert rebuilt.use_bias is False


def test_save_load_with_projection(tmp_path):
    inp = keras.layers.Input(shape=(16, 16, 6))
    out = PixelUnshuffle2D(scale=2, out_channels=24, name="px")(inp)
    model = keras.Model(inp, out)

    x = np.random.RandomState(0).randn(2, 16, 16, 6).astype(np.float32)
    pre = model.predict(x, verbose=0)

    path = os.path.join(tmp_path, "px_model.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    post = loaded.predict(x, verbose=0)
    np.testing.assert_allclose(pre, post, atol=1e-6)


def test_save_load_without_projection(tmp_path):
    inp = keras.layers.Input(shape=(16, 16, 6))
    out = PixelUnshuffle2D(scale=2, name="px")(inp)
    model = keras.Model(inp, out)

    x = np.random.RandomState(0).randn(2, 16, 16, 6).astype(np.float32)
    pre = model.predict(x, verbose=0)

    path = os.path.join(tmp_path, "px_no_proj.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    post = loaded.predict(x, verbose=0)
    np.testing.assert_allclose(pre, post, atol=1e-6)
