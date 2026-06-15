"""Tests for :class:`dl_techniques.layers.pixel_unshuffle.PixelShuffle2D`.

``PixelShuffle2D`` is the NHWC depth-to-space (pixel-shuffle) layer that
replaces the nonexistent ``keras.layers.DepthToSpace`` (D-002). It must be the
exact inverse of :class:`PixelUnshuffle2D` (space-to-depth).
"""

from __future__ import annotations

import os

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.layers.pixel_unshuffle import PixelShuffle2D, PixelUnshuffle2D


# ---------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------


def test_default_init():
    layer = PixelShuffle2D()
    assert layer.block_size == 2


@pytest.mark.parametrize("bad", [0, -1, 2.5, "2"])
def test_invalid_block_size(bad):
    with pytest.raises(ValueError):
        PixelShuffle2D(block_size=bad)


# ---------------------------------------------------------------------
# Forward shape: (B, H, W, 4C) -> (B, 2H, 2W, C)
# ---------------------------------------------------------------------


def test_shuffle_shape():
    x = tf.random.uniform((2, 8, 8, 12), seed=0)  # 4*C with C=3
    y = PixelShuffle2D(block_size=2)(x)
    assert tuple(y.shape) == (2, 16, 16, 3)


def test_compute_output_shape_matches_call():
    layer = PixelShuffle2D(block_size=2)
    x = tf.random.uniform((2, 8, 8, 12), seed=0)
    y = layer(x)
    inferred = layer.compute_output_shape(tuple(x.shape))
    assert tuple(y.shape) == inferred


# ---------------------------------------------------------------------
# Round-trip correctness vs PixelUnshuffle2D (catches wrong transpose order)
# ---------------------------------------------------------------------


def test_inverse_of_pixel_unshuffle():
    """PixelUnshuffle2D(scale=2)(PixelShuffle2D(2)(y)) must recover y."""
    y = tf.random.uniform((2, 8, 8, 12), seed=42, dtype=tf.float32)
    shuffled = PixelShuffle2D(block_size=2)(y)  # (2, 16, 16, 3)
    recon = PixelUnshuffle2D(scale=2)(shuffled)  # (2, 8, 8, 12)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y),
        keras.ops.convert_to_numpy(recon),
        atol=1e-6,
    )


def test_shuffle_matches_tf_depth_to_space():
    """PixelShuffle2D must match the reference tf.nn.depth_to_space."""
    y = tf.random.uniform((2, 8, 8, 12), seed=7, dtype=tf.float32)
    ours = PixelShuffle2D(block_size=2)(y)
    ref = tf.nn.depth_to_space(y, block_size=2)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(ours),
        keras.ops.convert_to_numpy(ref),
        atol=1e-6,
    )


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


def test_get_config_round_trip():
    layer = PixelShuffle2D(block_size=2)
    cfg = layer.get_config()
    rebuilt = PixelShuffle2D.from_config(cfg)
    assert rebuilt.block_size == 2


def test_save_load(tmp_path):
    inp = keras.layers.Input(shape=(8, 8, 12))
    out = PixelShuffle2D(block_size=2, name="px_shuffle")(inp)
    model = keras.Model(inp, out)

    x = np.random.RandomState(0).randn(2, 8, 8, 12).astype(np.float32)
    pre = model.predict(x, verbose=0)

    path = os.path.join(tmp_path, "px_shuffle_model.keras")
    model.save(path)
    reloaded = keras.models.load_model(path)
    post = reloaded.predict(x, verbose=0)

    np.testing.assert_allclose(pre, post, atol=1e-6)
    assert reloaded.get_layer("px_shuffle").block_size == 2
