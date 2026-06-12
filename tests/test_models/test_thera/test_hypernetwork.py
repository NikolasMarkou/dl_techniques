"""Scoped tests for the THERA hypernetwork + implicit-field decoder (step 7).

Covers: arbitrary-scale forward, output-shape tracking the coord grid, `.keras`
round-trip (per-weight compare), config round-trip, a memory-bounded larger
decode, and a coordinate-gradient sanity check (pre-validates the step-9 TV loss).
"""

import os
import tempfile

import keras
import numpy as np
import tensorflow as tf
import pytest

from dl_techniques.layers.grid_sample import make_grid
from dl_techniques.models.thera.hypernetwork import TheraHypernetwork


def _finite(t) -> bool:
    return bool(np.all(np.isfinite(keras.ops.convert_to_numpy(t))))


def _coords(batch: int, hq: int, wq: int):
    """A batched pixel-center query grid via the step-2 make_grid."""
    grid = make_grid((hq, wq))  # (hq, wq, 2) numpy
    g = keras.ops.convert_to_tensor(grid, dtype="float32")
    return keras.ops.broadcast_to(g[None, ...], (batch, hq, wq, 2))


# ---------------------------------------------------------------------
# 1. Forward at an arbitrary scale (query grid 12 differs from source 8).
# ---------------------------------------------------------------------


def test_forward_arbitrary_scale():
    hyper = TheraHypernetwork(hidden_dim=32, out_dim=3)
    encoding = keras.random.normal((2, 8, 8, 16))
    coords = _coords(2, 12, 12)
    t = keras.ops.ones((2, 1))
    out = hyper.decode(encoding, coords, t)
    assert tuple(out.shape) == (2, 12, 12, 3)
    assert _finite(out)


# ---------------------------------------------------------------------
# 2. Output shape tracks the coords grid (non-square).
# ---------------------------------------------------------------------


def test_output_shape_tracks_coords():
    hyper = TheraHypernetwork(hidden_dim=16, out_dim=3)
    encoding = keras.random.normal((2, 8, 8, 16))
    coords = _coords(2, 5, 7)
    t = keras.ops.ones((2, 1))
    out = hyper.decode(encoding, coords, t)
    assert tuple(out.shape) == (2, 5, 7, 3)
    assert _finite(out)


# ---------------------------------------------------------------------
# 3. `.keras` round-trip (per-weight compare, NOT just shapes).
# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class _HyperWrap(keras.Model):
    """Tiny model wrapping TheraHypernetwork with a 3-tuple input."""

    def __init__(self, hidden_dim=32, out_dim=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hyper = TheraHypernetwork(hidden_dim=hidden_dim, out_dim=out_dim)

    def call(self, inputs, training=None):
        return self.hyper(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "out_dim": self.out_dim})
        return config


def test_keras_round_trip():
    model = _HyperWrap(hidden_dim=32, out_dim=3)
    encoding = keras.random.normal((2, 8, 8, 16))
    coords = _coords(2, 10, 10)
    t = keras.ops.ones((2, 1))

    y0 = model((encoding, coords, t))  # dummy forward to build

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hyper.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)

    y1 = reloaded((encoding, coords, t))
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y0),
        keras.ops.convert_to_numpy(y1),
        atol=1e-4,
    )

    # Per-weight survival check (LESSONS iter-1: shapes are not enough).
    hf0 = model.hyper.heat_field
    hf1 = reloaded.hyper.heat_field
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(hf0.components),
        keras.ops.convert_to_numpy(hf1.components),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(hf0.k),
        keras.ops.convert_to_numpy(hf1.k),
        atol=1e-6,
    )
    # out_conv kernel + bias.
    oc0 = model.hyper.out_conv.get_weights()
    oc1 = reloaded.hyper.out_conv.get_weights()
    assert len(oc0) == len(oc1) == 2
    for a, b in zip(oc0, oc1):
        np.testing.assert_allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------
# 4. get_config / from_config round-trip reproduces ctor args.
# ---------------------------------------------------------------------


def test_config_round_trip():
    hyper = TheraHypernetwork(
        hidden_dim=24,
        out_dim=5,
        w0=2.0,
        c=4.0,
        k_init=0.123,
        components_init_scale=8.0,
    )
    cfg = hyper.get_config()
    rebuilt = TheraHypernetwork.from_config(cfg)
    for attr in (
        "hidden_dim",
        "out_dim",
        "w0",
        "c",
        "k_init",
        "components_init_scale",
        "output_size",
    ):
        assert getattr(rebuilt, attr) == getattr(hyper, attr)


# ---------------------------------------------------------------------
# 5. Memory-bounded sanity: a moderately larger decode completes + is finite.
# ---------------------------------------------------------------------


def test_memory_bounded_larger_decode():
    hyper = TheraHypernetwork(hidden_dim=32, out_dim=3)
    encoding = keras.random.normal((1, 16, 16, 32))
    coords = _coords(1, 48, 48)
    t = keras.ops.ones((1, 1))
    out = hyper.decode(encoding, coords, t)
    assert tuple(out.shape) == (1, 48, 48, 3)
    assert _finite(out)


# ---------------------------------------------------------------------
# 6. Gradient sanity: d(sum(decode)) / d(coords) is finite and non-zero.
# ---------------------------------------------------------------------


def test_coord_gradient_finite_nonzero():
    hyper = TheraHypernetwork(hidden_dim=32, out_dim=3)
    encoding = keras.random.normal((2, 8, 8, 16))
    coords = tf.Variable(keras.ops.convert_to_numpy(_coords(2, 10, 10)))
    t = keras.ops.ones((2, 1))

    with tf.GradientTape() as tape:
        tape.watch(coords)
        out = hyper.decode(encoding, coords, t)
        loss = tf.reduce_sum(out)

    grad = tape.gradient(loss, coords)
    assert grad is not None
    g = keras.ops.convert_to_numpy(grad)
    assert np.all(np.isfinite(g))
    assert np.abs(g).sum() > 0.0
