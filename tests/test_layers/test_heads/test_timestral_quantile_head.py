"""Tests for the Timestral quantile head."""

import os

import keras
import numpy as np
import pytest

from dl_techniques.layers.heads.time_series.timestral_quantile_head import (
    TimestralQuantileHead,
    monotone_quantiles,
)

LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
D_MODEL, PATCH, ROWS, N = 16, 4, 3, 5


def _head(**kwargs):
    return TimestralQuantileHead(
        d_model=D_MODEL, patch_size=PATCH, quantile_levels=LEVELS, width_factor=2.0, **kwargs)


def _tokens():
    return np.random.default_rng(0).standard_normal((ROWS, N, D_MODEL)).astype("float32")


def test_output_shape_is_one_quantile_vector_per_time_step():
    out = keras.ops.convert_to_numpy(_head()(_tokens()))
    assert out.shape == (ROWS, N * PATCH, len(LEVELS))


def test_quantiles_never_cross():
    out = keras.ops.convert_to_numpy(_head()(_tokens() * 5.0))
    assert np.all(np.diff(out, axis=-1) >= 0)


def test_the_median_channel_is_the_raw_median_output():
    raw = np.random.default_rng(1).standard_normal((2, 6, len(LEVELS))).astype("float32")
    out = keras.ops.convert_to_numpy(monotone_quantiles(raw, LEVELS, 2.0))
    np.testing.assert_array_equal(out[..., 2], raw[..., 2])


def test_the_weight_paths_the_checkpoint_loader_maps_exist():
    head = _head(name="quantile_head")
    head.build((ROWS, N, D_MODEL))
    projection = head.projection
    assert projection.name == "projection"
    for part in ("hidden_layer", "output_layer", "residual_layer"):
        assert getattr(projection, part).kernel is not None
        assert getattr(projection, part).bias is None  # use_bias=False


def test_config_round_trip():
    head = _head()
    clone = TimestralQuantileHead.from_config(head.get_config())
    assert clone.quantile_levels == LEVELS
    assert clone.get_config() == head.get_config()


def test_keras_serialization_round_trip(tmp_path):
    inp = keras.Input(shape=(N, D_MODEL))
    model = keras.Model(inp, _head(name="th")(inp))
    x = _tokens()
    before = keras.ops.convert_to_numpy(model(x))
    path = os.path.join(tmp_path, "th.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(loaded(x)), before, rtol=1e-5, atol=1e-5)
