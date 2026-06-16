"""Tests for the MothNet building blocks."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.mothnet_blocks import (
    AntennalLobeLayer,
    MushroomBodyLayer,
    HebbianReadoutLayer,
)

B, D = 4, 10


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, D)).astype("float32")


def _roundtrip(layer_cls, kwargs, units_attr_shape, sample, name, tmp_path):
    inp = keras.Input(shape=(D,))
    out = layer_cls(**kwargs, name=name)(inp)
    model = keras.Model(inp, out)
    y0 = model(sample)
    path = os.path.join(tmp_path, f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path, custom_objects={layer_cls.__name__: layer_cls})
    y1 = loaded(sample)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
        rtol=1e-5, atol=1e-5,
    )


class TestAntennalLobeLayer:
    def test_forward_and_shape(self, sample):
        layer = AntennalLobeLayer(units=8)
        out = layer(sample)
        assert tuple(out.shape) == (B, 8)
        assert layer.compute_output_shape((B, D)) == (B, 8)

    def test_serialization(self, sample, tmp_path):
        _roundtrip(AntennalLobeLayer, {"units": 8}, 8, sample, "antennal", tmp_path)

    def test_get_config_round_trip(self):
        layer = AntennalLobeLayer(units=8, inhibition_strength=0.3)
        rebuilt = AntennalLobeLayer.from_config(layer.get_config())
        assert rebuilt.units == 8 and rebuilt.inhibition_strength == 0.3


class TestMushroomBodyLayer:
    def test_forward_and_shape(self, sample):
        layer = MushroomBodyLayer(units=12)
        out = layer(sample)
        assert tuple(out.shape) == (B, 12)
        assert layer.compute_output_shape((B, D)) == (B, 12)

    def test_serialization(self, sample, tmp_path):
        _roundtrip(MushroomBodyLayer, {"units": 12}, 12, sample, "mushroom", tmp_path)


class TestHebbianReadoutLayer:
    def test_forward_and_shape(self, sample):
        layer = HebbianReadoutLayer(units=3)
        out = layer(sample)
        assert tuple(out.shape) == (B, 3)
        assert layer.compute_output_shape((B, D)) == (B, 3)

    def test_serialization(self, sample, tmp_path):
        _roundtrip(HebbianReadoutLayer, {"units": 3}, 3, sample, "hebbian", tmp_path)
