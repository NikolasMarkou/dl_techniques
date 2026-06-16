"""Tests for the io_preparation normalization / clipping layers."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.io_preparation import (
    ClipLayer,
    NormalizationLayer,
    DenormalizationLayer,
    TensorPreprocessingLayer,
)

B, D = 4, 6


@pytest.fixture
def sample():
    return (np.random.default_rng(0).uniform(0, 255, size=(B, D))).astype("float32")


def _roundtrip(layer, data, name, tmp_path):
    inp = keras.Input(shape=(D,))
    out = layer(inp)
    model = keras.Model(inp, out)
    y0 = model(data)
    path = os.path.join(tmp_path, f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    y1 = loaded(data)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1), atol=1e-6
    )


class TestClipLayer:
    def test_invalid_range(self):
        with pytest.raises(ValueError):
            ClipLayer(clip_min=1.0, clip_max=0.0)

    def test_forward(self, sample):
        out = keras.ops.convert_to_numpy(ClipLayer(clip_min=10.0, clip_max=100.0)(sample))
        assert out.min() >= 10.0 and out.max() <= 100.0

    def test_compute_output_shape(self):
        assert ClipLayer(0.0, 1.0).compute_output_shape((B, D)) == (B, D)

    def test_serialization(self, sample, tmp_path):
        _roundtrip(ClipLayer(clip_min=0.0, clip_max=200.0, name="clip"), sample, "clip", tmp_path)


class TestNormalizationLayer:
    def test_invalid_ranges(self):
        with pytest.raises(ValueError):
            NormalizationLayer(source_min=5.0, source_max=1.0)
        with pytest.raises(ValueError):
            NormalizationLayer(target_min=1.0, target_max=0.0)

    def test_forward_range(self, sample):
        out = keras.ops.convert_to_numpy(
            NormalizationLayer(0.0, 255.0, -0.5, 0.5)(sample)
        )
        assert out.min() >= -0.5 - 1e-6 and out.max() <= 0.5 + 1e-6

    def test_serialization(self, sample, tmp_path):
        _roundtrip(NormalizationLayer(name="norm"), sample, "norm", tmp_path)


class TestDenormalizationLayer:
    def test_forward(self, sample):
        norm = NormalizationLayer(0.0, 255.0, -0.5, 0.5)(sample)
        denorm = DenormalizationLayer(-0.5, 0.5, 0.0, 255.0)(norm)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(denorm), sample, atol=1e-3
        )

    def test_serialization(self, tmp_path):
        data = np.random.default_rng(1).uniform(-0.5, 0.5, size=(B, D)).astype("float32")
        _roundtrip(DenormalizationLayer(name="denorm"), data, "denorm", tmp_path)


class TestTensorPreprocessingLayer:
    def test_forward_with_clipping(self, sample):
        layer = TensorPreprocessingLayer(enable_final_clipping=True,
                                         final_clip_min=-0.4, final_clip_max=0.4)
        out = keras.ops.convert_to_numpy(layer(sample))
        assert out.min() >= -0.4 - 1e-6 and out.max() <= 0.4 + 1e-6

    def test_compute_output_shape(self):
        assert TensorPreprocessingLayer().compute_output_shape((B, D)) == (B, D)

    def test_serialization(self, sample, tmp_path):
        _roundtrip(
            TensorPreprocessingLayer(enable_final_clipping=True, name="prep"),
            sample, "prep", tmp_path,
        )
