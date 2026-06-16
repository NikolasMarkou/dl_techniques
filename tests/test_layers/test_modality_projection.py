"""Tests for the ModalityProjection layer (VLM visual->language projection)."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.modality_projection import ModalityProjection

B, IN, OUT = 2, 8, 16
SEQ = 5  # 1 CLS + 4 spatial tokens (h=2, divisible by scale_factor=2)


@pytest.fixture
def sample():
    return np.random.default_rng(0).standard_normal((B, SEQ, IN)).astype("float32")


class TestModalityProjection:

    def test_construction(self):
        layer = ModalityProjection(input_dim=IN, output_dim=OUT)
        assert layer.output_dim == OUT

    @pytest.mark.parametrize("bad", [
        {"input_dim": 0, "output_dim": OUT},
        {"input_dim": IN, "output_dim": 0},
        {"input_dim": IN, "output_dim": OUT, "scale_factor": 0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            ModalityProjection(**bad)

    def test_build_wrong_input_dim_raises(self, sample):
        layer = ModalityProjection(input_dim=IN + 1, output_dim=OUT)
        with pytest.raises(ValueError):
            layer(sample)

    def test_forward_pass(self, sample):
        layer = ModalityProjection(input_dim=IN, output_dim=OUT, scale_factor=2)
        out = layer(sample)
        assert out.shape[0] == B and out.shape[-1] == OUT

    def test_compute_output_shape_matches_call(self, sample):
        layer = ModalityProjection(input_dim=IN, output_dim=OUT, scale_factor=2)
        out = layer(sample)
        assert tuple(out.shape) == tuple(layer.compute_output_shape(sample.shape))

    def test_serialization_round_trip(self, sample, tmp_path):
        inp = keras.Input(shape=(SEQ, IN))
        out = ModalityProjection(input_dim=IN, output_dim=OUT, scale_factor=2, name="mp")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample)
        path = os.path.join(tmp_path, "mp.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"ModalityProjection": ModalityProjection}
        )
        y1 = loaded(sample)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        layer = ModalityProjection(input_dim=IN, output_dim=OUT, use_gelu=False, use_layer_norm=False)
        rebuilt = ModalityProjection.from_config(layer.get_config())
        assert rebuilt.use_gelu is False and rebuilt.use_layer_norm is False
