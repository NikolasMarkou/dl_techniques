"""Tests for the TemporalFusionLayer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.temporal_fusion import TemporalFusionLayer

B, CTX, NLAGS, OUT = 4, 12, 5, 3


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    return (rng.standard_normal((B, CTX)).astype("float32"),
            rng.standard_normal((B, NLAGS)).astype("float32"))


class TestTemporalFusionLayer:

    def test_construction(self):
        layer = TemporalFusionLayer(output_dim=OUT, num_lags=NLAGS)
        assert layer.output_dim == OUT and layer.num_lags == NLAGS

    @pytest.mark.parametrize("bad", [
        {"output_dim": 0, "num_lags": NLAGS},
        {"output_dim": OUT, "num_lags": 0},
    ])
    def test_invalid_args_raise(self, bad):
        with pytest.raises(ValueError):
            TemporalFusionLayer(**bad)

    def test_forward_pass(self, inputs):
        context, lags = inputs
        out = TemporalFusionLayer(output_dim=OUT, num_lags=NLAGS)([context, lags])
        assert tuple(out.shape) == (B, OUT)

    def test_compute_output_shape(self):
        layer = TemporalFusionLayer(output_dim=OUT, num_lags=NLAGS)
        assert layer.compute_output_shape([(B, CTX), (B, NLAGS)]) == (B, OUT)

    def test_serialization_round_trip(self, inputs, tmp_path):
        context, lags = inputs
        ctx_in = keras.Input(shape=(CTX,), name="ctx")
        lag_in = keras.Input(shape=(NLAGS,), name="lags")
        out = TemporalFusionLayer(output_dim=OUT, num_lags=NLAGS, name="tf")([ctx_in, lag_in])
        model = keras.Model([ctx_in, lag_in], out)
        y0 = model([context, lags])
        path = os.path.join(tmp_path, "tf.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"TemporalFusionLayer": TemporalFusionLayer}
        )
        y1 = loaded([context, lags])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
