"""Tests for the AdaptiveLagAttentionLayer."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.adaptive_lag_attention import AdaptiveLagAttentionLayer

B, CTX, NLAGS = 4, 12, 5


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    context = rng.standard_normal((B, CTX)).astype("float32")
    lags = rng.standard_normal((B, NLAGS)).astype("float32")
    return context, lags


class TestAdaptiveLagAttentionLayer:

    def test_construction(self):
        assert AdaptiveLagAttentionLayer(num_lags=NLAGS).num_lags == NLAGS

    def test_invalid_num_lags(self):
        with pytest.raises(ValueError):
            AdaptiveLagAttentionLayer(num_lags=0)

    def test_forward_pass(self, inputs):
        context, lags = inputs
        out = AdaptiveLagAttentionLayer(num_lags=NLAGS)([context, lags])
        assert tuple(out.shape) == (B, 1)

    def test_build_lag_mismatch_raises(self, inputs):
        context, lags = inputs
        layer = AdaptiveLagAttentionLayer(num_lags=NLAGS + 1)
        with pytest.raises(ValueError):
            layer([context, lags])

    def test_compute_output_shape(self):
        layer = AdaptiveLagAttentionLayer(num_lags=NLAGS)
        assert layer.compute_output_shape([(B, CTX), (B, NLAGS)]) == (B, 1)

    def test_serialization_round_trip(self, inputs, tmp_path):
        context, lags = inputs
        ctx_in = keras.Input(shape=(CTX,), name="ctx")
        lag_in = keras.Input(shape=(NLAGS,), name="lags")
        out = AdaptiveLagAttentionLayer(num_lags=NLAGS, name="ala")([ctx_in, lag_in])
        model = keras.Model([ctx_in, lag_in], out)
        y0 = model([context, lags])
        path = os.path.join(tmp_path, "ala.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"AdaptiveLagAttentionLayer": AdaptiveLagAttentionLayer}
        )
        y1 = loaded([context, lags])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
