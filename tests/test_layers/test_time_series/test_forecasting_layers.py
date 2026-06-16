"""Tests for the forecasting_layers (NaiveResidual / ForecastabilityGate /
ConformalQuantileHead).

NaiveResidual and ForecastabilityGate take multiple positional tensor inputs,
so their serialization is verified via a ``get_config`` -> ``from_config`` +
weight-transfer round-trip rather than a functional ``.keras`` model.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.forecasting_layers import (
    NaiveResidual,
    ForecastabilityGate,
    ConformalQuantileHead,
)

B, BACK, FL, F = 4, 12, 6, 3


def _f32(*shape):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal(shape).astype("float32")
    )


class TestNaiveResidual:
    def test_forward_and_shape(self):
        layer = NaiveResidual(forecast_length=FL)
        history = _f32(B, BACK, F)
        net_out = _f32(B, FL, F)
        out = layer(history, net_out)
        assert tuple(out.shape) == (B, FL, F)
        assert layer.compute_output_shape((B, BACK, F)) == (B, FL, F)

    def test_get_config_round_trip(self):
        rebuilt = NaiveResidual.from_config(NaiveResidual(forecast_length=FL).get_config())
        assert rebuilt.forecast_length == FL


class TestForecastabilityGate:
    def test_forward_and_serialization(self):
        layer = ForecastabilityGate(hidden_units=8)
        inputs = _f32(B, BACK, F)
        deep = _f32(B, FL, F)
        naive = _f32(B, FL, F)
        out0 = layer(inputs, deep, naive)
        assert tuple(out0.shape) == (B, FL, F)

        rebuilt = ForecastabilityGate.from_config(layer.get_config())
        rebuilt(inputs, deep, naive)  # build clone
        rebuilt.set_weights(layer.get_weights())
        out1 = rebuilt(inputs, deep, naive)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out0), keras.ops.convert_to_numpy(out1),
            rtol=1e-5, atol=1e-5,
        )


class TestConformalQuantileHead:
    def test_forward_and_shape(self):
        head = ConformalQuantileHead(forecast_length=FL, output_dim=F)
        # call() returns a single stacked (B, FL, F, 3) tensor (lower/median/upper).
        out = head(_f32(B, 10))
        assert tuple(out.shape) == (B, FL, F, 3)
        assert head.compute_output_shape((B, 10)) == (B, FL, F, 3)

    def test_serialization_round_trip(self, tmp_path):
        inp = keras.Input(shape=(10,))
        out = ConformalQuantileHead(forecast_length=FL, output_dim=F, name="cqh")(inp)
        model = keras.Model(inp, out)
        x = np.random.default_rng(0).standard_normal((B, 10)).astype("float32")
        y0 = model(x)
        path = os.path.join(tmp_path, "cqh.keras")
        model.save(path)
        loaded = keras.models.load_model(
            path, custom_objects={"ConformalQuantileHead": ConformalQuantileHead}
        )
        y1 = loaded(x)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
