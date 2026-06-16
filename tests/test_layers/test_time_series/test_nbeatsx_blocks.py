"""Tests for the N-BEATSx ExogenousBlock."""

import keras
import numpy as np
import pytest

from dl_techniques.layers.time_series.nbeatsx_blocks import ExogenousBlock

B = 4
UNITS, THETAS, BACK, FL, INPUT_DIM = 16, 4, 12, 6, 1
EXOG = 2


def _f32(*shape):
    return keras.ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal(shape).astype("float32")
    )


def _make(**kw):
    defaults = dict(
        exogenous_dim=EXOG, units=UNITS, thetas_dim=THETAS,
        backcast_length=BACK, forecast_length=FL, input_dim=INPUT_DIM,
    )
    defaults.update(kw)
    return ExogenousBlock(**defaults)


def _exog_inputs():
    return (_f32(B, BACK, EXOG), _f32(B, FL, EXOG))


class TestExogenousBlock:

    def test_construction(self):
        block = _make()
        assert block.exogenous_dim == EXOG
        assert block.use_tcn is True

    @pytest.mark.parametrize("use_tcn", [True, False])
    def test_forward_pass(self, use_tcn):
        block = _make(use_tcn=use_tcn)
        residual = _f32(B, BACK * INPUT_DIM)
        backcast, forecast = block(residual, exogenous_inputs=_exog_inputs())
        assert backcast.shape[0] == B and forecast.shape[0] == B
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(forecast)))

    def test_missing_exogenous_raises(self):
        block = _make()
        with pytest.raises(ValueError):
            block(_f32(B, BACK * INPUT_DIM))

    def test_serialization_round_trip(self):
        block = _make(use_tcn=True)
        residual = _f32(B, BACK * INPUT_DIM)
        exog = _exog_inputs()
        b0, f0 = block(residual, exogenous_inputs=exog)

        rebuilt = ExogenousBlock.from_config(block.get_config())
        rebuilt(residual, exogenous_inputs=exog)  # build clone
        rebuilt.set_weights(block.get_weights())
        b1, f1 = rebuilt(residual, exogenous_inputs=exog)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(f0), keras.ops.convert_to_numpy(f1),
            rtol=1e-5, atol=1e-5,
        )

    def test_get_config_round_trip(self):
        rebuilt = ExogenousBlock.from_config(_make(tcn_filters=8).get_config())
        assert rebuilt.exogenous_dim == EXOG
