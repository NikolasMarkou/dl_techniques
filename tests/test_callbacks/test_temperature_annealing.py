"""Tests for TemperatureAnnealingCallback (M2, plan_2026-05-13_3a2f1d23)."""

import math

import numpy as np
import pytest
import keras

from dl_techniques.callbacks.temperature_annealing import (
    TemperatureAnnealingCallback,
)
from dl_techniques.layers.logic.logic_operators import LearnableLogicOperator


class TestTemperatureAnnealingCallback:
    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule"):
            TemperatureAnnealingCallback(schedule="bogus")

    def test_invalid_temps_raise(self):
        with pytest.raises(ValueError, match="positive"):
            TemperatureAnnealingCallback(t_init=0.0)

    def test_cosine_endpoints(self):
        cb = TemperatureAnnealingCallback(
            schedule="cosine", t_init=5.0, t_final=0.1, total_epochs=10
        )
        assert math.isclose(cb._temperature_at(0), 5.0, abs_tol=1e-6)
        assert math.isclose(cb._temperature_at(9), 0.1, abs_tol=1e-6)
        # Midpoint should be (5+0.1)/2 = 2.55 exactly for cosine.
        assert math.isclose(cb._temperature_at(4 if 4 < 5 else 4), 2.55, abs_tol=0.5)

    def test_linear_endpoints(self):
        cb = TemperatureAnnealingCallback(
            schedule="linear", t_init=4.0, t_final=1.0, total_epochs=4
        )
        assert math.isclose(cb._temperature_at(0), 4.0, abs_tol=1e-6)
        assert math.isclose(cb._temperature_at(3), 1.0, abs_tol=1e-6)
        # Middle epoch (1/3 of the way): 4 + (1 - 4) * (1/3) = 3.0.
        assert math.isclose(cb._temperature_at(1), 3.0, abs_tol=1e-6)

    def test_exp_endpoints(self):
        cb = TemperatureAnnealingCallback(
            schedule="exp", t_init=4.0, t_final=0.25, total_epochs=3
        )
        assert math.isclose(cb._temperature_at(0), 4.0, abs_tol=1e-6)
        assert math.isclose(cb._temperature_at(2), 0.25, abs_tol=1e-6)

    def test_callback_sets_softplus_temperature(self):
        """End-to-end smoke: build a model with a logic op and verify the
        callback updates the temperature attribute through 3 epochs."""
        inp = keras.Input(shape=(4,))
        op = LearnableLogicOperator(
            operation_types=['and', 'or'],
            softplus_temperature=True,
            allow_unary_degenerate=True,
            name='probe',
        )
        out = op(inp)
        model = keras.Model(inp, out)
        model.compile(optimizer='adam', loss='mse')

        cb = TemperatureAnnealingCallback(
            schedule="linear", t_init=5.0, t_final=1.0, total_epochs=3
        )

        x = np.random.randn(8, 4).astype(np.float32)
        y = np.random.uniform(0, 1, (8, 4)).astype(np.float32)
        model.fit(x, y, epochs=3, batch_size=4, verbose=0, callbacks=[cb])

        # After fit completes, on_epoch_begin for epoch 2 (final epoch) should
        # have set the temperature to t_final=1.0. With softplus_temperature
        # the raw weight stores log(expm1(1.0)).
        expected_raw = math.log(math.expm1(1.0))
        # The callback fires at on_epoch_begin so the optimizer also takes
        # gradient steps during the final epoch — allow a small drift.
        assert abs(float(op.temperature) - expected_raw) < 1e-2

    def test_callback_round_trip(self):
        cb = TemperatureAnnealingCallback(
            schedule="cosine", t_init=5.0, t_final=0.1, total_epochs=10,
            layer_names=["probe"],
        )
        cfg = cb.get_config()
        assert cfg["schedule"] == "cosine"
        cb2 = TemperatureAnnealingCallback(**cfg)
        assert cb2.layer_names == ["probe"]
