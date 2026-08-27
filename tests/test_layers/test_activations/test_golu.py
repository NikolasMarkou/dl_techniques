"""Tests for the GoLU (Gompertz Linear Unit) activation layer."""

import os
import tempfile

import numpy as np
import keras
import pytest
from keras import ops

from dl_techniques.layers.activations.golu import GoLU


def _x() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((4, 8)).astype("float32")


class TestGoLU:

    def test_construction_defaults(self) -> None:
        layer = GoLU()
        assert layer.alpha == 1.0 and layer.beta == 1.0 and layer.gamma == 1.0

    def test_construction_custom(self) -> None:
        layer = GoLU(alpha=2.0, beta=0.5, gamma=1.5)
        assert layer.alpha == 2.0 and layer.beta == 0.5 and layer.gamma == 1.5

    def test_forward_pass(self) -> None:
        layer = GoLU()
        y = layer(_x())
        assert tuple(y.shape) == (4, 8)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_compute_output_shape(self) -> None:
        layer = GoLU()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(8,))
        out = GoLU(alpha=1.5, beta=0.7, gamma=1.2)(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "golu.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )


# ---------------------------------------------------------------------
# Mechanism oracles -- plan-2026-08-27T103353-60745fe0 / iter-2/step-9.
#
# The iter-2/step-8 mutation probe replaced the Gompertz gate
# (`alpha * exp(-beta * exp(-gamma * x))` -> `keras.ops.sigmoid(inputs)`),
# which degenerates GoLU into SiLU and makes `alpha`, `beta` and `gamma`
# DEAD KNOBS. That moved the output by max|delta| = 0.2419 on all 48 values
# and all five pre-existing tests still passed: the suite was BLIND. Nothing
# here read a value; the file tested construction, finiteness, output shape
# and a save/load round-trip, and a round-trip of a dead knob round-trips
# just as cleanly as a live one.
#
# Two oracles below: the closed-form Gompertz gate in NumPy, and a knob-effect
# arm proving each of the three parameters actually reaches the output.
# ---------------------------------------------------------------------


class TestGoLUMechanismOracle:

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [(1.0, 1.0, 1.0), (1.5, 0.7, 1.2), (2.0, 0.5, 1.5)],
    )
    def test_matches_the_closed_form_gompertz_gate(
            self, alpha: float, beta: float, gamma: float
    ) -> None:
        """Equals `x * alpha * exp(-beta * exp(-gamma * x))` in float64 NumPy.

        The Gompertz gate is asymmetric and is NOT a sigmoid; writing it out
        in NumPy is what separates GoLU from SiLU. All three parameters appear
        in the reference, so a gate that ignores them fails here.

        Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute error
        across the three parameter triples is 1.160e-07 on outputs of
        magnitude <= ~4, i.e. about one float32 ulp at that magnitude
        (ulp(4.0) = 4.77e-07 -- the measurement is in fact tighter than one
        ulp there). rtol is pinned to 0 for a pure absolute bound.
        """
        x = _x()
        y = keras.ops.convert_to_numpy(GoLU(alpha=alpha, beta=beta, gamma=gamma)(x))

        v = x.astype(np.float64)
        reference = v * alpha * np.exp(-beta * np.exp(-gamma * v))

        np.testing.assert_allclose(y, reference, atol=1e-6, rtol=0.0)

    @pytest.mark.parametrize(
        "knob,value,minimum_delta",
        [("alpha", 1.5, 0.05), ("beta", 0.5, 0.05), ("gamma", 2.0, 0.05)],
    )
    def test_each_gate_parameter_reaches_the_output(
            self, knob: str, value: float, minimum_delta: float
    ) -> None:
        """Changing one of `alpha` / `beta` / `gamma` must change the output.

        A dead-knob assertion. The step-8 mutation replaced the gate with a
        parameterless sigmoid, which leaves all three constructor arguments
        stored on the layer (so `get_config` still round-trips them) while
        none of them reach `call`.

        Threshold 0.05. Derivation: the measured `max|delta|` against the
        default `GoLU()` is 0.5294 (alpha=1.5), 0.1911 (beta=0.5) and 0.2215
        (gamma=2.0) on this input. 0.05 is below the smallest of those by
        ~3.8x and far above float32 noise (~1e-07), so the arm neither
        tracks a rounding wobble nor sits at the edge of its own measurement.
        """
        x = _x()
        baseline = keras.ops.convert_to_numpy(GoLU()(x))
        changed = keras.ops.convert_to_numpy(GoLU(**{knob: value})(x))

        assert np.abs(changed - baseline).max() > minimum_delta
