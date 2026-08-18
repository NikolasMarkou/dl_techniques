"""
RED proofs for ``tests/test_models/knob_sensitivity_oracle.py``.

An instrument that has never been shown to fail is the same defect one layer up
as the ``test_different_X`` sweep it replaces. Each test here builds a toy model
TWICE -- once with a builder that honours the knob and once with a builder that
silently ignores it, which is the exact shape of the defect the oracle exists to
convict -- and requires the oracle to pass on the first and raise on the second.
"""

import keras
import numpy as np
import pytest

from .knob_sensitivity_oracle import (
    assert_structural_knob_changes_weights,
    assert_value_knob_changes_output,
    knob_output_deltas,
    weight_signature,
)


def _mlp(units: int, activation: str) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(units, activation=activation),
            keras.layers.Dense(3),
        ]
    )
    model.build((None, 4))
    return model


X = np.random.default_rng(0).random((2, 4)).astype("float32")


class TestStructuralInstrument:
    def test_honoured_structural_knob_passes(self):
        builders = {u: (lambda u=u: _mlp(u, "relu")) for u in (4, 8, 16)}
        sigs = assert_structural_knob_changes_weights(builders, knob="units")
        assert sigs[4] != sigs[16]

    def test_dropped_structural_knob_raises(self):
        # The builder ignores `units` -- the defect this oracle exists for.
        builders = {u: (lambda u=u: _mlp(8, "relu")) for u in (4, 8, 16)}
        with pytest.raises(AssertionError, match="units is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="units")

    def test_a_partially_dropped_knob_raises_on_the_inert_pair(self):
        # 4 -> 8 responds, 8 -> 16 does not. An outs[0]-vs-outs[-1] comparison
        # would pass here; the adjacent-pair comparison must not.
        sizes = {4: 4, 8: 8, 16: 8}
        builders = {u: (lambda u=u: _mlp(sizes[u], "relu")) for u in (4, 8, 16)}
        with pytest.raises(AssertionError, match=r"units=8 and units=16"):
            assert_structural_knob_changes_weights(builders, knob="units")


class TestValueInstrument:
    def test_honoured_value_knob_passes(self):
        builders = {a: (lambda a=a: _mlp(8, a)) for a in ("relu", "tanh")}
        deltas = assert_value_knob_changes_output(builders, X, knob="activation")
        assert all(d > 1e-5 for d in deltas.values())

    def test_dropped_value_knob_raises(self):
        builders = {a: (lambda a=a: _mlp(8, "relu")) for a in ("relu", "tanh")}
        with pytest.raises(AssertionError, match="activation is a no-op"):
            assert_value_knob_changes_output(builders, X, knob="activation")

    def test_a_dropped_value_knob_measures_exactly_zero(self):
        # The seeded-build contract: identical signature + identical seed =>
        # bit-identical weights, so an inert knob measures 0.0 and not noise.
        builders = {a: (lambda a=a: _mlp(8, "relu")) for a in ("relu", "tanh")}
        deltas = knob_output_deltas(builders, X)
        assert list(deltas.values()) == [0.0]

    def test_a_structural_knob_is_rejected_by_the_value_instrument(self):
        builders = {u: (lambda u=u: _mlp(u, "relu")) for u in (4, 8)}
        with pytest.raises(AssertionError, match="STRUCTURAL knob"):
            assert_value_knob_changes_output(builders, X, knob="units")


class TestSeedingContract:
    def test_same_config_built_twice_is_bit_identical(self):
        # If this ever fails, every value-knob assertion in the suite is
        # measuring RNG drift instead of the knob.
        builders = {"a": (lambda: _mlp(8, "relu")), "b": (lambda: _mlp(8, "relu"))}
        assert list(knob_output_deltas(builders, X).values()) == [0.0]

    def test_signature_is_empty_for_an_unbuilt_model(self):
        assert weight_signature(keras.Sequential([keras.layers.Dense(3)])) == ()

    def test_a_single_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            assert_structural_knob_changes_weights(
                {4: (lambda: _mlp(4, "relu"))}, knob="units"
            )
