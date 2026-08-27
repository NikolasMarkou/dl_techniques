"""Tests for the SquashLayer (capsule squashing) activation."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.activations.squash import SquashLayer


def _x() -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.standard_normal((4, 3, 8)).astype("float32")


class TestSquashLayer:

    def test_construction_default_epsilon(self) -> None:
        layer = SquashLayer()
        assert layer.axis == -1
        assert layer.epsilon > 0

    def test_invalid_axis(self) -> None:
        with pytest.raises(ValueError):
            SquashLayer(axis=1.5)

    def test_invalid_epsilon(self) -> None:
        with pytest.raises(ValueError):
            SquashLayer(epsilon=-1.0)

    def test_forward_norm_below_one(self) -> None:
        y = ops.convert_to_numpy(SquashLayer()(_x()))
        assert y.shape == (4, 3, 8)
        norms = np.linalg.norm(y, axis=-1)
        assert np.all(norms <= 1.0 + 1e-5)

    def test_compute_output_shape(self) -> None:
        layer = SquashLayer()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(3, 8))
        out = SquashLayer()(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "squash.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )


# ---------------------------------------------------------------------
# Mechanism oracles -- plan-2026-08-27T103353-60745fe0 / iter-2/step-9.
#
# The iter-2/step-8 mutation probe replaced the norm-dependent scale
# (`scale = squared_norm / (1.0 + squared_norm)` -> `scale = 1.0`), turning
# the layer into plain L2 normalization. That moved the output by
# max|delta| = 0.366316 on all 48 values and all six pre-existing tests still
# passed: the suite was BLIND. `test_forward_norm_below_one` asserts
# `||y|| <= 1`, and L2 normalization gives EXACTLY 1, which clears that bound.
#
# The two oracles below are derivable from the published squashing formula
# alone: the closed form, and the length law `||squash(v)|| = s / (1 + s)`
# with `s = ||v||^2`, which is strictly below 1 and strictly increasing in
# `||v||`. L2 normalization returns length 1 for every input, so the length
# law fails at every scale -- most violently at small norms.
# ---------------------------------------------------------------------


class TestSquashMechanismOracle:

    def test_matches_the_closed_form_squash_formula(self) -> None:
        """Equals `(s / (1 + s)) * v / sqrt(s + epsilon)` with `s = ||v||^2`.

        Evaluated in float64 NumPy. `epsilon` sits inside the square root
        because that is the layer's documented zero-vector guard and its
        default is `keras.backend.epsilon()`; it is read from the layer's
        config, not from its source.

        Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute error
        is 7.588e-08 on outputs of magnitude <= 1, i.e. under one float32 ulp
        of 1.0 (1.192e-07). rtol is pinned to 0 so this is a pure absolute
        bound -- `assert_allclose`'s default rtol=1e-7 would otherwise
        contribute silently to a nominally-atol failure.
        """
        layer = SquashLayer()
        x = _x()
        y = keras.ops.convert_to_numpy(layer(x))

        v = x.astype(np.float64)
        squared_norm = np.sum(v ** 2, axis=-1, keepdims=True)
        reference = (
            (squared_norm / (1.0 + squared_norm))
            * v / np.sqrt(squared_norm + layer.epsilon)
        )
        np.testing.assert_allclose(y, reference, atol=1e-6, rtol=0.0)

    @pytest.mark.parametrize("scale", [1e-2, 0.1, 1.0, 3.0, 10.0])
    def test_output_length_follows_the_squashing_law(self, scale: float) -> None:
        """`||squash(v)|| == s / (1 + s)` where `s = ||v||^2`.

        A pure algebraic invariant of the squashing non-linearity: the output
        length is a strictly increasing function of the input length, bounded
        in [0, 1), and it is NOT constant. The `scale` sweep spans four orders
        of magnitude of input norm so the law is pinned across the whole
        range, not at one convenient point. At scale=1e-2 the predicted length
        is ~3.2e-04; L2 normalization would return 1.0 there.

        Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute error
        across all five scales is 7.588e-08, under one float32 ulp of 1.0.
        """
        x = (_x() * scale).astype("float32")
        y = keras.ops.convert_to_numpy(SquashLayer()(x))

        v = x.astype(np.float64)
        squared_norm = np.sum(v ** 2, axis=-1)
        predicted_length = squared_norm / (1.0 + squared_norm)
        measured_length = np.linalg.norm(y.astype(np.float64), axis=-1)

        np.testing.assert_allclose(
            measured_length, predicted_length, atol=1e-6, rtol=0.0
        )
        assert np.all(measured_length < 1.0)

    def test_output_length_is_strictly_increasing_in_input_length(self) -> None:
        """Longer input vector -> strictly longer output vector.

        The monotone half of the squashing law, stated on its own so a failure
        says which property broke. L2 normalization is constant in the input
        length, so this is RED for it at every pair of scales.
        """
        unit = np.zeros((5, 4), dtype="float32")
        unit[:, 0] = 1.0
        scales = np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype="float32")
        x = unit * scales[:, None]

        y = keras.ops.convert_to_numpy(SquashLayer()(x))
        lengths = np.linalg.norm(y.astype(np.float64), axis=-1)

        assert np.all(np.diff(lengths) > 0.0), lengths
