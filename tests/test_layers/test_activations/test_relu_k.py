"""Tests for the ReLUK activation layer."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.activations.relu_k import ReLUK


def _x() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.standard_normal((4, 8)).astype("float32")


class TestReLUK:

    def test_construction(self) -> None:
        assert ReLUK(k=2).k == 2

    def test_invalid_k_type(self) -> None:
        with pytest.raises(TypeError):
            ReLUK(k=2.5)

    def test_invalid_k_value(self) -> None:
        with pytest.raises(ValueError):
            ReLUK(k=0)

    def test_forward_pass(self) -> None:
        y = ops.convert_to_numpy(ReLUK(k=3)(_x()))
        assert y.shape == (4, 8)
        assert np.all(y >= 0.0)

    def test_compute_output_shape(self) -> None:
        layer = ReLUK(k=3)
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self) -> None:
        inp = keras.Input(shape=(8,))
        out = ReLUK(k=2)(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reluk.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )


# ---------------------------------------------------------------------
# Mechanism oracles -- plan-2026-08-27T103353-60745fe0 / iter-2/step-9.
#
# The iter-2/step-8 mutation probe deleted this layer's defining step
# (`return keras.ops.power(relu_output, float(self.k))` -> `return
# relu_output`, leaving the `k == 1` fast path intact so the layer becomes
# plain ReLU for EVERY k). That moved the output by max|delta| = 5.57226 and
# all six pre-existing tests in this file still passed: the suite was BLIND.
# `test_forward_pass` only checks `y >= 0`, which plain ReLU satisfies too.
#
# The two oracles below are derivable without reading relu_k.py: one is the
# closed form `max(0, x)**k` evaluated in NumPy, the other is the algebraic
# invariant that defines the layer -- degree-k positive homogeneity.
# ---------------------------------------------------------------------


class TestReLUKMechanismOracle:

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_matches_the_numpy_closed_form(self, k: int) -> None:
        """`ReLUK(k)(x)` equals `max(0, x) ** k`, evaluated in float64 NumPy.

        Tolerance rtol=1e-6, atol=0. Derivation: the two sides differ only by
        float32 rounding of the power op. Measured max relative error over
        k in {1,2,3,4} on this input is 1.207e-07 -- about one float32 ulp
        (np.finfo(float32).eps = 1.192e-07) -- so 1e-6 carries ~8 ulp of
        headroom. atol is deliberately 0: every negative entry is exactly 0.0
        on both sides, so an absolute floor could only mask a real error on
        the positive half, where the values are O(1).
        """
        x = _x()
        y = keras.ops.convert_to_numpy(ReLUK(k=k)(x))
        reference = np.maximum(0.0, x.astype(np.float64)) ** k
        np.testing.assert_allclose(y, reference, rtol=1e-6, atol=0.0)

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    @pytest.mark.parametrize("c", [0.5, 2.0, 3.0])
    def test_is_positively_homogeneous_of_degree_k(self, k: int, c: float) -> None:
        """`f(c*x) == c**k * f(x)` for every `c > 0`.

        This is the whole point of the layer and it needs no reference
        implementation: `max(0, cx)^k = c^k max(0, x)^k` exactly, in real
        arithmetic, for c > 0. Plain ReLU is homogeneous of degree 1 ONLY, so
        the k=2/3/4 arms fail immediately if the power step is removed.

        Tolerance rtol=2e-6, atol=0. Derivation: measured max relative error
        is 3.600e-07 (k=4, c=3, where the compared values reach ~1449), about
        3 float32 ulp; 2e-6 is ~17 ulp. atol=0 for the same reason as above --
        the negative half is exactly 0.0 on both sides, so `assert_allclose`
        compares 0 against 0 there and passes with a zero budget.
        """
        x = _x()
        y = keras.ops.convert_to_numpy(ReLUK(k=k)(x))
        y_scaled = keras.ops.convert_to_numpy(ReLUK(k=k)((c * x).astype("float32")))
        np.testing.assert_allclose(y_scaled, (c ** k) * y, rtol=2e-6, atol=0.0)
