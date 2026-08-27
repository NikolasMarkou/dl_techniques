"""Tests for the HardSigmoid and HardSwish activation layers."""

import os
import tempfile

import numpy as np
import keras
from keras import ops
import pytest

from dl_techniques.layers.activations.hard_sigmoid import HardSigmoid
from dl_techniques.layers.activations.hard_swish import HardSwish


def _x() -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.standard_normal((4, 8)) * 5.0).astype("float32")


@pytest.mark.parametrize("layer_cls", [HardSigmoid, HardSwish])
class TestHardActivations:

    def test_construction(self, layer_cls) -> None:
        layer = layer_cls()
        assert isinstance(layer, keras.layers.Layer)

    def test_forward_pass(self, layer_cls) -> None:
        layer = layer_cls()
        y = layer(_x())
        assert tuple(y.shape) == (4, 8)
        assert np.all(np.isfinite(ops.convert_to_numpy(y)))

    def test_hard_sigmoid_range(self, layer_cls) -> None:
        if layer_cls is not HardSigmoid:
            pytest.skip("range check is HardSigmoid-specific")
        y = ops.convert_to_numpy(HardSigmoid()(_x()))
        assert y.min() >= 0.0 - 1e-6 and y.max() <= 1.0 + 1e-6

    def test_compute_output_shape(self, layer_cls) -> None:
        layer = layer_cls()
        x = _x()
        assert tuple(layer.compute_output_shape(x.shape)) == tuple(layer(x).shape)

    def test_serialization_round_trip(self, layer_cls) -> None:
        inp = keras.Input(shape=(8,))
        out = layer_cls()(inp)
        model = keras.Model(inp, out)
        x = _x()
        y0 = model(x)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "act.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
        y1 = loaded(x)
        np.testing.assert_allclose(
            ops.convert_to_numpy(y0), ops.convert_to_numpy(y1), atol=1e-6
        )


# ---------------------------------------------------------------------
# Mechanism oracles -- plan-2026-08-27T103353-60745fe0 / iter-2/step-9.
#
# The iter-2/step-8 mutation probe replaced the piecewise-linear ReLU6 form
# in BOTH layers (`self.activation(inputs + 3.0) / 6.0` ->
# `keras.ops.sigmoid(inputs)`) with a smooth sigmoid. That moved the output by
# max|delta| = 0.101919 on all 96 values and all nine pre-existing tests
# still passed: the suite was BLIND. The near-miss was
# `test_hard_sigmoid_range`, which asserts `0 <= y <= 1` -- a true sigmoid
# satisfies that too, so the mutation slid straight through it.
#
# The defining property of a HARD activation is EXACT saturation: it reaches
# exactly 0.0 and exactly 1.0 at finite inputs, and is exactly linear in
# between. A smooth sigmoid is never exactly 0 or 1 for any finite input.
# The oracles below assert that, plus the closed forms. Note these live
# OUTSIDE the module's `layer_cls`-parametrized class on purpose: the
# saturation constants differ per layer (HardSigmoid saturates to 0 and 1,
# HardSwish to 0 and the identity), so a shared parametrization would have to
# branch, and a branching test is how `test_hard_sigmoid_range` ended up
# skipping half its own ids.
# ---------------------------------------------------------------------


#: Knots and interior points of the piecewise-linear form. -3 and +3 are the
#: exact breakpoints of `ReLU6(x + 3) / 6`; the +/-1e-4 neighbours pin which
#: side of each breakpoint the saturation starts on.
_SATURATION_GRID = np.array(
    [-100.0, -10.0, -3.5, -3.0, -2.9999, -1.5, 0.0, 1.5, 2.9999, 3.0, 3.5, 10.0, 100.0],
    dtype="float32",
)


class TestHardSigmoidSaturatesExactly:

    def test_is_exactly_zero_at_and_below_minus_three(self) -> None:
        """`HardSigmoid(x) == 0.0` EXACTLY for every `x <= -3`.

        Exact equality, no tolerance: `ReLU6(x + 3)` is `max(0, x + 3)` capped
        at 6, and `max(0, 0.0)` is the float 0.0, so `0.0 / 6.0` is exactly
        0.0 in IEEE-754. There is no rounding for a tolerance to absorb.
        `keras.ops.sigmoid(-3.0)` is 0.0474, so this is RED for a smooth gate.
        """
        x = _SATURATION_GRID[_SATURATION_GRID <= -3.0]
        y = keras.ops.convert_to_numpy(HardSigmoid()(x))
        assert np.all(y == 0.0), y

    def test_is_exactly_one_at_and_above_three(self) -> None:
        """`HardSigmoid(x) == 1.0` EXACTLY for every `x >= 3`.

        Exact equality for the same reason: the ReLU6 cap returns exactly 6.0
        and `6.0 / 6.0` is exactly 1.0. `keras.ops.sigmoid(3.0)` is 0.9526.
        """
        x = _SATURATION_GRID[_SATURATION_GRID >= 3.0]
        y = keras.ops.convert_to_numpy(HardSigmoid()(x))
        assert np.all(y == 1.0), y

    def test_is_exactly_linear_between_the_breakpoints(self) -> None:
        """`HardSigmoid(x) == (x + 3) / 6` for `-3 < x < 3`.

        The middle segment is a straight line, which a sigmoid's S-curve is
        not. Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute
        error against the float64 line is 3.974e-08 on outputs in [0, 1],
        under one float32 ulp of 1.0 (1.192e-07).
        """
        x = _SATURATION_GRID[(_SATURATION_GRID > -3.0) & (_SATURATION_GRID < 3.0)]
        y = keras.ops.convert_to_numpy(HardSigmoid()(x))
        reference = (x.astype(np.float64) + 3.0) / 6.0
        np.testing.assert_allclose(y, reference, atol=1e-6, rtol=0.0)


class TestHardSwishSaturatesExactly:

    def test_is_exactly_zero_at_and_below_minus_three(self) -> None:
        """`HardSwish(x) == 0.0` EXACTLY for every `x <= -3`.

        `x * 0.0` is exactly 0.0 (signed zero compares equal to 0.0), so the
        multiply preserves the exactness of the hard sigmoid factor. A smooth
        gate gives `-3 * 0.0474 = -0.142` here.
        """
        x = _SATURATION_GRID[_SATURATION_GRID <= -3.0]
        y = keras.ops.convert_to_numpy(HardSwish()(x))
        assert np.all(y == 0.0), y

    def test_is_exactly_the_identity_at_and_above_three(self) -> None:
        """`HardSwish(x) == x` EXACTLY for every `x >= 3`.

        Above the upper breakpoint the gate is exactly 1.0, so `x * 1.0` is
        `x` bit-for-bit. This is the property that makes HardSwish linear in
        its upper tail, and it is the one a smooth gate cannot reproduce.
        """
        x = _SATURATION_GRID[_SATURATION_GRID >= 3.0]
        y = keras.ops.convert_to_numpy(HardSwish()(x))
        assert np.all(y == x), (y, x)

    def test_matches_the_closed_form_between_the_breakpoints(self) -> None:
        """`HardSwish(x) == x * (x + 3) / 6` for `-3 < x < 3`.

        Tolerance atol=1e-6, rtol=0. Derivation: measured max absolute error
        against the float64 parabola is 1.209e-07 on outputs of magnitude
        <= ~3, i.e. under one float32 ulp at that magnitude
        (ulp(3.0) = 2.38e-07).
        """
        x = _SATURATION_GRID[(_SATURATION_GRID > -3.0) & (_SATURATION_GRID < 3.0)]
        y = keras.ops.convert_to_numpy(HardSwish()(x))
        v = x.astype(np.float64)
        np.testing.assert_allclose(y, v * (v + 3.0) / 6.0, atol=1e-6, rtol=0.0)
