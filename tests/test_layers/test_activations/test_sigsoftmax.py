"""Tests for the sigsoftmax activation functions.

This module lands in parts. Step 2 of plan-2026-09-03T085145-3384c4dc ships the
independent float64 oracle and three arms: the closed-form comparison, the
all-negative regression guard, and the not-plain-softmax mechanism arm. Step 3
adds the ``SigSoftmax`` layer's construction, axis-validation, shape, config and
wrapper-identity arms. The remaining arms (dtype floor in both directions,
serialization, gradients, XLA, factory, export contract) arrive in step 5.
"""

import numpy as np
import keras
import pytest
from scipy.special import expit

from dl_techniques.layers.activations.sigsoftmax import (
    SigSoftmax,
    log_sigsoftmax,
    sigsoftmax,
)

# ---------------------------------------------------------------------
# Independent float64 oracles.
#
# Written from the paper's Definition 1 / Eq. (18), not from the module under
# test. No `keras.` call appears in either body: an oracle that reaches for
# `keras.ops.log_sigmoid` / `logsumexp` reproduces the implementation's own
# arithmetic and would credit a broken implementation with zero error.
# ---------------------------------------------------------------------


def _reference_sigsoftmax(z: np.ndarray) -> np.ndarray:
    """Normalised ``exp(z) * sigmoid(z)`` along the last axis, in float64 NumPy.

    This is the paper's Eq. (18) written out directly. Its validity range is
    roughly ``|z| < 300``: ``np.exp`` overflows float64 above ``z ~ 709.8``,
    and ``exp(z) * expit(z) ~ exp(2z)`` underflows to exactly 0 below
    ``z ~ -372``, at which point the normalisation becomes 0/0. Keep fixtures
    inside that range rather than rewriting this into log space, which would
    turn the oracle into a second copy of the implementation.

    :param z: real-valued logits, any shape.
    :type z: numpy.ndarray
    :return: probabilities summing to 1 along the last axis, float64.
    :rtype: numpy.ndarray
    """
    v = np.asarray(z, dtype=np.float64)
    n = np.exp(v) * expit(v)
    return n / n.sum(axis=-1, keepdims=True)


def _asymptotic_log_sigsoftmax(z: np.ndarray) -> np.ndarray:
    """Log-probabilities for a row where every logit is strongly negative.

    A second closed form, derived independently of the one above so that the
    all-negative row has a reference at all. For strongly negative ``z``,
    ``sigmoid(z) -> exp(z)``, so ``g(z) = exp(z) * sigmoid(z) -> exp(2z)`` and
    the normalised log-probability is ``2 * (z - max z)``.

    :param z: real-valued logits, strongly negative, any shape.
    :type z: numpy.ndarray
    :return: log-probabilities along the last axis, float64.
    :rtype: numpy.ndarray
    """
    v = np.asarray(z, dtype=np.float64)
    return 2.0 * (v - v.max(axis=-1, keepdims=True))


def _moderate_logits() -> np.ndarray:
    """Fixture well inside the float64 oracle's validity range.

    :return: ``(64, 20)`` float32 draws from a standard normal, all within +-6.
    :rtype: numpy.ndarray
    """
    return np.random.default_rng(0).standard_normal((64, 20)).astype("float32")


# ---------------------------------------------------------------------


def test_matches_the_closed_form_on_moderate_logits() -> None:
    """Equals the float64 NumPy form of Eq. (18) on ordinary logits.

    Tolerance atol=1e-6, rtol=0. Measured max absolute error on this fixture
    is 8.417e-08 on the default device and 1.212e-07 on CPU, on outputs
    bounded by 1. The bound sits about an order above the larger reading and
    is a pure absolute bound, so rtol is pinned to 0.
    """
    x = _moderate_logits()
    y = keras.ops.convert_to_numpy(sigsoftmax(x))

    reference = _reference_sigsoftmax(x)

    np.testing.assert_allclose(y, reference, atol=1e-6, rtol=0.0)


def test_the_all_negative_row_does_not_underflow_to_nan() -> None:
    """A row of strongly negative logits stays finite, sums to 1, and is right.

    The regression guard for the log-space formulation. The max-shift form
    returns ``nan nan nan`` here: ``sigmoid(z)`` does not shift with the max,
    so every lane of the linear-space numerator underflows to exactly 0 and
    the normalisation is 0/0.

    Three assertions, because finiteness alone is satisfied by a wrong
    ``[0, 0, 1]``. The lane-level check runs on the log scale: in float32 the
    correct probabilities are ``[0, 0, 1]`` anyway, since the true first lane
    is 5.148e-131 and float32's smallest subnormal is 1.4e-45. The
    log-probabilities carry the same information and are representable --
    measured ``[-300, -100, 0]``, matching ``2 * (z - max z)`` exactly.
    """
    x = np.array([[-300.0, -200.0, -150.0]], dtype="float32")

    y = keras.ops.convert_to_numpy(sigsoftmax(x))
    log_y = keras.ops.convert_to_numpy(log_sigsoftmax(x))

    assert np.all(np.isfinite(y)), f"sigsoftmax was not finite, observed {y}"
    assert np.all(np.isfinite(log_y)), (
        f"log_sigsoftmax was not finite, observed {log_y}"
    )

    np.testing.assert_allclose(
        y.sum(axis=-1), np.ones(1), atol=1e-6, rtol=0.0
    )

    np.testing.assert_allclose(
        log_y, _asymptotic_log_sigsoftmax(x), rtol=1e-3, atol=0.0
    )


def test_is_not_plain_softmax() -> None:
    """The sigmoid factor moves the output away from softmax.

    A refactor that collapses this module to ``keras.ops.softmax`` has to
    redden here. On the pinned row ``[[1, 2, 3]]`` sigsoftmax is
    ``[0.0719267, 0.2355637, 0.6925095]`` and softmax is
    ``[0.0900306, 0.2447285, 0.6652409]``, so the measured max absolute
    difference is 0.027268589. The threshold 0.02 sits 1.36x below that and
    five orders above float32 noise.
    """
    x = np.array([[1.0, 2.0, 3.0]], dtype="float32")

    y = keras.ops.convert_to_numpy(sigsoftmax(x))
    plain = keras.ops.convert_to_numpy(keras.ops.softmax(x))

    assert np.abs(y - plain).max() > 0.02


# ---------------------------------------------------------------------
# The layer. Step 3 arms only; the rest of the suite lands in step 5.
# ---------------------------------------------------------------------


class TestSigSoftmax:
    """Construction, axis validation, shape, config and wrapper identity."""

    def test_constructs_with_the_default_axis(self) -> None:
        """The default axis is -1."""
        assert SigSoftmax().axis == -1

    def test_constructs_with_a_custom_axis(self) -> None:
        """A supplied axis is stored unchanged, including a positive one."""
        assert SigSoftmax(axis=-2).axis == -2
        assert SigSoftmax(axis=0).axis == 0

    def test_a_bool_axis_raises(self) -> None:
        """``axis=True`` is rejected rather than silently read as ``axis=1``.

        ``bool`` subclasses ``int``, so a bare ``isinstance(axis, int)`` check
        accepts ``True`` and the layer normalises along axis 1.
        """
        with pytest.raises(ValueError, match="axis must be an integer"):
            SigSoftmax(axis=True)

    @pytest.mark.parametrize("bad_axis", ["1", 1.5, None])
    def test_a_non_int_axis_raises(self, bad_axis) -> None:
        """A non-integer axis is rejected by ``__init__``."""
        with pytest.raises(ValueError, match="axis must be an integer"):
            SigSoftmax(axis=bad_axis)

    def test_an_out_of_range_axis_raises_from_call(self) -> None:
        """``call`` range-checks the axis against the rank it actually sees."""
        layer = SigSoftmax(axis=3)
        x = np.zeros((2, 4), dtype="float32")

        with pytest.raises(ValueError, match=r"out of range.*rank 2"):
            layer(x)

    def test_an_out_of_range_axis_raises_from_compute_output_shape(self) -> None:
        """``compute_output_shape`` enforces the same range as ``call``.

        Both read ``common.axis_is_in_range``. A symbolic build that skipped
        this check would be told a shape the forward pass cannot produce.
        """
        layer = SigSoftmax(axis=-4)

        with pytest.raises(ValueError, match=r"out of range.*rank 3"):
            layer.compute_output_shape((2, 4, 5))

    def test_compute_output_shape_answers_on_an_unbuilt_layer(self) -> None:
        """The shape comes from stored config, with no built attributes read."""
        layer = SigSoftmax(axis=-2)

        assert not layer.built
        assert layer.compute_output_shape((2, 4, 5)) == (2, 4, 5)

    def test_compute_output_shape_matches_the_realised_forward_shape(self) -> None:
        """The declared shape equals the shape the forward pass produces."""
        layer = SigSoftmax(axis=-2)
        x = np.random.default_rng(1).standard_normal((3, 4, 5)).astype("float32")

        declared = layer.compute_output_shape(x.shape)
        realised = keras.ops.convert_to_numpy(layer(x)).shape

        assert declared == realised

    def test_get_config_carries_axis_and_reconstructs(self) -> None:
        """``axis`` survives ``get_config`` / ``from_config``."""
        config = SigSoftmax(axis=-2).get_config()

        assert config["axis"] == -2

        rebuilt = SigSoftmax.from_config(config)

        assert rebuilt.axis == -2

    def test_the_layer_is_the_module_function(self) -> None:
        """The layer output equals ``sigsoftmax``'s exactly, to 0.0.

        Pins the layer as a wrapper. A second derivation inside ``call``
        would drift from the function at float32 rounding scale and redden
        here, since the bound is exact equality rather than a tolerance.
        """
        x = np.random.default_rng(2).standard_normal((8, 6)).astype("float32")

        from_layer = keras.ops.convert_to_numpy(SigSoftmax()(x))
        from_function = keras.ops.convert_to_numpy(sigsoftmax(x))

        assert np.abs(from_layer - from_function).max() == 0.0

    def test_axis_minus_two_normalises_along_that_axis(self) -> None:
        """With ``axis=-2`` the rank-3 output sums to 1 along axis -2 only.

        The last-axis sums are asserted to be away from 1 as well, so the arm
        cannot pass for a layer that ignored ``axis`` and normalised the last
        dimension. On this fixture the last-axis sums range over
        [0.5502415, 1.8734096], so the measured deviation from 1 is 0.8734096
        against a threshold of 0.1.
        """
        x = np.random.default_rng(3).standard_normal((2, 4, 5)).astype("float32")

        y = keras.ops.convert_to_numpy(SigSoftmax(axis=-2)(x))

        np.testing.assert_allclose(
            y.sum(axis=-2), np.ones((2, 5)), atol=1e-6, rtol=0.0
        )
        assert np.abs(y.sum(axis=-1) - 1.0).max() > 0.1
