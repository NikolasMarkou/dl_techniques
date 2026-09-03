"""Tests for the sigsoftmax activation functions and the ``SigSoftmax`` layer.

Covers the closed-form comparison against an independent float64 NumPy oracle,
the all-negative regression guard that the max-shift formulation fails, the
not-plain-softmax mechanism arm, the layer's construction, axis validation,
shape and config contract, the ``exp(log_sigsoftmax)`` identity, a ``.keras``
round trip compared on values, the dtype floor in both directions, gradient
flow, XLA agreement, the factory, and the package export contract.

``tf`` is imported for the gradient tape only. The package ``__init__`` is
imported only by the export-contract arms; every other arm imports the subject
from its defining module.
"""

import os
import tempfile

import numpy as np
import keras
import pytest
import tensorflow as tf
from scipy.special import expit

import dl_techniques.layers.activations as activations
from dl_techniques.layers.activations.factory import (
    STRICT_DROPPED_KEY_MARKER,
    create_activation_layer,
)
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


# ---------------------------------------------------------------------
# The exp/log identity. Invariant 3: `sigsoftmax` is `exp(log_sigsoftmax)`,
# never a second derivation.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sigsoftmax_is_exp_of_log_sigsoftmax(dtype: str) -> None:
    """``sigsoftmax`` equals ``exp(log_sigsoftmax)`` to exactly 0.0.

    Both functions share ``_log_sigsoftmax_widened``, and for float32 and
    float64 the reduction dtype equals the input dtype, so the two paths
    differ only in the position of an ``exp`` that runs on identical bits.
    Measured max absolute difference on a ``(16, 7)`` fixture: 0.0 in
    float32 and 0.0 in float64. The bound is exact equality rather than a
    tolerance, so a future edit that re-derives one function from the other
    by a different route reddens here.
    """
    x = np.random.default_rng(4).standard_normal((16, 7)).astype(dtype)

    direct = keras.ops.convert_to_numpy(sigsoftmax(x))
    via_log = keras.ops.convert_to_numpy(keras.ops.exp(log_sigsoftmax(x)))

    assert direct.dtype == np.dtype(dtype)
    assert np.abs(direct.astype(np.float64) - via_log.astype(np.float64)).max() == 0.0


def test_the_exp_log_identity_is_approximate_in_float16() -> None:
    """In float16 the identity holds to 2.44e-04, not to 0.0. Measured.

    The two functions cast at different points. ``sigsoftmax`` exponentiates
    in the widened float32 reduction dtype and casts the probabilities down;
    ``exp(log_sigsoftmax(x))`` casts the log-probabilities down to float16
    first and exponentiates there. The widening is what makes them differ, so
    the exact-0.0 bound asserted above for float32 and float64 is not
    available here and is not weakened silently: the measured max absolute
    difference on the same ``(16, 7)`` fixture is 2.44140625e-04, which is
    float16's own resolution near 1, and the bound below is one order above
    it.
    """
    x = np.random.default_rng(4).standard_normal((16, 7)).astype("float16")

    direct = keras.ops.convert_to_numpy(sigsoftmax(x)).astype(np.float64)
    via_log = keras.ops.convert_to_numpy(
        keras.ops.exp(log_sigsoftmax(x))
    ).astype(np.float64)

    assert np.abs(direct - via_log).max() < 1e-3


# ---------------------------------------------------------------------
# Serialization.
# ---------------------------------------------------------------------


def test_a_saved_model_reproduces_its_output_values() -> None:
    """A ``.keras`` round trip preserves the OUTPUT VALUES, not just the shape.

    A shape-only round trip is satisfied by a model that restored zero
    weights, so the comparison is on values at ``atol=1e-6, rtol=0.0``;
    measured max absolute difference is exactly 0.0. ``axis=-2`` rather than
    the default, so a ``get_config`` that dropped ``axis`` would rebuild a
    last-axis layer and disagree. ``training=False`` is passed explicitly to
    both arms.
    """
    inputs = keras.Input(shape=(4, 5))
    model = keras.Model(inputs, SigSoftmax(axis=-2)(inputs))

    x = np.random.default_rng(7).standard_normal((3, 4, 5)).astype("float32")
    before = keras.ops.convert_to_numpy(model(x, training=False))

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "sigsoftmax_model.keras")
        model.save(path)
        restored = keras.models.load_model(path)

    after = keras.ops.convert_to_numpy(restored(x, training=False))

    np.testing.assert_allclose(after, before, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------
# The dtype floor (guide rule L-30), both directions.
#
# There is no enumerating guard file for SigSoftmax to join:
# `test_the_dtype_floor_never_narrows.py` is scoped entirely to
# `RoutingProbabilitiesLayer`. The property is pinned here instead.
# ---------------------------------------------------------------------


def test_a_float64_caller_is_not_narrowed_to_float32(float64_policy) -> None:
    """Under a genuine float64 policy the reduction stays float64.

    The mutation this is RED against is an unconditional
    ``reduction_dtype = "float32"`` in ``_log_sigsoftmax_widened``. Measured
    on this fixture: the shipped path agrees with the float64 NumPy oracle at
    1.665e-16, roughly one float64 ulp; the mutated path reads 8.810e-08,
    float32's own floor. The two readings are five orders apart, so the
    1e-12 bound is not a knife edge -- any threshold in [1e-14, 1e-09] grades
    them identically.

    The realised dtypes are asserted first. The ``float64_policy`` fixture
    only makes float64 reachable; without these asserts the arm would agree
    with float32 to eight digits and could not fail. The layer is built
    inside the test because ``set_floatx`` does not re-point an
    already-materialised policy.
    """
    x = np.random.default_rng(0).standard_normal((64, 20))
    assert x.dtype == np.float64

    layer = SigSoftmax()
    assert layer.compute_dtype == "float64"

    realised_input = keras.ops.convert_to_numpy(keras.ops.convert_to_tensor(x))
    assert realised_input.dtype == np.float64

    y = keras.ops.convert_to_numpy(layer(x))
    assert y.dtype == np.float64

    assert np.abs(y - _reference_sigsoftmax(x)).max() < 1e-12


def test_the_all_negative_float16_row_stays_finite(mixed_float16_policy) -> None:
    """A float16 all-negative row is finite and correct on the log scale.

    The float16 analogue of the all-negative regression guard. The max-shift
    form returns ``nan nan nan`` on this exact row (measured, in float16), so
    a wrong implementation still fails here.

    The lane-level comparison runs on the log scale, where the information is
    representable. The true first probability is ``exp(-30) = 9.36e-14`` and
    float16's smallest subnormal is 6e-08, so that lane is exactly 0.0 for
    ANY correct implementation and a "no exact zeros" assertion could not be
    satisfied. In log space the measured reading is
    ``[-30, -10, -4.5776e-05]`` against the asymptotic ``2 * (z - max z) =
    [-30, -10, 0]``, i.e. a max absolute deviation of 4.5776e-05.
    Probability-space assertions are kept to the ones that are true there:
    finite, non-negative, summing to 1 (measured 1.0000454, which is why the
    sum bound is 1e-3 and not 1e-6).

    The premise is asserted rather than assumed: the row genuinely underflows
    in half precision, so the arm cannot pass for free on an input that never
    exercised the regime.
    """
    row = np.array([[-30.0, -20.0, -15.0]], dtype="float16")

    # Anti-vacuity premise, computed in NumPy independently of whatever dtype
    # `call()` chose: the first lane is unrepresentable in float16.
    assert np.float16(np.exp(np.float64(-30.0))) == np.float16(0.0)
    assert np.exp(np.float64(-30.0)) < np.finfo(np.float16).smallest_subnormal

    layer = SigSoftmax()
    assert layer.compute_dtype == "float16"

    y = keras.ops.convert_to_numpy(layer(row))
    assert y.dtype == np.dtype("float16")

    y64 = y.astype(np.float64)
    assert np.all(np.isfinite(y64)), f"sigsoftmax was not finite, observed {y}"
    assert np.all(y64 >= 0.0)
    np.testing.assert_allclose(y64.sum(axis=-1), np.ones(1), atol=1e-3, rtol=0.0)

    log_y = keras.ops.convert_to_numpy(log_sigsoftmax(row)).astype(np.float64)
    assert np.all(np.isfinite(log_y)), (
        f"log_sigsoftmax was not finite, observed {log_y}"
    )
    np.testing.assert_allclose(
        log_y, _asymptotic_log_sigsoftmax(row), atol=1e-3, rtol=0.0
    )


# ---------------------------------------------------------------------
# Gradients and XLA.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("bias_init", [0.0, -50.0])
def test_the_upstream_kernel_receives_a_finite_non_zero_gradient(
    bias_init: float,
) -> None:
    """Gradient flows back through the layer to a preceding Dense kernel.

    Parametrized over the bias initializer so that ``bias_init=-50.0`` drives
    every logit strongly negative -- the regime in which the max-shift form
    produces NaN outputs and therefore NaN gradients. Measured max absolute
    kernel gradient: 0.6026 at ``bias_init=0.0`` and 1.1749 at
    ``bias_init=-50.0``; the logit range in the second case is
    [-51.94, -48.71], so the all-negative regime is genuinely reached rather
    than assumed.
    """
    keras.utils.set_random_seed(11)
    x = np.random.default_rng(6).standard_normal((4, 5)).astype("float32")

    dense = keras.layers.Dense(
        6, bias_initializer=keras.initializers.Constant(bias_init)
    )
    layer = SigSoftmax()

    with tf.GradientTape() as tape:
        logits = dense(keras.ops.convert_to_tensor(x))
        loss = keras.ops.sum(layer(logits) ** 2)

    logits_np = keras.ops.convert_to_numpy(logits)
    if bias_init == -50.0:
        assert logits_np.max() < 0.0, "the all-negative regime was not reached"

    gradient = keras.ops.convert_to_numpy(tape.gradient(loss, dense.kernel))

    assert np.all(np.isfinite(gradient)), (
        f"the Dense kernel gradient was not finite, observed {gradient}"
    )
    assert np.abs(gradient).max() > 0.0


def test_xla_matches_eager(assert_xla_matches_eager) -> None:
    """A traced ``jit_compile=True`` graph agrees with the eager call.

    The regime ``fit()`` runs in is a traced ``tf.function``, not eager. The
    fixture's call itself asserts that XLA can lower the graph at all.
    """
    x = np.random.default_rng(5).standard_normal((4, 3, 6)).astype("float32")

    # atol derived from a measurement, not tuned: max|eager - xla| on this
    # fixture reads exactly 0.0 (GPU, TF32 enabled). The layer carries no
    # matmul, so it has no TF32-sensitive stage; 1e-6 leaves float32 headroom
    # over a reading of zero rather than pinning the bound to it.
    deviation = assert_xla_matches_eager(
        SigSoftmax(axis=-2), x, 1e-6, "SigSoftmax(axis=-2)"
    )

    assert deviation == 0.0


# ---------------------------------------------------------------------
# Factory and package surface.
# ---------------------------------------------------------------------


class TestActivationFactory:
    """``create_activation_layer('sigsoftmax')`` builds the layer correctly."""

    def test_the_default_axis_survives_the_factory(self) -> None:
        """The registry's declared default reaches the constructor."""
        layer = create_activation_layer("sigsoftmax")

        assert isinstance(layer, SigSoftmax)
        assert layer.axis == -1

    def test_a_supplied_axis_survives_the_factory(self) -> None:
        """A caller-supplied ``axis`` is not dropped on the way through."""
        assert create_activation_layer("sigsoftmax", axis=-2).axis == -2

    def test_an_undeclared_keyword_raises(self) -> None:
        """The factory rejects rather than silently filtering a typo.

        The package's factory contract: a misspelled keyword that is dropped
        instead of raising has shipped dead configuration repo-wide before.

        The marker is compared as a SUBSTRING, not with ``match=``: it is the
        literal ``"unsupported parameter(s)"``, whose ``(s)`` is a regex group
        that never matches the message it was taken from.
        """
        with pytest.raises(ValueError) as raised:
            create_activation_layer("sigsoftmax", bogus_key=1)

        assert STRICT_DROPPED_KEY_MARKER in str(raised.value)


class TestExportContract:
    """The package's public surface after the three new names were added."""

    def test_all_declares_thirty_four_names(self) -> None:
        """``__all__`` grew from 31 to 34: two functions and one layer."""
        assert len(activations.__all__) == 34

    def test_every_exported_name_resolves(self) -> None:
        """Every name in ``__all__`` is actually importable from the package.

        A grep proves a row exists in ``__all__``; it cannot prove the row is
        true. A documentation-repair pass in this repository once shipped four
        unimportable names past eyeballing twice, and only a ``hasattr`` sweep
        caught it.
        """
        missing = [n for n in activations.__all__ if not hasattr(activations, n)]

        assert missing == [], f"names in __all__ that do not resolve: {missing}"
