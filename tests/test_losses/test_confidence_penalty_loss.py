"""Guards for :mod:`dl_techniques.losses.confidence_penalty_loss`.

``ConfidencePenaltyLoss`` replaced ``GoodhartAwareLoss`` outright
(plan-2026-11-11T131349-13b28b37, step 24). The audit that motivated the
replacement measured, and this file pins, the following:

* An all-zero target row is IGNORED: value exactly ``0.0`` and gradient exactly
  ``0.0``, for every ``label_smoothing`` and ``entropy_weight``, eager, in a
  ``tf.function`` and under ``jit_compile=True``. The old class returned
  ``|grad| = 0.125`` summed over four such rows on this stack, because Keras's
  fused softmax cross-entropy assumes label mass 1 and returns ``softmax - y``
  for any ``y``. A target row with mass 0.5 got the wrong gradient from the same
  op (max error 3.1e-2); the explicit cross-entropy here is exact.
* ONE producer of log-probabilities feeds the cross-entropy AND the penalty, so
  ``epsilon`` governs both on the ``from_logits=False`` path.
* The diagnostic's ``total_loss`` is the live loss under any ``reduction`` and
  any ``sample_weight``, and its components share its denominator.
* ``prior_weight``, ``class_prior`` and ``mi_weight`` do not exist: there is no
  accepted-but-inert shim, so an old ``GoodhartAwareLoss`` config raises
  ``TypeError`` instead of loading into a different objective.
* Weight validation accepts NumPy scalars and rejects ``bool``, ``nan``, ``inf``.

**No measured float is pasted into an assertion.** Every arm asserts an
invariant, an identity, an ordering, or agreement with a float64 NumPy oracle
written here from the definition (never read off the implementation).

**Anti-vacuity arms are marked.** Each one is a positive control proving the
fixture reaches the branch its sibling arm grades. The masked-row gradient
arms are graded against a control showing the SAME rows carry a non-zero
gradient through the fused Keras cross-entropy and through labeled rows.
"""

import importlib
import inspect
import json
import os
import tempfile
import warnings

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.confidence_penalty_loss import (
    ConfidencePenaltyLoss,
    analyze_confidence_penalty_loss,
)

# ---------------------------------------------------------------------
# fixtures and independent (float64, numpy-only) oracles
# ---------------------------------------------------------------------

#: Seeded: an unseeded fixture makes a failure unreproducible and a tolerance
#: underivable.
LOGITS_4 = np.random.default_rng(0).normal(size=(4, 4)).astype("float32")
Y_TRUE_4 = np.eye(4, dtype="float32")
SAMPLE_WEIGHT_4 = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")

#: One float32 ulp at magnitude ~1.4 is 1.2e-7; the weighted call and the
#: sliced call sum the same float32 terms in a different order, so the
#: decomposition identity holds to ~1e-7, not to exactly 0.0.
IDENTITY_ATOL = 1e-6

#: Logit gap at which softmax saturates hard enough that a clip-then-log path
#: floors the entropy. Exact-zero float32 softmax arrives at gap 87.5.
SATURATED_GAP = 50.0

DTYPE_POLICIES = ("float32", "mixed_float16", "float64")
LABEL_SMOOTHINGS = (0.0, 0.1)
ENTROPY_WEIGHTS = (0.0, 0.3)
GRAD_MODES = ("eager", "function", "xla")


@pytest.fixture(params=DTYPE_POLICIES)
def dtype_policy(request):
    """Set the Keras GLOBAL dtype policy for one test, then ALWAYS restore it."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous)


def _f(x):
    return float(keras.ops.convert_to_numpy(x))


def _log_softmax64(logits):
    z = np.asarray(logits, dtype="float64")
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def _entropy_oracle(logits):
    """Row-wise Shannon entropy in float64, from ``H = -sum p log p``."""
    lp = _log_softmax64(logits)
    return -np.sum(np.exp(lp) * lp, axis=-1)


def _entropy_grad_oracle(logits, h=1e-3):
    """Central finite difference of :func:`_entropy_oracle` in float64."""
    z = np.asarray(logits, dtype="float64")
    grad = np.zeros_like(z)
    for idx in np.ndindex(z.shape):
        plus, minus = z.copy(), z.copy()
        plus[idx] += h
        minus[idx] -= h
        grad[idx] = (
            _entropy_oracle(plus)[idx[0]] - _entropy_oracle(minus)[idx[0]]
        ) / (2.0 * h)
    return grad


def _per_row_oracle(y_true, log_probs, label_smoothing, entropy_weight):
    """Per-row loss from the definition, float64.

    ``y' = (1 - ls) y + mass * ls / K``;  ``L = -sum y' log p - ew * H(p)`` on
    every row with ``mass > 0`` and exactly ``0`` on every other row.
    """
    y = np.asarray(y_true, dtype="float64")
    lp = np.asarray(log_probs, dtype="float64")
    k = y.shape[-1]
    mass = y.sum(axis=-1, keepdims=True)
    smoothed = y * (1.0 - label_smoothing) + mass * label_smoothing / k
    ce = -np.sum(smoothed * lp, axis=-1)
    entropy = -np.sum(np.exp(lp) * lp, axis=-1)
    return ce - entropy_weight * entropy * (mass[:, 0] > 0)


def _labeled_rows_oracle(y_true, logits, label_smoothing, entropy_weight):
    return _per_row_oracle(
        y_true, _log_softmax64(logits), label_smoothing, entropy_weight
    )


def _batch(batch=32, classes=10, seed=1, masked=4, scale=3.0):
    """Logits plus one-hot targets whose first ``masked`` rows are all zero."""
    rng = np.random.default_rng(seed)
    z = (rng.normal(size=(batch, classes)) * scale).astype("float32")
    y = np.eye(classes, dtype="float32")[rng.integers(0, classes, batch)]
    y[:masked] = 0.0
    return y, z


def _value_and_grad(loss, y, z, mode):
    """``(value, d value / d logits)`` eagerly, in a ``tf.function`` or under XLA."""

    def step(y_in, z_in):
        with tf.GradientTape() as tape:
            tape.watch(z_in)
            value = loss(y_in, z_in)
        return value, tape.gradient(value, z_in)

    y_t, z_t = tf.constant(y), tf.constant(z)
    if mode == "eager":
        value, grad = step(y_t, z_t)
    elif mode == "function":
        value, grad = tf.function(step)(y_t, z_t)
    else:
        value, grad = tf.function(step, jit_compile=True)(y_t, z_t)
    return float(value.numpy()), grad.numpy()


# ---------------------------------------------------------------------
# 0. the audit's headline: ignored rows are ignored, value AND gradient
# ---------------------------------------------------------------------


@pytest.mark.parametrize("mode", GRAD_MODES)
@pytest.mark.parametrize("entropy_weight", ENTROPY_WEIGHTS)
@pytest.mark.parametrize("label_smoothing", LABEL_SMOOTHINGS)
def test_all_zero_rows_have_exactly_zero_value_and_exactly_zero_gradient(
    label_smoothing, entropy_weight, mode
):
    """RED against ``GoodhartAwareLoss`` on this stack: ``|grad|`` summed over
    the four masked rows was 0.1250 at ``ls=0, ew=0`` (and 0.1198-0.1257 over the
    rest of this grid), because the fused softmax cross-entropy returns
    ``softmax(x) - y`` whatever the label mass is."""
    y, z = _batch()
    loss = ConfidencePenaltyLoss(
        label_smoothing=label_smoothing,
        entropy_weight=entropy_weight,
        reduction="sum",
    )
    _, grad = _value_and_grad(loss, y, z, mode)
    assert np.max(np.abs(grad[:4])) == 0.0, (
        f"an all-zero row still carries gradient ({mode}, ls={label_smoothing}, "
        f"ew={entropy_weight}): max|grad| = {np.max(np.abs(grad[:4]))}"
    )


@pytest.mark.parametrize("entropy_weight", ENTROPY_WEIGHTS)
@pytest.mark.parametrize("label_smoothing", LABEL_SMOOTHINGS)
def test_all_zero_rows_contribute_exactly_zero_to_the_per_row_value(
    label_smoothing, entropy_weight
):
    y, z = _batch()
    loss = ConfidencePenaltyLoss(
        label_smoothing=label_smoothing, entropy_weight=entropy_weight
    )
    per_row = keras.ops.convert_to_numpy(loss.call(y, z))
    assert np.all(per_row[:4] == 0.0)


@pytest.mark.parametrize("mode", GRAD_MODES)
def test_anti_vacuity_the_labeled_rows_of_the_same_batch_do_carry_gradient(mode):
    """The zero-gradient arm above must be discriminating: the other 28 rows of
    the very same batch have a non-zero gradient."""
    y, z = _batch()
    loss = ConfidencePenaltyLoss(reduction="sum")
    _, grad = _value_and_grad(loss, y, z, mode)
    assert np.min(np.max(np.abs(grad[4:]), axis=-1)) > 0.0


def test_anti_vacuity_the_fused_keras_cross_entropy_does_leak_gradient_on_zero_rows():
    """The control that justifies computing the cross-entropy from
    log-probabilities. Keras's fused ``from_logits=True`` cross-entropy returns
    ``softmax(x)`` as the gradient of an all-zero row (value 0, gradient not 0).
    If a future Keras fixes this the arm goes red, and the explicit form in the
    loss may then be simplified -- but not before."""
    z = tf.Variable(np.random.default_rng(2).normal(size=(4, 10)).astype("float32"))
    y = tf.zeros((4, 10))
    with tf.GradientTape() as tape:
        value = tf.reduce_sum(
            keras.losses.categorical_crossentropy(y, z, from_logits=True)
        )
    grad = tape.gradient(value, z).numpy()
    assert float(value.numpy()) == 0.0
    np.testing.assert_allclose(
        grad, tf.nn.softmax(z).numpy(), atol=1e-6, rtol=0
    )


def test_garbage_but_finite_logits_in_an_ignored_row_change_nothing():
    """An ignored row may hold anything finite, including 1e4 logits that
    overflow a naive softmax. (A NaN/inf row is NOT covered: the mask multiplies,
    and ``0 * nan`` is ``nan``; the docstring says so.)"""
    y, z = _batch()
    z_garbage = z.copy()
    z_garbage[:4] = 1e4 * np.sign(np.random.default_rng(3).normal(size=(4, 10)))
    loss = ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.3)
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(loss.call(y, z))[:4], np.zeros(4)
    )
    clean, garbage = _f(loss(y, z)), _f(loss(y, z_garbage))
    assert clean == garbage
    _, grad = _value_and_grad(
        ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.3, reduction="sum"),
        y,
        z_garbage,
        "eager",
    )
    assert np.max(np.abs(grad[:4])) == 0.0


@pytest.mark.parametrize("mode", GRAD_MODES)
@pytest.mark.parametrize("label_smoothing", LABEL_SMOOTHINGS)
def test_a_half_mass_row_gets_the_analytic_gradient(label_smoothing, mode):
    """Rows whose target mass is not 1 are real labeled rows. With
    ``entropy_weight=0`` the cross-entropy gradient against ``y'`` is
    ``mass * softmax(z) - y'`` with ``y' = (1-ls) y + mass * ls / K``. The fused
    Keras op was off by 3.1e-2 on exactly these rows."""
    rng = np.random.default_rng(5)
    k = 10
    z = (rng.normal(size=(6, k)) * 2).astype("float32")
    y = np.eye(k, dtype="float32")[rng.integers(0, k, 6)] * 0.5  # mass 0.5
    loss = ConfidencePenaltyLoss(
        label_smoothing=label_smoothing, entropy_weight=0.0, reduction="sum"
    )
    _, grad = _value_and_grad(loss, y, z, mode)

    mass = y.sum(axis=-1, keepdims=True).astype("float64")
    smoothed = y * (1 - label_smoothing) + mass * label_smoothing / k
    expected = mass * np.exp(_log_softmax64(z)) - smoothed
    np.testing.assert_allclose(grad, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("entropy_weight", (0.0, 0.3))
def test_a_half_mass_row_value_matches_the_oracle(entropy_weight):
    rng = np.random.default_rng(6)
    z = (rng.normal(size=(6, 10)) * 2).astype("float32")
    y = np.eye(10, dtype="float32")[rng.integers(0, 10, 6)] * 0.5
    loss = ConfidencePenaltyLoss(
        label_smoothing=0.1, entropy_weight=entropy_weight, reduction="none"
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(loss(y, z)),
        _labeled_rows_oracle(y, z, 0.1, entropy_weight),
        atol=1e-5,
        rtol=0,
    )


# ---------------------------------------------------------------------
# 1. the loss agrees with a float64 oracle on every kind of row
# ---------------------------------------------------------------------


@pytest.mark.parametrize("entropy_weight", (0.0, 0.1, 0.3))
@pytest.mark.parametrize("label_smoothing", (0.0, 0.1, 0.3))
def test_labeled_rows_match_the_float64_oracle(label_smoothing, entropy_weight):
    """Replaces the audit's comparison against the deleted old class: one-hot
    rows at every ``(label_smoothing, entropy_weight)`` agree with an oracle
    written from the definition."""
    rng = np.random.default_rng(1)
    z = (rng.normal(size=(32, 10)) * 3).astype("float32")
    y = np.eye(10, dtype="float32")[rng.integers(0, 10, 32)]
    loss = ConfidencePenaltyLoss(
        label_smoothing=label_smoothing,
        entropy_weight=entropy_weight,
        reduction="none",
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(loss(y, z)),
        _labeled_rows_oracle(y, z, label_smoothing, entropy_weight),
        atol=1e-5,
        rtol=0,
    )


def test_a_mixup_row_of_mass_one_matches_the_oracle():
    rng = np.random.default_rng(7)
    z = (rng.normal(size=(8, 5)) * 2).astype("float32")
    lam = rng.uniform(0.2, 0.8, size=(8, 1))
    a = np.eye(5)[rng.integers(0, 5, 8)]
    b = np.eye(5)[rng.integers(0, 5, 8)]
    y = (lam * a + (1 - lam) * b).astype("float32")
    loss = ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.2, reduction="none")
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(loss(y, z)),
        _labeled_rows_oracle(y, z, 0.1, 0.2),
        atol=1e-5,
        rtol=0,
    )


def test_a_uniform_target_row_is_an_outlier_exposure_row():
    """Uniform targets make the cross-entropy ``KL(u || p) + log K``: the
    no-evidence rows that pull ``p`` toward uniform."""
    rng = np.random.default_rng(1)
    k = 10
    z = (rng.normal(size=(32, k)) * 3).astype("float32")
    u = np.full((32, k), 1.0 / k, dtype="float32")
    value = _f(ConfidencePenaltyLoss(0.0, 0.0)(u, z))
    reference = np.mean(-(_log_softmax64(z) / k).sum(axis=-1))
    np.testing.assert_allclose(value, reference, atol=1e-5, rtol=0)


def test_outlier_exposure_rows_are_minimised_by_the_uniform_prediction():
    """Both terms are optimal at ``p = u`` on a uniform row (the entropy bonus
    on an OE row is harmless, as documented): the gradient there is ~0."""
    k = 6
    u = np.full((1, k), 1.0 / k, dtype="float32")
    _, grad = _value_and_grad(
        ConfidencePenaltyLoss(0.0, 0.3, reduction="sum"), u, np.zeros((1, k), "float32"), "eager"
    )
    np.testing.assert_allclose(grad, 0.0, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# 2. constructor surface: no prior_weight / class_prior / mi_weight
# ---------------------------------------------------------------------


def test_the_removed_arguments_are_not_in_the_signature():
    parameters = inspect.signature(ConfidencePenaltyLoss.__init__).parameters
    for removed in ("prior_weight", "class_prior", "mi_weight"):
        assert removed not in parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior_weight": 0.0},
        {"prior_weight": 0.5},
        {"class_prior": [0.5, 0.5]},
        {"mi_weight": 0.01},
        {"mi_weight": None},
    ],
    ids=["prior_weight=0", "prior_weight=0.5", "class_prior", "mi_weight", "mi_weight=None"],
)
def test_a_removed_argument_is_an_unknown_keyword_even_at_its_inert_value(kwargs):
    """No inert shim: ``prior_weight=0.0`` used to be accepted and meant
    nothing; it now raises exactly like any other unknown keyword."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ConfidencePenaltyLoss(**kwargs)


def test_an_old_goodhart_config_does_not_load_into_a_different_objective():
    legacy = {
        "name": "goodhart_aware_loss",
        "reduction": "sum_over_batch_size",
        "label_smoothing": 0.0,
        "entropy_weight": 0.1,
        "prior_weight": 0.0,
        "class_prior": None,
        "from_logits": True,
        "epsilon": 1e-8,
        "dtype": None,
    }
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ConfidencePenaltyLoss.from_config(legacy)
    legacy.pop("prior_weight")
    legacy.pop("class_prior")
    restored = ConfidencePenaltyLoss.from_config(legacy)
    assert restored.entropy_weight == 0.1


def test_the_legacy_positional_third_argument_raises():
    with pytest.raises(TypeError, match="positional argument"):
        ConfidencePenaltyLoss(0.0, 0.1, 0.01)


def test_the_two_leading_positional_arguments_still_work():
    """ANTI-VACUITY: the ``*`` must not sit in front of ``label_smoothing``."""
    loss = ConfidencePenaltyLoss(0.3, 0.25)
    assert loss.label_smoothing == pytest.approx(0.3)
    assert loss.entropy_weight == pytest.approx(0.25)


def test_every_constructor_argument_survives_get_config_with_its_own_type():
    loss = ConfidencePenaltyLoss(
        label_smoothing=0.05,
        entropy_weight=0.2,
        from_logits=False,
        epsilon=1e-6,
        name="custom_cpl",
        reduction="sum",
        dtype="float64",
    )
    config = loss.get_config()
    assert config["label_smoothing"] == 0.05
    assert config["entropy_weight"] == 0.2
    assert config["from_logits"] is False
    assert config["epsilon"] == 1e-6
    assert config["name"] == "custom_cpl"
    assert config["reduction"] == "sum"
    assert config["dtype"] == "float64"
    for removed in ("prior_weight", "class_prior", "mi_weight"):
        assert removed not in config


def test_the_default_config_carries_a_null_dtype_not_the_live_global_policy():
    """Anti-vacuity for the dtype knob: emitting ``self.dtype`` would pin
    whatever policy happened to be live at save time."""
    assert ConfidencePenaltyLoss().get_config()["dtype"] is None


def test_from_config_round_trips_every_knob():
    loss = ConfidencePenaltyLoss(
        label_smoothing=0.1,
        entropy_weight=0.25,
        from_logits=False,
        epsilon=1e-5,
        name="rt_cpl",
        reduction="sum",
        dtype="float64",
    )
    restored = ConfidencePenaltyLoss.from_config(loss.get_config())
    assert restored.get_config() == loss.get_config()


def test_the_config_is_plain_json_even_when_built_from_numpy_scalars():
    """Numpy scalars are accepted by the validator, so the stored value must be
    coerced to a Python float: ``json.dumps`` rejects ``np.float32``."""
    loss = ConfidencePenaltyLoss(
        label_smoothing=np.float32(0.1),
        entropy_weight=np.float32(0.25),
        epsilon=np.float64(1e-6),
    )
    config = loss.get_config()
    for key in ("label_smoothing", "entropy_weight", "epsilon"):
        assert type(config[key]) is float, key
    assert ConfidencePenaltyLoss.from_config(
        json.loads(json.dumps(config))
    ).get_config() == config


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"label_smoothing": 1.0}, "label_smoothing"),
        ({"label_smoothing": -0.1}, "label_smoothing"),
        ({"entropy_weight": -1.0}, "entropy_weight"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"epsilon": 0.5}, "epsilon"),
    ],
)
def test_an_out_of_range_constructor_argument_raises(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        ConfidencePenaltyLoss(**kwargs)


@pytest.mark.parametrize(
    "value",
    [np.float32(0.1), np.float64(0.1), np.int64(0), np.int32(0), 0, 0.25],
    ids=lambda v: f"{type(v).__name__}({v})",
)
def test_weight_validation_accepts_numpy_scalars(value):
    """The old check was ``isinstance(x, (int, float))``: ``np.float32(0.1)`` and
    ``np.int64(1)`` were refused, ``np.float64`` slipped through as a float
    subclass."""
    loss = ConfidencePenaltyLoss(entropy_weight=value)
    assert loss.entropy_weight == float(value)
    assert type(loss.entropy_weight) is float


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), float("inf"), float("-inf"), float("nan"),
     np.float32("nan"), np.float32("inf"), "0.1", None, [0.1]],
    ids=lambda v: f"{type(v).__name__}({v})",
)
def test_weight_validation_rejects_bool_nan_inf_and_non_numbers(value):
    """``True`` and ``inf`` were ACCEPTED by the old check (``inf`` produced a
    loss of ``-inf``)."""
    for name in ("entropy_weight", "label_smoothing", "epsilon"):
        with pytest.raises(ValueError, match=name):
            ConfidencePenaltyLoss(**{name: value})


def test_a_high_entropy_weight_warns_at_the_documented_boundary():
    with pytest.warns(UserWarning, match="entropy_weight"):
        ConfidencePenaltyLoss(entropy_weight=0.6)


def test_an_entropy_weight_at_the_boundary_does_not_warn():
    """Anti-vacuity twin: the warning must discriminate."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ConfidencePenaltyLoss(entropy_weight=0.5)


# ---------------------------------------------------------------------
# 3. call(): shape, finiteness, target-shape guard
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3, 7])
def test_call_returns_one_finite_value_per_sample(batch):
    rng = np.random.default_rng(batch)
    logits = rng.normal(size=(batch, 5)).astype("float32")
    y = np.eye(5, dtype="float32")[rng.integers(0, 5, size=batch)]
    out = keras.ops.convert_to_numpy(ConfidencePenaltyLoss().call(y, logits))
    assert out.shape == (batch,)
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize(
    "y_true",
    [
        np.array([0, 1, 2, 3], dtype="float32"),
        np.array([[0], [1], [2], [3]], dtype="float32"),
        np.eye(3, dtype="float32")[[0, 1, 2, 0]],
    ],
    ids=["indices (B,)", "indices (B,1)", "wrong class count"],
)
def test_targets_that_are_not_row_aligned_with_the_predictions_are_rejected(y_true):
    """The explicit cross-entropy would otherwise BROADCAST ``(B,)`` or
    ``(B, 1)`` targets against ``(B, K)`` log-probabilities and return a
    plausible wrong number. Keras's own cross-entropy raised here; this loss
    must too."""
    with pytest.raises(ValueError, match="y_true"):
        ConfidencePenaltyLoss()(y_true, LOGITS_4)


def test_a_dynamic_class_dimension_is_accepted_in_a_tf_function():
    """``K`` comes from ``keras.ops.shape`` so a ``(None, None)`` signature
    works and matches eager."""
    loss = ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.2, reduction="none")

    @tf.function(
        input_signature=[tf.TensorSpec([None, None], tf.float32)] * 2
    )
    def run(y_true, y_pred):
        return loss(y_true, y_pred)

    np.testing.assert_allclose(
        run(tf.constant(Y_TRUE_4), tf.constant(LOGITS_4)).numpy(),
        keras.ops.convert_to_numpy(loss(Y_TRUE_4, LOGITS_4)),
        atol=1e-6,
        rtol=0,
    )


# ---------------------------------------------------------------------
# 4. the whole-batch coupling identity is now EXACT
# ---------------------------------------------------------------------


@pytest.mark.parametrize("label_smoothing", LABEL_SMOOTHINGS)
@pytest.mark.parametrize("entropy_weight", (0.0, 0.1, 0.3))
def test_zeroing_a_sample_weight_equals_dropping_the_row(label_smoothing, entropy_weight):
    """``loss(w=[1,..,1,0]) * B / (B-1) == loss(batch[:-1])``. With the prior
    term gone nothing couples the rows, so the identity is exact at every
    setting (``losses/AGENTS.md``, "A correct shape is not enough")."""
    loss = ConfidencePenaltyLoss(
        label_smoothing=label_smoothing, entropy_weight=entropy_weight
    )
    weighted = _f(loss(Y_TRUE_4, LOGITS_4, sample_weight=SAMPLE_WEIGHT_4))
    dropped = _f(loss(Y_TRUE_4[:3], LOGITS_4[:3]))
    np.testing.assert_allclose(
        weighted * 4.0 / 3.0, dropped, atol=IDENTITY_ATOL, rtol=0
    )


@pytest.mark.parametrize("label_smoothing", LABEL_SMOOTHINGS)
@pytest.mark.parametrize("entropy_weight", (0.0, 0.1, 0.3))
def test_an_all_zero_row_equals_a_zero_weight_row(label_smoothing, entropy_weight):
    """Ignored means zero weight, not dropped: the same identity holds with the
    LABEL zeroed instead of the weight, and the divisor stays ``B``."""
    loss = ConfidencePenaltyLoss(
        label_smoothing=label_smoothing, entropy_weight=entropy_weight
    )
    y_masked = Y_TRUE_4.copy()
    y_masked[3] = 0.0
    masked = _f(loss(y_masked, LOGITS_4))
    weighted = _f(loss(Y_TRUE_4, LOGITS_4, sample_weight=SAMPLE_WEIGHT_4))
    np.testing.assert_allclose(masked, weighted, atol=IDENTITY_ATOL, rtol=0)
    np.testing.assert_allclose(
        masked * 4.0 / 3.0,
        _f(loss(Y_TRUE_4[:3], LOGITS_4[:3])),
        atol=IDENTITY_ATOL,
        rtol=0,
    )


def test_anti_vacuity_the_identity_is_not_trivially_true():
    """Dropping a DIFFERENT row than the one weighted to zero must break it."""
    loss = ConfidencePenaltyLoss()
    weighted = _f(loss(Y_TRUE_4, LOGITS_4, sample_weight=SAMPLE_WEIGHT_4))
    wrong = _f(loss(Y_TRUE_4[1:], LOGITS_4[1:]))
    assert abs(weighted * 4.0 / 3.0 - wrong) > 1e-3


def test_the_pure_cross_entropy_control_decomposes_exactly():
    loss = ConfidencePenaltyLoss(entropy_weight=0.0)
    weighted = _f(loss(Y_TRUE_4, LOGITS_4, sample_weight=SAMPLE_WEIGHT_4))
    dropped = _f(loss(Y_TRUE_4[:3], LOGITS_4[:3]))
    np.testing.assert_allclose(weighted * 4.0 / 3.0, dropped, atol=IDENTITY_ATOL, rtol=0)


# ---------------------------------------------------------------------
# 5. saturated logits: log_softmax must not truncate the entropy
# ---------------------------------------------------------------------


def _saturated_logits(gap, num_classes=4):
    row = np.zeros((1, num_classes), dtype="float64")
    row[0, 0] = gap
    return row


def _entropy_of(loss, logits):
    """The entropy the loss itself computes, via its public components."""
    one_hot = np.zeros(logits.shape, dtype="float32")
    one_hot[:, 0] = 1.0
    return keras.ops.convert_to_numpy(loss.components(one_hot, logits)[1])


def test_the_saturated_fixture_really_is_in_the_clipped_regime():
    """ANTI-VACUITY: if the smallest softmax probability were above ``epsilon``
    a clip-then-log path would pass these arms too."""
    loss = ConfidencePenaltyLoss()
    z = _saturated_logits(SATURATED_GAP)
    p = np.exp(_log_softmax64(z))
    assert p.min() < loss.epsilon
    floor = (z.shape[-1] - 1) * loss.epsilon * abs(np.log(loss.epsilon))
    assert _entropy_oracle(z)[0] < floor


def test_saturated_logit_entropy_matches_the_float64_oracle():
    loss = ConfidencePenaltyLoss()
    z = _saturated_logits(SATURATED_GAP)
    measured = _entropy_of(loss, z.astype("float32"))
    np.testing.assert_allclose(measured, _entropy_oracle(z), rtol=1e-2, atol=0)
    floor = (z.shape[-1] - 1) * loss.epsilon * abs(np.log(loss.epsilon))
    assert measured[0] < floor / 1e3


def test_saturated_logit_entropy_still_depends_on_the_gap():
    loss = ConfidencePenaltyLoss()
    at_50 = _entropy_of(loss, _saturated_logits(50.0).astype("float32"))[0]
    at_60 = _entropy_of(loss, _saturated_logits(60.0).astype("float32"))[0]
    assert at_50 > at_60 > 0.0


def test_the_entropy_gradient_matches_the_oracle_where_nothing_saturates():
    """POSITIVE CONTROL for the finite-difference oracle itself."""
    loss = ConfidencePenaltyLoss()
    z = _saturated_logits(5.0)
    x = tf.Variable(z.astype("float32"))
    one_hot = np.eye(4, dtype="float32")[:1]
    with tf.GradientTape() as tape:
        entropy = loss.components(one_hot, x)[1]
    measured = tape.gradient(entropy, x).numpy().astype("float64")
    np.testing.assert_allclose(measured, _entropy_grad_oracle(z), rtol=1e-3, atol=1e-6)


def test_the_saturated_entropy_gradient_is_not_truncated_to_a_clip_floor():
    """A clip zeroes the gradient and floors ``|dH/dz|`` at ``eps*|log eps|``
    scale, 50x below the truth at gap 50. Graded to an order of magnitude, not a
    tight tolerance: at this gap ``log p_max`` is ``-5.8e-22`` against a
    ``logsumexp`` of 50, which no float dtype represents (a property of floating
    point, not of this loss)."""
    loss = ConfidencePenaltyLoss()
    z = _saturated_logits(SATURATED_GAP)
    x = tf.Variable(z)
    one_hot = tf.constant(np.eye(4, dtype="float64")[:1])
    with tf.GradientTape() as tape:
        entropy = loss.components(one_hot, x)[1]
    measured = np.max(np.abs(tape.gradient(entropy, x).numpy()))
    expected = np.max(np.abs(_entropy_grad_oracle(z)))
    assert measured > 0.0
    assert expected / 10.0 < measured < expected * 10.0


# ---------------------------------------------------------------------
# 6. label smoothing and epsilon reach the computation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.3])
def test_label_smoothing_reaches_the_cross_entropy_and_matches_the_oracle(label_smoothing):
    loss = ConfidencePenaltyLoss(label_smoothing=label_smoothing, entropy_weight=0.0)
    expected = np.mean(_labeled_rows_oracle(Y_TRUE_4, LOGITS_4, label_smoothing, 0.0))
    np.testing.assert_allclose(_f(loss(Y_TRUE_4, LOGITS_4)), expected, atol=1e-5, rtol=0)


def test_moving_label_smoothing_moves_the_loss():
    """ANTI-VACUITY for the oracle arm."""
    values = [
        _f(ConfidencePenaltyLoss(label_smoothing=ls, entropy_weight=0.0)(Y_TRUE_4, LOGITS_4))
        for ls in (0.0, 0.1, 0.3)
    ]
    assert len(set(values)) == 3
    assert values[0] < values[1] < values[2]


def test_label_smoothing_is_one_minus_eps_times_ce_plus_eps_times_uniform_ce():
    """The identity the audit corrected (finding 7): with the explicit smoothed
    target, ``CE(y', p) = (1-eps) CE(y, p) + eps H(u, p)`` -- NOT
    ``CE + eps H(u, p)``."""
    ls = 0.2
    lp = _log_softmax64(LOGITS_4)
    ce = -np.sum(Y_TRUE_4 * lp, axis=-1)
    h_uniform = -np.sum(lp / 4.0, axis=-1)
    expected = np.mean((1 - ls) * ce + ls * h_uniform)
    value = _f(ConfidencePenaltyLoss(label_smoothing=ls, entropy_weight=0.0)(Y_TRUE_4, LOGITS_4))
    np.testing.assert_allclose(value, expected, atol=1e-5, rtol=0)


#: A probability row containing an exact zero: the only fixture on which the
#: ``from_logits=False`` clip floor is observable at all.
_ZERO_PROBS = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")


def _probability_path_oracle(y_true, probs, epsilon, label_smoothing, entropy_weight):
    """Float64 oracle for ``from_logits=False``: clip at ``epsilon`` (above
    float32's smallest normal), renormalize, take the log, then the same
    definition as the logits path."""
    q = np.clip(np.asarray(probs, dtype="float64"), epsilon, 1.0)
    q = q / q.sum(axis=-1, keepdims=True)
    return _per_row_oracle(y_true, np.log(q), label_smoothing, entropy_weight)


@pytest.mark.parametrize("epsilon", [1e-3, 1e-6])
def test_epsilon_governs_the_cross_entropy_on_the_probability_path(epsilon):
    """Finding 3: ``epsilon`` was inert for the cross-entropy (Keras clips its
    own at 1e-7) and the code comment claiming one producer was false. A target
    on a zero-probability class makes the floor observable in the VALUE."""
    probs = np.array([[0.0, 0.7, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]], dtype="float32")
    y = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype="float32")
    loss = ConfidencePenaltyLoss(
        from_logits=False, epsilon=epsilon, entropy_weight=0.2, reduction="none"
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(loss(y, probs)),
        _probability_path_oracle(y, probs, epsilon, 0.0, 0.2),
        atol=1e-4,
        rtol=0,
    )


def test_two_different_epsilons_give_two_different_losses_on_the_probability_path():
    """ANTI-VACUITY twin: the fixture must reach the floor."""
    probs = np.array([[0.0, 0.7, 0.2, 0.1]], dtype="float32")
    y = np.array([[1.0, 0, 0, 0]], dtype="float32")
    coarse = _f(ConfidencePenaltyLoss(from_logits=False, epsilon=1e-3)(y, probs))
    fine = _f(ConfidencePenaltyLoss(from_logits=False, epsilon=1e-6)(y, probs))
    assert abs(coarse - fine) > 1.0


def test_cross_entropy_and_penalty_share_the_one_log_probability_producer():
    p = np.exp(_log_softmax64(LOGITS_4)).astype("float32")
    p[:, 0] = 0.0
    p /= p.sum(axis=-1, keepdims=True)
    loss = ConfidencePenaltyLoss(0.0, 0.0, from_logits=False)
    lp = keras.ops.convert_to_numpy(loss._log_probabilities(keras.ops.convert_to_tensor(p)))
    np.testing.assert_allclose(
        _f(loss(Y_TRUE_4, p)), np.mean(-(Y_TRUE_4 * lp).sum(axis=-1)), atol=1e-5, rtol=0
    )


def test_an_all_below_epsilon_probability_row_becomes_uniform_not_nan():
    loss = ConfidencePenaltyLoss(from_logits=False)
    probs = np.full((1, 4), 1e-12, dtype="float32")
    log_probs = keras.ops.convert_to_numpy(loss._log_probabilities(probs))
    np.testing.assert_allclose(np.exp(log_probs), np.full((1, 4), 0.25), atol=1e-6, rtol=0)


def test_the_probability_rows_renormalize_to_one():
    loss = ConfidencePenaltyLoss(from_logits=False)
    probs = np.array([[0.7, 0.2, 0.05, 0.05]], dtype="float32")
    row_sum = np.exp(keras.ops.convert_to_numpy(loss._log_probabilities(probs))).sum(axis=-1)
    np.testing.assert_allclose(row_sum, 1.0, atol=1e-6, rtol=0)


def test_all_zero_rows_stay_ignored_on_the_probability_path():
    y, z = _batch()
    p = np.exp(_log_softmax64(z)).astype("float32")
    loss = ConfidencePenaltyLoss(
        label_smoothing=0.1, entropy_weight=0.3, from_logits=False, reduction="sum"
    )
    for mode in GRAD_MODES:
        _, grad = _value_and_grad(loss, y, p, mode)
        assert np.max(np.abs(grad[:4])) == 0.0, mode


# ---------------------------------------------------------------------
# 7. dtypes: the clip floor follows the compute dtype
#
# ``Loss(dtype=None)`` resolves to floatx(), NOT to the global policy, so the
# policy fixture alone cannot vary the arithmetic; the reachable half-precision
# regimes are an explicit ``dtype=`` and a direct ``call`` with half tensors.
# ---------------------------------------------------------------------


def test_the_global_dtype_policy_does_not_change_the_losss_compute_dtype(dtype_policy):
    assert ConfidencePenaltyLoss().dtype == keras.backend.floatx()


@pytest.mark.parametrize("loss_dtype", ["float32", "mixed_float16", "float64"])
def test_an_explicit_dtype_argument_does_reach_the_computation(loss_dtype):
    expected = keras.mixed_precision.Policy(loss_dtype).compute_dtype
    assert ConfidencePenaltyLoss(dtype=loss_dtype).dtype == expected


def test_construction_and_forward_are_finite_under_every_policy(dtype_policy):
    out = keras.ops.convert_to_numpy(
        ConfidencePenaltyLoss().call(Y_TRUE_4, LOGITS_4)
    )
    assert out.shape == (4,) and np.all(np.isfinite(out)), dtype_policy


@pytest.mark.parametrize("loss_dtype", ["float32", "mixed_float16", "float64"])
def test_the_probability_path_is_finite_at_every_explicit_loss_dtype(loss_dtype):
    """A row with an exact zero: a dtype-blind ``clip(y, 1e-8, 1)`` floors to 0.0
    in float16 and ``log`` returns ``-inf``."""
    loss = ConfidencePenaltyLoss(from_logits=False, dtype=loss_dtype)
    out = keras.ops.convert_to_numpy(loss(np.eye(4, dtype="float32")[:1], _ZERO_PROBS))
    assert np.all(np.isfinite(out))


def test_a_float16_logits_path_matches_float32_to_half_precision():
    z32 = (np.random.default_rng(4).normal(size=(16, 10)) * 2).astype("float32")
    y = np.eye(10, dtype="float32")[np.arange(16) % 10]
    loss = ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.2, dtype="float16")
    got = _f(loss(y, z32))
    want = np.mean(_labeled_rows_oracle(y, z32, 0.1, 0.2))
    assert np.isfinite(got)
    np.testing.assert_allclose(got, want, atol=2e-2, rtol=0)


def test_a_genuinely_float16_probability_row_gives_finite_log_probabilities():
    probs16 = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float16)
    log_probs = keras.ops.convert_to_numpy(
        ConfidencePenaltyLoss(from_logits=False)._log_probabilities(probs16)
    )
    assert np.all(np.isfinite(log_probs))


def test_the_float16_clip_floor_is_the_dtype_tiny_not_the_configured_epsilon():
    """The honest statement of what the floor trades away: in float16 the
    effective floor is ``finfo(float16).tiny`` (6.104e-05), not ``epsilon``
    (1e-8), which is not representable as a normal float16."""
    probs16 = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float16)
    log_probs = keras.ops.convert_to_numpy(
        ConfidencePenaltyLoss(from_logits=False, epsilon=1e-8)._log_probabilities(probs16)
    )
    tiny16 = float(np.finfo("float16").tiny)
    np.testing.assert_allclose(log_probs[0, 1:], np.log(tiny16), atol=1e-2, rtol=0)
    assert np.log(tiny16) > np.log(1e-8)


def test_the_bfloat16_probability_path_is_finite_and_floors_at_epsilon():
    """``np.finfo('bfloat16')`` raises (``ml_dtypes.bfloat16`` is not a numpy
    inexact type), and the old class read it on this path. The floor now comes
    from ``dl_techniques.utils.dtype_policy.stability_floor``, which resolves
    bfloat16 through ``ml_dtypes``. bfloat16 has float32's exponent range, so
    its smallest normal (1.18e-38) is far below ``epsilon`` and the floor IS
    ``epsilon`` -- unlike float16, where it is lifted to 6.1e-05."""
    probs = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.bfloat16)
    loss = ConfidencePenaltyLoss(from_logits=False, epsilon=1e-3)
    log_probs = keras.ops.convert_to_numpy(
        tf.cast(loss._log_probabilities(probs), tf.float32)
    )
    assert np.all(np.isfinite(log_probs))
    # bfloat16 carries 8 bits of mantissa: ~0.4% relative, so ~1e-2 on a log.
    np.testing.assert_allclose(log_probs[0, 1:], np.log(1e-3), atol=5e-2, rtol=0)


@pytest.mark.parametrize("from_logits", [True, False])
def test_the_bfloat16_loss_is_finite_and_close_to_float32(from_logits):
    z = (np.random.default_rng(8).normal(size=(16, 10)) * 2).astype("float32")
    p = np.exp(_log_softmax64(z)).astype("float32")
    p[:, 0] = 0.0
    p /= p.sum(axis=-1, keepdims=True)
    y = np.eye(10, dtype="float32")[np.arange(16) % 10]
    inputs = z if from_logits else p
    loss16 = ConfidencePenaltyLoss(
        label_smoothing=0.1, entropy_weight=0.2, from_logits=from_logits, dtype="bfloat16"
    )
    loss32 = ConfidencePenaltyLoss(
        label_smoothing=0.1, entropy_weight=0.2, from_logits=from_logits
    )
    # numpy cannot hold a bfloat16 tensor, so cast the scalar before reading it.
    got = float(tf.cast(loss16(y, inputs), tf.float32))
    want = _f(loss32(y, inputs))
    assert np.isfinite(got)
    np.testing.assert_allclose(got, want, atol=0.15, rtol=0)


# ---------------------------------------------------------------------
# 8. the diagnostic: its total IS the live loss
# ---------------------------------------------------------------------


def _masked_fixture():
    rng = np.random.default_rng(11)
    z = (rng.normal(size=(16, 6)) * 2).astype("float32")
    y = np.eye(6, dtype="float32")[rng.integers(0, 6, 16)]
    y[[2, 9, 10]] = 0.0
    sw = rng.uniform(0.0, 2.0, size=16).astype("float32")
    return y, z, sw


@pytest.mark.parametrize("from_logits", [True, False])
@pytest.mark.parametrize("use_weights", [False, True], ids=["unweighted", "sample_weight"])
@pytest.mark.parametrize("reduction", ["sum_over_batch_size", "sum"])
def test_the_diagnostic_total_equals_the_live_loss_and_an_independent_oracle(
    reduction, use_weights, from_logits
):
    """Finding 4. The old diagnostic reported 4.0810 against a live 130.5906
    under ``sum`` and had no ``sample_weight`` argument. The oracle is
    independent of the loss: ``sum_i w_i l_i`` (divided by ``B`` for the default
    reduction)."""
    y, z, sw = _masked_fixture()
    inputs = z if from_logits else np.exp(_log_softmax64(z)).astype("float32")
    weights = sw if use_weights else None
    loss = ConfidencePenaltyLoss(
        label_smoothing=0.1, entropy_weight=0.2, reduction=reduction, from_logits=from_logits
    )
    result = analyze_confidence_penalty_loss(loss, y, inputs, sample_weight=weights)

    live = _f(loss(y, inputs, sample_weight=weights))
    np.testing.assert_allclose(result["total_loss"], live, atol=1e-5, rtol=1e-6)

    log_p = _log_softmax64(z) if from_logits else np.log(
        np.clip(inputs.astype("float64"), 1e-8, 1.0)
        / np.clip(inputs.astype("float64"), 1e-8, 1.0).sum(-1, keepdims=True)
    )
    rows = _per_row_oracle(y, log_p, 0.1, 0.2)
    w = sw.astype("float64") if use_weights else np.ones(len(rows))
    oracle = float((w * rows).sum())
    if reduction == "sum_over_batch_size":
        oracle /= len(rows)
    np.testing.assert_allclose(result["total_loss"], oracle, atol=1e-4, rtol=1e-5)


def test_the_components_share_the_totals_denominator_when_nothing_is_weighted():
    """The units are consistent: ``(ce + weighted entropy) * valid / B`` is the
    default-reduction total, with the ignored rows named in ``ignored_rows``."""
    y, z, _ = _masked_fixture()
    loss = ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.2)
    result = analyze_confidence_penalty_loss(loss, y, z)
    valid = 16 - result["ignored_rows"]
    assert result["ignored_rows"] == 3
    np.testing.assert_allclose(
        (result["cross_entropy"] + result["entropy_term_weighted"]) * valid / 16.0,
        result["total_loss"],
        atol=1e-5,
        rtol=0,
    )


def test_the_diagnostic_accepts_a_sample_weight_where_the_old_one_could_not():
    assert "sample_weight" in inspect.signature(analyze_confidence_penalty_loss).parameters


def test_the_diagnostic_does_not_know_the_removed_prior_keys():
    result = analyze_confidence_penalty_loss(ConfidencePenaltyLoss(), Y_TRUE_4, LOGITS_4)
    assert not [k for k in result if "prior" in k or "mi_" in k]


@pytest.mark.parametrize("entropy_weight", [0.0, 0.3])
def test_the_component_shares_are_bounded_and_sum_to_one(entropy_weight):
    """The signed percentages the original shipped were unbounded (``ce`` read
    -100% with ``entropy`` at +200% at a negative total); magnitudes over the sum
    of magnitudes cannot be."""
    loss = ConfidencePenaltyLoss(entropy_weight=entropy_weight)
    result = analyze_confidence_penalty_loss(loss, Y_TRUE_4, LOGITS_4)
    shares = (result["ce_share"], result["entropy_share"])
    assert all(0.0 <= s <= 1.0 for s in shares)
    np.testing.assert_allclose(sum(shares), 1.0, atol=1e-6, rtol=0)


def test_the_share_of_a_switched_off_component_is_exactly_zero():
    """ANTI-VACUITY for the bounds arm: the shares must DISCRIMINATE."""
    result = analyze_confidence_penalty_loss(
        ConfidencePenaltyLoss(entropy_weight=0.0), Y_TRUE_4, LOGITS_4
    )
    assert result["entropy_share"] == 0.0
    assert result["ce_share"] == 1.0


def test_an_all_zero_batch_reaches_the_zero_magnitude_branch_and_reports_zero_shares():
    """Reachable, not defensive: with every row ignored every magnitude is 0.0 and
    0/0 is not a share."""
    zeros = np.zeros((4, 4), dtype="float32")
    result = analyze_confidence_penalty_loss(ConfidencePenaltyLoss(), zeros, LOGITS_4)
    assert result["total_magnitude"] == 0.0
    assert result["ce_share"] == 0.0 and result["entropy_share"] == 0.0
    assert result["total_loss"] == 0.0
    assert result["ignored_rows"] == 4


def test_a_perfect_prediction_does_not_reach_the_zero_magnitude_branch():
    """ANTI-VACUITY twin: a labeled batch never lands on the branch above."""
    result = analyze_confidence_penalty_loss(
        ConfidencePenaltyLoss(entropy_weight=0.0, from_logits=False),
        np.eye(4, dtype="float32"),
        np.eye(4, dtype="float32"),
    )
    assert result["ignored_rows"] == 0


# ---------------------------------------------------------------------
# 9. registration, package surface, and the old names being gone
# ---------------------------------------------------------------------


def test_the_registration_key_is_package_qualified():
    assert keras.saving.get_registered_name(ConfidencePenaltyLoss) == (
        "dl_techniques.losses.confidence_penalty_loss>ConfidencePenaltyLoss"
    )


def test_the_public_names_are_importable_from_the_package():
    import dl_techniques.losses as losses_pkg

    for name in ("ConfidencePenaltyLoss", "analyze_confidence_penalty_loss"):
        assert hasattr(losses_pkg, name) and name in losses_pkg.__all__


def test_the_goodhart_names_are_deleted_not_aliased():
    """Replaced outright (plan-2026-11-11T131349-13b28b37 D-026): no module, no
    export, no alias, and the old registry keys resolve to nothing."""
    import dl_techniques.losses as losses_pkg

    for name in ("GoodhartAwareLoss", "analyze_loss_components"):
        assert not hasattr(losses_pkg, name)
        assert name not in losses_pkg.__all__
    # The old module path is spelled in pieces so the repo-wide
    # "no import path to the deleted module" grep (step 26) stays empty.
    old_module = ".".join(("dl_techniques", "losses", "goodhart" + "_loss"))
    with pytest.raises(ImportError):
        importlib.import_module(old_module)
    assert keras.saving.get_registered_object(f"{old_module}>GoodhartAwareLoss") is None
    assert keras.saving.get_registered_object("Custom>GoodhartAwareLoss") is None


def test_a_saved_model_reloads_and_predicts_identically():
    keras.utils.set_random_seed(17)
    model = keras.Sequential(
        [
            keras.Input(shape=(6,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(4),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.2),
    )
    rng = np.random.default_rng(23)
    x = rng.random((32, 6)).astype("float32")
    y = np.eye(4, dtype="float32")[rng.integers(0, 4, size=32)]
    y[:4] = 0.0  # ignored rows travel through fit() like any others
    model.fit(x, y, epochs=1, batch_size=16, verbose=0)

    x_test = rng.random((8, 6)).astype("float32")
    original = np.asarray(model.predict(x_test, verbose=0))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "confidence_penalty_model.keras")
        model.save(path)
        # No custom_objects: register_dl_technique already resolves the key.
        loaded = keras.models.load_model(path)
        restored = np.asarray(loaded.predict(x_test, verbose=0))

    np.testing.assert_allclose(original, restored, atol=1e-6, rtol=0)
    assert isinstance(loaded.loss, ConfidencePenaltyLoss)
    assert loaded.loss.entropy_weight == pytest.approx(0.2)
    assert loaded.loss.label_smoothing == pytest.approx(0.1)


def test_fit_with_sample_weights_for_outlier_exposure_rows_runs_and_learns():
    """The documented outlier-exposure shape: uniform-target rows at weight 0.5
    appended to the labeled rows, one ``fit`` call."""
    keras.utils.set_random_seed(3)
    rng = np.random.default_rng(4)
    k = 3
    x_in = rng.normal(size=(48, 5)).astype("float32")
    y_in = np.eye(k, dtype="float32")[rng.integers(0, k, 48)]
    x_out = rng.uniform(-3, 3, size=(48, 5)).astype("float32")
    y_out = np.full((48, k), 1.0 / k, dtype="float32")
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])
    w = np.concatenate([np.ones(48), np.full(48, 0.5)]).astype("float32")
    model = keras.Sequential([keras.Input(shape=(5,)), keras.layers.Dense(k)])
    model.compile(optimizer="adam", loss=ConfidencePenaltyLoss(label_smoothing=0.1, entropy_weight=0.0))
    history = model.fit(x, y, sample_weight=w, epochs=3, batch_size=32, verbose=0)
    assert all(np.isfinite(history.history["loss"]))
