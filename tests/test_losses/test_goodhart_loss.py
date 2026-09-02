"""Guards for :mod:`dl_techniques.losses.goodhart_loss`.

Until 2026-09-02 this class had ZERO dedicated coverage: exactly one node in a
27,680-node suite touched it, and that node stayed green with both regularizers
deleted. In that blind spot the class shipped a term documented as the mutual
information ``I(X; Yhat)`` which was in fact ``H(mean_p) - mean H(p_i)`` added
with a POSITIVE sign -- maximized at the accurate confident classifier and
minimized at marginal collapse, the exact opposite of the intended effect --
plus a ``clip(eps) -> log`` path that floored the confidence penalty's entropy
at ``(K-1) * eps * |log eps|`` and truncated its gradient by 50x, plus a
batch-scalar confidence penalty that broke ``sample_weight`` row decomposition.

**Every assertion below was proven RED against the defect it guards.** The
mutation was injected into the real ``src/`` module (never a copied tree --
``pyproject.toml``'s ``pythonpath = ["src"]`` overrides ``PYTHONPATH``, so a
mutant in a scratch copy gives a FALSE GREEN) and the failure message recorded.

**No measured float is pasted into an assertion.** Three separate readings of
the same 4x4 fixture in this repo gave three different totals (109.18 / 109.11 /
108.06) because the value drifts with what else the process constructed first.
Every arm asserts an INVARIANT, an IDENTITY, an ORDERING, or agreement with an
oracle recomputed here in float64 -- never an observation.

**Anti-vacuity arms are marked.** Each one is a positive control proving the
fixture actually reaches the branch its sibling arm grades.
"""

import os
import tempfile
import warnings

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.goodhart_loss import (
    GoodhartAwareLoss,
    analyze_loss_components,
)

# ---------------------------------------------------------------------
# fixtures and independent (float64, numpy-only) oracles
# ---------------------------------------------------------------------

#: The standard 4x4 logit/one-hot fixture. Seeded: an unseeded fixture makes a
#: failure unreproducible and a tolerance underivable.
LOGITS_4 = np.random.default_rng(0).normal(size=(4, 4)).astype("float32")
Y_TRUE_4 = np.eye(4, dtype="float32")

SAMPLE_WEIGHT_4 = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")

#: 0.7 ulp at magnitude ~1.4. The per-row confidence penalty makes the weighted
#: call and the sliced call sum float32 terms in a different ORDER, so the
#: decomposition identity holds to ~4e-08, not to exactly 0.0. Measured
#: 2026-09-02, CPU, float32.
IDENTITY_ATOL = 1e-6

#: Logit gap at which softmax saturates hard enough that the pre-fix
#: ``clip(epsilon)`` bites. Exact-zero float32 softmax arrives at gap 87.5, so
#: 50 is saturated but not degenerate.
SATURATED_GAP = 50.0

DTYPE_POLICIES = ("float32", "mixed_float16", "float64")


@pytest.fixture(params=DTYPE_POLICIES)
def dtype_policy(request):
    """Set the Keras GLOBAL dtype policy for one test, then ALWAYS restore it."""
    previous = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy(request.param)
    try:
        yield request.param
    finally:
        keras.mixed_precision.set_global_policy(previous)


def _saturated_logits(gap, num_classes=4):
    """One row, one dominant class: ``[gap, 0, 0, ...]``."""
    row = np.zeros((1, num_classes), dtype="float64")
    row[0, 0] = gap
    return row


def _entropy_oracle(logits):
    """Row-wise Shannon entropy computed in float64 from the definition.

    Written from ``H = -sum p log p`` with ``p = softmax(logits)``, NOT from the
    implementation. An oracle copied from the code it grades passes forever.
    """
    z = np.asarray(logits, dtype="float64")
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log(p), axis=-1)


def _entropy_grad_oracle(logits, h=1e-3):
    """Central finite difference of :func:`_entropy_oracle` in float64.

    The entropy of a saturated row is proportional to ``exp(-gap)``, so a
    central difference in float64 is accurate to many digits even though the
    value itself is ~1e-20.
    """
    z = np.asarray(logits, dtype="float64")
    grad = np.zeros_like(z)
    for idx in np.ndindex(z.shape):
        plus = z.copy()
        minus = z.copy()
        plus[idx] += h
        minus[idx] -= h
        grad[idx] = (
            _entropy_oracle(plus)[idx[0]] - _entropy_oracle(minus)[idx[0]]
        ) / (2.0 * h)
    return grad


def _clip_entropy_floor(num_classes, epsilon):
    """The PRE-FIX entropy floor ``(K-1) * eps * |log eps|``.

    Derived here from ``K`` and ``epsilon``, never pasted: it is K-DEPENDENT
    (5.53e-07 at K=4, 1.66e-06 at K=10), so a K-free constant would be wrong
    for every other fixture.
    """
    return (num_classes - 1) * epsilon * abs(np.log(epsilon))


def _collapsed_and_diverse(num_classes=4, rows_per_class=3, gap=8.0):
    """Two batches with EQUAL cross-entropy and EQUAL per-row entropy.

    Both are confidently correct, so cross-entropy and the confidence penalty
    cannot produce the ordering the anti-collapse arm asserts; only the
    batch-MARGINAL differs. ``diverse`` covers every class uniformly;
    ``collapsed`` puts every row on class 0.
    """
    n = num_classes * rows_per_class
    diverse_cls = np.arange(n) % num_classes
    collapsed_cls = np.zeros(n, dtype="int64")

    def _build(classes):
        logits = np.zeros((n, num_classes), dtype="float32")
        logits[np.arange(n), classes] = gap
        y = np.zeros((n, num_classes), dtype="float32")
        y[np.arange(n), classes] = 1.0
        return y, logits

    return _build(collapsed_cls), _build(diverse_cls)


def _f(x):
    return float(keras.ops.convert_to_numpy(x))


# ---------------------------------------------------------------------
# 1. (a) every constructor knob is pinned, with the instrument matching
#    its class
# ---------------------------------------------------------------------


def test_every_constructor_argument_survives_get_config_with_its_own_type():
    loss = GoodhartAwareLoss(
        label_smoothing=0.05,
        entropy_weight=0.2,
        prior_weight=0.3,
        class_prior=[1.0, 3.0],
        from_logits=False,
        epsilon=1e-6,
        name="custom_goodhart",
        reduction="sum",
        dtype="float64",
    )
    config = loss.get_config()

    assert config["label_smoothing"] == 0.05
    assert config["entropy_weight"] == 0.2
    assert config["prior_weight"] == 0.3
    assert config["class_prior"] == [1.0, 3.0]
    assert config["from_logits"] is False
    assert config["epsilon"] == 1e-6
    assert config["name"] == "custom_goodhart"
    assert config["reduction"] == "sum"
    # The CALLER's dtype argument, not the resolved global policy.
    assert config["dtype"] == "float64"
    # The removed knob must not reappear in a saved config.
    assert "mi_weight" not in config


def test_the_default_config_carries_a_null_dtype_not_the_live_global_policy():
    """Anti-vacuity for the dtype knob: emitting ``self.dtype`` would pin
    whatever policy happened to be live at save time."""
    loss = GoodhartAwareLoss()
    assert loss.get_config()["dtype"] is None


def test_from_config_round_trips_every_knob():
    loss = GoodhartAwareLoss(
        label_smoothing=0.1,
        entropy_weight=0.25,
        prior_weight=0.5,
        class_prior=[0.2, 0.3, 0.5],
        from_logits=False,
        epsilon=1e-5,
        name="rt_goodhart",
        reduction="sum",
        dtype="float64",
    )
    restored = GoodhartAwareLoss.from_config(loss.get_config())
    assert restored.get_config() == loss.get_config()


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"label_smoothing": 1.0}, "label_smoothing"),
        ({"label_smoothing": -0.1}, "label_smoothing"),
        ({"entropy_weight": -1.0}, "Entropy weight"),
        ({"prior_weight": -1.0}, "Prior weight"),
        ({"epsilon": 0.0}, "Epsilon"),
        ({"epsilon": 0.5}, "Epsilon"),
        ({"class_prior": []}, "non-empty"),
        ({"class_prior": [1.0, np.nan]}, "finite"),
        ({"class_prior": [1.0, 0.0]}, "strictly positive"),
        ({"class_prior": [1.0, -1.0]}, "strictly positive"),
    ],
)
def test_an_out_of_range_constructor_argument_raises(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        GoodhartAwareLoss(**kwargs)


def test_a_high_entropy_weight_warns_at_the_documented_boundary():
    """The threshold must match the range the docstring advertises ([0.001,
    0.5]); it used to warn only above 1.0."""
    with pytest.warns(UserWarning, match="entropy_weight"):
        GoodhartAwareLoss(entropy_weight=0.6)


def test_an_entropy_weight_inside_the_documented_range_does_not_warn():
    """Anti-vacuity twin for the warning above: the warning must discriminate,
    and its boundary must be the one the docstring advertises."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        GoodhartAwareLoss(entropy_weight=0.5)


def test_a_class_prior_whose_length_disagrees_with_y_pred_raises():
    loss = GoodhartAwareLoss(prior_weight=1.0, class_prior=[0.5, 0.5])
    with pytest.raises(ValueError, match="class_prior has length"):
        loss(Y_TRUE_4, LOGITS_4)


# ---------------------------------------------------------------------
# 2. (i) the removed mi_weight knob raises loudly
# ---------------------------------------------------------------------


def test_passing_mi_weight_by_keyword_raises_and_names_prior_weight():
    """A silent drop would let an archived config deserialize into a DIFFERENT
    objective. NOTE: ``mi_weight`` now sits LAST in the signature, so a legacy
    POSITIONAL third argument sets ``prior_weight`` and does not raise -- see
    the twin below."""
    with pytest.raises(ValueError, match="prior_weight"):
        GoodhartAwareLoss(mi_weight=0.01)


def test_the_mi_weight_message_says_the_old_term_had_the_wrong_sign():
    with pytest.raises(ValueError) as excinfo:
        GoodhartAwareLoss(mi_weight=0.0)
    message = str(excinfo.value)
    assert "mi_weight" in message
    assert "prior_weight" in message
    assert "H(mean_p) - mean H(p_i)" in message


def test_the_legacy_positional_third_argument_is_prior_weight_not_a_raise():
    """Recorded, not asserted as desirable: the signature reorder means an old
    positional call silently becomes a small CORRECTLY-signed prior term. Zero
    callers of any kind were measured, and this arm exists so the behaviour is
    pinned rather than discovered."""
    loss = GoodhartAwareLoss(0.0, 0.1, 0.01)
    assert loss.prior_weight == pytest.approx(0.01)


# ---------------------------------------------------------------------
# 3. (d) call() is rank 1, batch-length and finite
# ---------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 3, 7])
def test_call_returns_one_finite_value_per_sample(batch):
    rng = np.random.default_rng(batch)
    logits = rng.normal(size=(batch, 5)).astype("float32")
    y = np.eye(5, dtype="float32")[rng.integers(0, 5, size=batch)]
    out = keras.ops.convert_to_numpy(GoodhartAwareLoss().call(y, logits))
    assert out.ndim == 1
    assert out.shape[0] == batch
    assert np.all(np.isfinite(out))


def test_call_stays_rank_one_when_the_prior_term_is_active():
    out = keras.ops.convert_to_numpy(
        GoodhartAwareLoss(prior_weight=1.0).call(Y_TRUE_4, LOGITS_4)
    )
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))


def test_integer_targets_are_rejected_rather_than_silently_broadcast():
    loss = GoodhartAwareLoss()
    with pytest.raises(ValueError):
        loss(np.array([0, 1, 2, 3], dtype="float32"), LOGITS_4)


# ---------------------------------------------------------------------
# 4. (e) the confidence penalty is PER SAMPLE, so sample_weight decomposes
# ---------------------------------------------------------------------


def test_zeroing_a_sample_weight_equals_dropping_the_row_at_the_default():
    """The identity, not a pasted float: at ``prior_weight = 0.0`` the loss
    decomposes exactly per row, so ``loss(w=[1,1,1,0]) * B/(B-1)`` is
    ``loss(batch[:3])``. A batch-scalar confidence penalty breaks this."""
    loss = GoodhartAwareLoss()
    weighted = _f(loss(Y_TRUE_4, LOGITS_4, sample_weight=SAMPLE_WEIGHT_4))
    dropped = _f(loss(Y_TRUE_4[:3], LOGITS_4[:3]))
    np.testing.assert_allclose(
        weighted * 4.0 / 3.0,
        dropped,
        atol=IDENTITY_ATOL,
        rtol=0,
        err_msg=(
            "the confidence penalty does not ride per row: zeroing a "
            "sample_weight is not equivalent to dropping the row"
        ),
    )


def test_the_decomposition_identity_breaks_once_the_prior_term_is_on():
    """The documented batch-level residual, and the anti-vacuity twin of the
    arm above: the identity is a LIVE instrument, not vacuously true."""
    loss = GoodhartAwareLoss(prior_weight=0.5)
    weighted = _f(loss(Y_TRUE_4, LOGITS_4, sample_weight=SAMPLE_WEIGHT_4))
    dropped = _f(loss(Y_TRUE_4[:3], LOGITS_4[:3]))
    residual = abs(weighted * 4.0 / 3.0 - dropped)
    assert residual > 1e-4, (
        "the prior term is documented as irreducibly batch-level, but the "
        f"per-row identity still holds to {residual}"
    )


def test_the_pure_cross_entropy_control_decomposes_exactly():
    """Anti-vacuity: with BOTH regularizers off the identity holds to exactly
    0.0, so any residual seen above is caused by a regularizer and by nothing
    else in the fixture."""
    loss = GoodhartAwareLoss(entropy_weight=0.0, prior_weight=0.0)
    weighted = _f(loss(Y_TRUE_4, LOGITS_4, sample_weight=SAMPLE_WEIGHT_4))
    dropped = _f(loss(Y_TRUE_4[:3], LOGITS_4[:3]))
    np.testing.assert_allclose(
        weighted * 4.0 / 3.0, dropped, atol=IDENTITY_ATOL, rtol=0
    )


# ---------------------------------------------------------------------
# 5. (f) saturated logits: log_softmax must not truncate the entropy or
#    its gradient
# ---------------------------------------------------------------------


def test_the_saturated_fixture_really_is_in_the_clipped_regime():
    """ANTI-VACUITY for the two arms below. If the smallest softmax probability
    were above ``epsilon`` the clip would never bite and the arms would pass
    with the defect restored."""
    loss = GoodhartAwareLoss()
    z = _saturated_logits(SATURATED_GAP)
    p = np.exp(z - z.max())
    p = p / p.sum()
    assert p.min() < loss.epsilon, (
        f"smallest probability {p.min()} is above epsilon {loss.epsilon}: "
        "this fixture does not exercise the clip"
    )
    assert _entropy_oracle(z)[0] < _clip_entropy_floor(z.shape[-1], loss.epsilon)


def test_saturated_logit_entropy_matches_the_float64_oracle():
    """The clip+log path floors this at ``(K-1) * eps * |log eps|``, which is
    thirteen orders of magnitude above the true value."""
    loss = GoodhartAwareLoss()
    z = _saturated_logits(SATURATED_GAP)
    measured = keras.ops.convert_to_numpy(
        loss._conditional_entropy(loss._log_probabilities(z.astype("float32")))
    )
    expected = _entropy_oracle(z)
    np.testing.assert_allclose(
        measured,
        expected,
        rtol=1e-2,
        atol=0,
        err_msg="saturated-logit entropy disagrees with the float64 oracle",
    )
    assert measured[0] < _clip_entropy_floor(z.shape[-1], loss.epsilon) / 1e3


def test_saturated_logit_entropy_still_depends_on_the_gap():
    """A clipped entropy is CONSTANT in the gap -- it reads the same floor at
    50 and at 60. The true entropy falls by ~e^-10."""
    loss = GoodhartAwareLoss()
    at_50 = _f(
        loss._conditional_entropy(
            loss._log_probabilities(_saturated_logits(50.0).astype("float32"))
        )[0]
    )
    at_60 = _f(
        loss._conditional_entropy(
            loss._log_probabilities(_saturated_logits(60.0).astype("float32"))
        )[0]
    )
    assert at_50 > at_60 > 0.0, (
        f"entropy is not strictly decreasing in the logit gap: "
        f"H(50) = {at_50}, H(60) = {at_60}"
    )


def test_the_entropy_gradient_matches_the_oracle_exactly_where_nothing_clips():
    """POSITIVE CONTROL for the finite-difference oracle itself. At a gap of 5
    every probability is far above ``epsilon``, so autodiff and the oracle must
    agree to many digits. If this arm ever fails, the oracle is broken and the
    saturated arm below grades nothing."""
    loss = GoodhartAwareLoss()
    z = _saturated_logits(5.0)
    x = tf.Variable(z)
    with tf.GradientTape() as tape:
        entropy = loss._conditional_entropy(loss._log_probabilities(x))
    measured = tape.gradient(entropy, x).numpy().astype("float64")
    np.testing.assert_allclose(
        measured, _entropy_grad_oracle(z), rtol=1e-6, atol=0
    )


def test_the_saturated_entropy_gradient_is_not_truncated_to_the_clip_floor():
    """The clip zeroes the gradient through it and floors ``|dH/dz|`` at
    ``eps * |log eps|``-scale, a factor of 50 below the truth.

    Graded to an ORDER OF MAGNITUDE against the float64 oracle, not to a tight
    tolerance, and here is the measured reason: at gap 50, ``log p_max`` is
    ``-5.8e-22`` while ``logsumexp`` is ``50``, so NO float dtype -- float64
    included -- can represent it. The ``-p_max log p_max`` term is lost, which
    biases the autodiff gradient by exactly the factor ``50 / 49``, and makes
    the ``dH/dz_max`` entry read 0.0 instead of ``-2.8e-20``. That is a property
    of floating point, not of this loss. The truncation this arm exists to
    catch is 50x, well outside the factor-of-10 window."""
    loss = GoodhartAwareLoss()
    z = _saturated_logits(SATURATED_GAP)
    x = tf.Variable(z)
    with tf.GradientTape() as tape:
        entropy = loss._conditional_entropy(loss._log_probabilities(x))
    measured = np.max(np.abs(tape.gradient(entropy, x).numpy().astype("float64")))
    expected = np.max(np.abs(_entropy_grad_oracle(z)))

    assert measured > 0.0, "the confidence penalty has no gradient at all"
    assert expected / 10.0 < measured < expected * 10.0, (
        "the confidence penalty's gradient at a saturated logit is off by more "
        f"than an order of magnitude from the float64 oracle: {measured} vs "
        f"{expected} (the clip truncates it)"
    )


# ---------------------------------------------------------------------
# 6. (g) the anti-collapse term is correctly SIGNED
# ---------------------------------------------------------------------


def test_the_collapse_fixture_really_does_move_the_batch_marginal():
    """ANTI-VACUITY for the sign arm: the two batches must differ in their
    MARGINAL, and must NOT differ in their cross-entropy or per-row entropy,
    or the ordering below would prove nothing about the prior term."""
    (yc, zc), (yd, zd) = _collapsed_and_diverse()
    loss = GoodhartAwareLoss()

    marginal_c = keras.ops.convert_to_numpy(
        keras.ops.mean(keras.ops.softmax(zc, axis=-1), axis=0)
    )
    marginal_d = keras.ops.convert_to_numpy(
        keras.ops.mean(keras.ops.softmax(zd, axis=-1), axis=0)
    )
    assert np.max(np.abs(marginal_c - marginal_d)) > 0.5, (
        f"marginals are not distinguishable: {marginal_c} vs {marginal_d}"
    )

    per_row_c = keras.ops.convert_to_numpy(loss.call(yc, zc))
    per_row_d = keras.ops.convert_to_numpy(loss.call(yd, zd))
    np.testing.assert_allclose(per_row_c, per_row_d, atol=1e-6, rtol=0)


def test_a_collapsed_marginal_scores_strictly_higher_than_a_diverse_one():
    """The sign the removed ``mi_weight`` term had BACKWARDS. Only the ordering
    is asserted -- the magnitude drifts with the fixture and with float32
    summation order."""
    (yc, zc), (yd, zd) = _collapsed_and_diverse()
    loss = GoodhartAwareLoss(prior_weight=1.0)
    collapsed = _f(loss(yc, zc))
    diverse = _f(loss(yd, zd))
    assert collapsed > diverse, (
        "the anti-collapse term is signed backwards: a collapsed batch "
        f"marginal scored {collapsed}, a diverse one {diverse}"
    )


def test_the_two_batches_are_exactly_equal_at_the_default_prior_weight():
    """ANTI-VACUITY twin: the ordering above is produced by the prior term and
    by nothing else in the fixture."""
    (yc, zc), (yd, zd) = _collapsed_and_diverse()
    loss = GoodhartAwareLoss()
    # atol, not exact equality: the two batches present the same 12 float32
    # row losses in a different ORDER, so their means reassociate and differ in
    # the last bits (~5e-11). The prior term the sibling arm measures moves
    # these values by ~1e+00, eight orders above this tolerance.
    np.testing.assert_allclose(
        _f(loss(yc, zc)), _f(loss(yd, zd)), atol=1e-6, rtol=0
    )


def test_the_prior_term_is_zero_when_the_marginal_matches_a_matching_prior():
    """A KL is zero only at agreement; giving the collapsed batch its OWN
    marginal as the prior must switch the penalty off."""
    (yc, zc), _ = _collapsed_and_diverse()
    marginal = keras.ops.convert_to_numpy(
        keras.ops.mean(keras.ops.softmax(zc, axis=-1), axis=0)
    ).astype("float64")
    matched = GoodhartAwareLoss(
        prior_weight=1.0, class_prior=list(marginal)
    )
    uniform = GoodhartAwareLoss(prior_weight=1.0)
    assert _f(matched(yc, zc)) < _f(uniform(yc, zc))
    np.testing.assert_allclose(
        _f(matched(yc, zc)), _f(GoodhartAwareLoss()(yc, zc)), atol=1e-5, rtol=0
    )


# ---------------------------------------------------------------------
# 7. (h) N = 1: the prior term is live where the deleted MI term was dead
# ---------------------------------------------------------------------


def test_at_batch_size_one_the_prior_term_is_nonzero_and_has_a_gradient():
    """The deleted MI term was identically 0.0 with a zero gradient at N=1,
    because a single row's marginal IS that row. ``KL(prior || mean_p)`` is
    not."""
    y = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    z = np.array([[3.0, 0.0, -1.0, 0.5]], dtype="float32")

    with_prior = _f(GoodhartAwareLoss(prior_weight=1.0)(y, z))
    without = _f(GoodhartAwareLoss()(y, z))
    assert with_prior != without

    x = tf.Variable(z)
    loss = GoodhartAwareLoss(prior_weight=1.0)
    with tf.GradientTape() as tape:
        term = loss._prior_matching_regularization(loss._log_probabilities(x))
    grad = tape.gradient(term, x).numpy()
    assert np.all(np.isfinite(grad))
    assert np.max(np.abs(grad)) > 0.0, (
        "the anti-collapse term has no gradient at N=1, which is the defect "
        "the deleted MI term had"
    )


# ---------------------------------------------------------------------
# 8. (j) analyze_loss_components: bounded shares that agree with the loss
# ---------------------------------------------------------------------


@pytest.mark.parametrize("prior_weight", [0.0, 0.5])
def test_the_component_shares_are_bounded_and_sum_to_one(prior_weight):
    """The shipped signed percentages were unbounded: ``ce_contrib_pct`` read
    over 100 on an ordinary fixture, and at a negative total ``ce`` read -100%
    with ``entropy`` at +200%."""
    loss = GoodhartAwareLoss(prior_weight=prior_weight)
    result = analyze_loss_components(loss, Y_TRUE_4, LOGITS_4)
    shares = {k: result[k] for k in ("ce_share", "entropy_share", "prior_share")}
    for name, value in shares.items():
        assert 0.0 <= value <= 1.0, f"{name} is out of [0, 1]: {value}"
    np.testing.assert_allclose(sum(shares.values()), 1.0, atol=1e-6, rtol=0)


@pytest.mark.parametrize("from_logits", [True, False])
def test_the_diagnostic_total_is_the_same_number_the_live_loss_returns(from_logits):
    """The diagnostic used to run its OWN log path, so on ``from_logits=False``
    it skipped the clip that ``call()`` applied and reported a different number.

    ``atol``, not bit equality: the diagnostic reduces the entropy term to a
    scalar BEFORE adding it, while ``call()`` adds it per row, so the two float32
    sums reassociate and land ~1e-07 apart. Bit equality was measured to hold on
    one fixture and not on another -- it is not a property of the code."""
    y_pred = LOGITS_4
    if not from_logits:
        y_pred = keras.ops.convert_to_numpy(
            keras.ops.softmax(keras.ops.convert_to_tensor(LOGITS_4), axis=-1)
        )
    loss = GoodhartAwareLoss(prior_weight=0.5, from_logits=from_logits)
    result = analyze_loss_components(loss, Y_TRUE_4, y_pred)
    np.testing.assert_allclose(
        result["total_loss"],
        _f(loss(Y_TRUE_4, y_pred)),
        atol=1e-6,
        rtol=0,
        err_msg="the diagnostic disagrees with the live loss",
    )


def test_the_diagnostic_reports_no_mi_keys_and_does_report_prior_keys():
    result = analyze_loss_components(GoodhartAwareLoss(), Y_TRUE_4, LOGITS_4)
    assert not [k for k in result if "mi_" in k]
    assert "prior_term_unweighted" in result
    assert "prior_term_weighted" in result
    assert "prior_weight" in result


def test_the_share_of_a_switched_off_component_is_exactly_zero():
    """ANTI-VACUITY for the bounds arm: the shares must DISCRIMINATE, not just
    happen to sit inside [0, 1]."""
    result = analyze_loss_components(
        GoodhartAwareLoss(entropy_weight=0.0, prior_weight=0.0),
        Y_TRUE_4,
        LOGITS_4,
    )
    assert result["entropy_share"] == 0.0
    assert result["prior_share"] == 0.0
    assert result["ce_share"] == 1.0


# ---------------------------------------------------------------------
# 9. (b, c) serialization: registration key and a .keras round trip
# ---------------------------------------------------------------------


def test_the_registration_key_is_package_qualified():
    assert keras.saving.get_registered_name(GoodhartAwareLoss) == (
        "dl_techniques.losses.goodhart_loss>GoodhartAwareLoss"
    )


def test_both_public_names_are_importable_from_the_package():
    import dl_techniques.losses as losses_pkg

    assert hasattr(losses_pkg, "GoodhartAwareLoss")
    assert hasattr(losses_pkg, "analyze_loss_components")


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
        loss=GoodhartAwareLoss(entropy_weight=0.2, prior_weight=0.3),
    )
    rng = np.random.default_rng(23)
    x = rng.random((32, 6)).astype("float32")
    y = np.eye(4, dtype="float32")[rng.integers(0, 4, size=32)]
    model.fit(x, y, epochs=1, batch_size=16, verbose=0)

    x_test = rng.random((8, 6)).astype("float32")
    original = np.asarray(model.predict(x_test, verbose=0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "goodhart_model.keras")
        model.save(path)
        # No custom_objects: register_dl_technique already resolves the key. A
        # dict keyed by BARE CLASS NAME is ignored by Keras 3 entirely.
        loaded = keras.models.load_model(path)
        restored = np.asarray(loaded.predict(x_test, verbose=0))

    np.testing.assert_allclose(original, restored, atol=1e-6, rtol=0)
    assert isinstance(loaded.loss, GoodhartAwareLoss)
    assert loaded.loss.prior_weight == pytest.approx(0.3)
    assert loaded.loss.entropy_weight == pytest.approx(0.2)


# ---------------------------------------------------------------------
# 10. (k) dtype policies
# ---------------------------------------------------------------------


def test_construction_and_forward_are_finite_under_every_policy(dtype_policy):
    loss = GoodhartAwareLoss(prior_weight=0.5)
    out = keras.ops.convert_to_numpy(loss.call(Y_TRUE_4, LOGITS_4))
    assert out.shape == (4,)
    assert np.all(np.isfinite(out)), f"non-finite loss under {dtype_policy}"


def test_the_from_logits_false_path_is_finite_under_every_policy(dtype_policy):
    """``epsilon = 1e-8`` is below float16's smallest normal (~6.1e-5), so a
    dtype-blind clip floors to 0.0 and ``log`` returns -inf."""
    probs = keras.ops.convert_to_numpy(
        keras.ops.softmax(keras.ops.convert_to_tensor(LOGITS_4), axis=-1)
    )
    loss = GoodhartAwareLoss(from_logits=False, prior_weight=0.5)
    out = keras.ops.convert_to_numpy(loss.call(Y_TRUE_4, probs))
    assert np.all(np.isfinite(out)), f"non-finite loss under {dtype_policy}"


def test_an_all_below_epsilon_probability_row_becomes_uniform_not_nan():
    """Documented, not fixed: after clip and renormalize such a row is uniform,
    and the gradient through the clip is zero."""
    loss = GoodhartAwareLoss(from_logits=False)
    probs = np.full((1, 4), 1e-12, dtype="float32")
    log_probs = keras.ops.convert_to_numpy(loss._log_probabilities(probs))
    np.testing.assert_allclose(
        np.exp(log_probs), np.full((1, 4), 0.25), atol=1e-6, rtol=0
    )


def test_the_from_logits_false_rows_renormalize_to_one():
    loss = GoodhartAwareLoss(from_logits=False)
    probs = np.array([[0.7, 0.2, 0.05, 0.05]], dtype="float32")
    row_sum = np.exp(
        keras.ops.convert_to_numpy(loss._log_probabilities(probs))
    ).sum(axis=-1)
    np.testing.assert_allclose(row_sum, 1.0, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------
# 11. the anti-collapse term's KL DIRECTION
#
# The shipped term is ``KL(q || mean_p)``. The plausible wrong direction is
# ``KL(mean_p || q)``. Both are positive on a collapsed marginal, so an
# ORDERING arm cannot separate them -- and the first suite's ordering arm did
# not. The separator is BOUNDEDNESS, re-derived here and not transcribed:
#
#     KL(mean_p || u) = sum_k mp_k log(mp_k * K) = log K - H(mean_p)
#
# and ``H >= 0``, so the reverse direction is bounded ABOVE by ``log K`` for
# EVERY marginal, with equality only in the limit of total collapse. The
# shipped direction with a uniform ``q`` is
#
#     KL(u || mean_p) = -log K - mean_k log(mp_k)
#
# which grows like ``(K-1)/K * gap`` as the collapse deepens and is unbounded.
# Measured on the K=4 fixture below: shipped 1.667 / 4.615 / 10.614 at
# gap 4 / 8 / 16, against a reverse-direction oracle of 1.124 / 1.377 / 1.386
# which never crosses ``log 4 = 1.3863``.
# ---------------------------------------------------------------------


def _collapsed_batch(gap, rows=12, num_classes=4):
    """One batch whose every row is confidently class 0, at a chosen ``gap``."""
    logits = np.zeros((rows, num_classes), dtype="float32")
    logits[:, 0] = gap
    y = np.zeros((rows, num_classes), dtype="float32")
    y[:, 0] = 1.0
    return y, logits


def _prior_term(loss, logits):
    return _f(loss._prior_matching_regularization(loss._log_probabilities(logits)))


def test_the_uniform_anti_collapse_term_exceeds_log_k_which_the_reverse_kl_cannot():
    """The DIRECTION guard. ``KL(mean_p || u) = log K - H(mean_p) <= log K`` for
    every marginal that exists, so a value above ``log K`` is reachable only by
    ``KL(u || mean_p)`` -- the direction this loss ships."""
    _, logits = _collapsed_batch(gap=8.0)
    num_classes = logits.shape[-1]
    log_k = float(np.log(num_classes))

    measured = _prior_term(GoodhartAwareLoss(prior_weight=1.0), logits)
    assert measured > log_k, (
        "the anti-collapse term is at most log K on a hard-collapsed marginal "
        f"({measured} <= {log_k}), which is the signature of the REVERSED "
        "direction KL(mean_p || q); the shipped direction KL(q || mean_p) is "
        "unbounded above"
    )


def test_the_uniform_anti_collapse_penalty_is_unbounded_as_the_collapse_deepens():
    """Second half of the direction guard: the shipped direction GROWS without
    bound as the collapse deepens, while the reverse saturates at ``log K``.
    Doubling the logit gap must add at least another ``log K``."""
    num_classes = 4
    log_k = float(np.log(num_classes))
    loss = GoodhartAwareLoss(prior_weight=1.0)

    shallow = _prior_term(loss, _collapsed_batch(gap=8.0, num_classes=num_classes)[1])
    deep = _prior_term(loss, _collapsed_batch(gap=16.0, num_classes=num_classes)[1])

    assert deep - shallow > log_k, (
        "deepening the collapse barely moved the anti-collapse term "
        f"({shallow} -> {deep}); a term that saturates is the reverse KL, "
        "which is capped at log K = {0}".format(log_k)
    )


def test_the_uniform_branch_agrees_with_the_explicit_uniform_prior_branch():
    """ANTI-VACUITY for the two arms above: the ``class_prior is None``
    shortcut must compute the same number as passing the uniform prior
    explicitly, so the direction guard grades the same expression both ways."""
    _, logits = _collapsed_batch(gap=8.0)
    implicit = _prior_term(GoodhartAwareLoss(prior_weight=1.0), logits)
    explicit = _prior_term(
        GoodhartAwareLoss(prior_weight=1.0, class_prior=[0.25] * 4), logits
    )
    np.testing.assert_allclose(implicit, explicit, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------
# 12. class_prior: the per-class WEIGHTING and the NORMALIZATION
# ---------------------------------------------------------------------


def _marginal_log_oracle(logits):
    """``log`` of the batch-marginal prediction, computed in float64 from the
    definition ``mean_i softmax(z_i)`` -- not from the implementation's
    logsumexp form."""
    z = np.asarray(logits, dtype="float64")
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.log(p.mean(axis=0))


def _kl_oracle(prior, logits):
    """``KL(q || mean_p) = sum_k q_k (log q_k - log mp_k)`` in float64, with
    ``q`` normalized here."""
    q = np.asarray(prior, dtype="float64")
    q = q / q.sum()
    return float(np.sum(q * (np.log(q) - _marginal_log_oracle(logits))))


#: A batch whose marginal is skewed but NOT collapsed, so every class keeps a
#: non-negligible mass and a skewed prior is a meaningful alternative.
_SKEWED_LOGITS = np.tile(
    np.array([[2.0, 1.0, 0.0, 0.0]], dtype="float32"), (12, 1)
)


@pytest.mark.parametrize(
    "prior", [[0.25] * 4, [0.1, 0.2, 0.3, 0.4], [0.85, 0.05, 0.05, 0.05]]
)
def test_the_prior_term_agrees_with_a_float64_weighted_kl_oracle(prior):
    """The per-class WEIGHTING guard: ``sum_k q_k log(q_k / mp_k)``, not the
    unweighted ``mean_k log(q_k / mp_k)``. The two agree for a uniform ``q``
    and disagree everywhere else, which is why a uniform-prior arm alone cannot
    see the weights being dropped."""
    loss = GoodhartAwareLoss(prior_weight=1.0, class_prior=prior)
    measured = _prior_term(loss, _SKEWED_LOGITS)
    np.testing.assert_allclose(
        measured, _kl_oracle(prior, _SKEWED_LOGITS), atol=1e-5, rtol=0
    )


def test_a_skewed_prior_gives_a_different_penalty_than_the_uniform_default():
    """ANTI-VACUITY for the oracle arm: the prior must actually be read. If it
    were ignored, every row of the parametrization above would grade the same
    number."""
    uniform = _prior_term(GoodhartAwareLoss(prior_weight=1.0), _SKEWED_LOGITS)
    skewed = _prior_term(
        GoodhartAwareLoss(prior_weight=1.0, class_prior=[0.1, 0.2, 0.3, 0.4]),
        _SKEWED_LOGITS,
    )
    assert abs(skewed - uniform) > 1e-3, (
        f"a skewed class_prior scored the same as the uniform default "
        f"({skewed} vs {uniform}); the prior is not reaching the computation"
    )


@pytest.mark.parametrize(
    "prior", [[0.25] * 4, [0.1, 0.2, 0.3, 0.4], [0.85, 0.05, 0.05, 0.05]]
)
def test_the_prior_term_is_non_negative_for_every_prior(prior):
    """A KL is non-negative by Gibbs' inequality. The unweighted
    ``mean_k log(q_k / mp_k)`` is NOT: on this batch it reads -0.543 for the
    prior skewed toward the dominant class, which is not a divergence."""
    measured = _prior_term(
        GoodhartAwareLoss(prior_weight=1.0, class_prior=prior), _SKEWED_LOGITS
    )
    assert measured >= -1e-6, (
        f"the anti-collapse term is negative ({measured}) for prior {prior}; "
        "no divergence can be, so the term is not the KL it claims to be"
    )


def test_the_penalty_is_minimized_by_the_prior_that_matches_the_batch_marginal():
    """The prior-MATCHING property: among candidate priors, the batch's own
    marginal is the unique minimizer, and it drives the term to zero."""
    matching = np.exp(_marginal_log_oracle(_SKEWED_LOGITS))
    at_match = _prior_term(
        GoodhartAwareLoss(prior_weight=1.0, class_prior=list(matching)),
        _SKEWED_LOGITS,
    )
    np.testing.assert_allclose(at_match, 0.0, atol=1e-5, rtol=0)

    for other in ([0.25] * 4, [0.1, 0.2, 0.3, 0.4], [0.85, 0.05, 0.05, 0.05]):
        assert _prior_term(
            GoodhartAwareLoss(prior_weight=1.0, class_prior=other), _SKEWED_LOGITS
        ) > at_match, f"prior {other} did not score above the matching prior"


@pytest.mark.parametrize(
    "raw, normalized",
    [
        ([2.0, 2.0, 2.0, 2.0], [0.25] * 4),
        ([1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]),
        ([7.0, 21.0, 14.0, 28.0], [0.1, 0.3, 0.2, 0.4]),
    ],
)
def test_an_unnormalized_class_prior_equals_its_normalized_twin(raw, normalized):
    """The NORMALIZATION guard. The docstring promises ``the sequence is
    normalized to sum to one if it does not already``; without the division the
    term is ``sum_k c*q_k (log(c*q_k) - log mp_k)``, which is neither the same
    number nor a divergence."""
    raw_term = _prior_term(
        GoodhartAwareLoss(prior_weight=1.0, class_prior=raw), _SKEWED_LOGITS
    )
    normalized_term = _prior_term(
        GoodhartAwareLoss(prior_weight=1.0, class_prior=normalized), _SKEWED_LOGITS
    )
    np.testing.assert_allclose(raw_term, normalized_term, atol=1e-5, rtol=0)


def test_the_unnormalized_prior_is_round_tripped_verbatim_not_pre_normalized():
    """ANTI-VACUITY twin: the equality above is produced by normalizing at USE
    time, not by the constructor having quietly rewritten the caller's list."""
    loss = GoodhartAwareLoss(prior_weight=1.0, class_prior=[1.0, 2.0, 3.0, 4.0])
    assert loss.get_config()["class_prior"] == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------
# 13. label_smoothing and epsilon actually reach the computation
# ---------------------------------------------------------------------


def _smoothed_ce_oracle(y_true, logits, label_smoothing):
    """Mean cross-entropy against the smoothed target, float64, from the
    definition: ``y' = (1 - eps) y + eps / K`` then ``-sum y' log softmax(z)``.
    Written from Szegedy et al.'s formula, not read off the implementation."""
    y = np.asarray(y_true, dtype="float64")
    z = np.asarray(logits, dtype="float64")
    num_classes = y.shape[-1]
    smoothed = y * (1.0 - label_smoothing) + label_smoothing / num_classes
    z = z - z.max(axis=-1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    return float(np.mean(-np.sum(smoothed * log_probs, axis=-1)))


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1, 0.3])
def test_label_smoothing_reaches_the_cross_entropy_and_matches_the_oracle(
    label_smoothing,
):
    """``label_smoothing`` was a live constructor knob with no behavioural
    coverage: disconnecting it from ``call()`` left the whole suite green.
    ``entropy_weight=0`` isolates the cross-entropy so the oracle is exact."""
    loss = GoodhartAwareLoss(
        label_smoothing=label_smoothing, entropy_weight=0.0
    )
    np.testing.assert_allclose(
        _f(loss(Y_TRUE_4, LOGITS_4)),
        _smoothed_ce_oracle(Y_TRUE_4, LOGITS_4, label_smoothing),
        atol=1e-5,
        rtol=0,
    )


def test_moving_label_smoothing_moves_the_loss():
    """ANTI-VACUITY for the oracle arm: the three parametrized values must be
    three different numbers, or the oracle could be agreeing by accident."""
    values = [
        _f(GoodhartAwareLoss(label_smoothing=ls, entropy_weight=0.0)(
            Y_TRUE_4, LOGITS_4
        ))
        for ls in (0.0, 0.1, 0.3)
    ]
    assert len(set(values)) == 3, (
        f"label_smoothing does not change the loss: {values}"
    )
    # On a one-hot target the smoothed target moves mass onto classes the model
    # scores lower, so the cross-entropy can only rise.
    assert values[0] < values[1] < values[2], (
        f"label smoothing did not increase the cross-entropy: {values}"
    )


#: A probability row containing an exact zero: the only fixture on which the
#: ``from_logits=False`` clip floor is observable at all.
_ZERO_PROBS = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")


@pytest.mark.parametrize("epsilon", [1e-3, 1e-6])
def test_the_users_epsilon_is_the_clip_floor_on_the_probability_path(epsilon):
    """``epsilon`` was validated in four ``raises`` rows and read once as an
    ATTRIBUTE, but its effect was never graded: hard-coding the floor at 1e-7
    left the suite green. Above float32's smallest normal the effective floor is
    ``epsilon`` itself, so the oracle clips at exactly that."""
    loss = GoodhartAwareLoss(from_logits=False, epsilon=epsilon)
    measured = keras.ops.convert_to_numpy(loss._log_probabilities(_ZERO_PROBS))

    floor = max(epsilon, float(np.finfo("float32").tiny))
    expected = np.clip(_ZERO_PROBS.astype("float64"), floor, 1.0)
    expected = expected / expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(measured, np.log(expected), atol=1e-4, rtol=0)


def test_two_different_epsilons_give_two_different_log_probabilities():
    """ANTI-VACUITY twin: the fixture must actually reach the floor, or the two
    oracle rows above would be graded against the same clipped array."""
    coarse = keras.ops.convert_to_numpy(
        GoodhartAwareLoss(from_logits=False, epsilon=1e-3)._log_probabilities(
            _ZERO_PROBS
        )
    )
    fine = keras.ops.convert_to_numpy(
        GoodhartAwareLoss(from_logits=False, epsilon=1e-6)._log_probabilities(
            _ZERO_PROBS
        )
    )
    assert np.max(np.abs(coarse - fine)) > 1.0, (
        "epsilon=1e-3 and epsilon=1e-6 produced the same log-probabilities "
        f"({coarse} vs {fine}); the user's epsilon is being ignored"
    )


# ---------------------------------------------------------------------
# 14. the dtype-aware clip floor, on a dtype that actually reaches it
#
# MEASURED 2026-09-02: ``keras.losses.Loss.__init__(dtype=None)`` resolves to
# ``backend.floatx()``, NOT to the global policy -- ``GoodhartAwareLoss().dtype``
# reads ``'float32'`` under a ``float32``, ``mixed_float16`` AND ``float64``
# global policy. So the global-policy fixture alone cannot vary the arithmetic,
# and in a real ``mixed_float16`` ``fit()`` the loss receives ``y_pred`` already
# cast to ``self.dtype``. The reachable float16 regimes are an EXPLICIT
# ``dtype=`` on the loss and a direct call with float16 tensors; both are
# exercised below.
# ---------------------------------------------------------------------


def test_the_global_dtype_policy_does_not_change_the_losss_compute_dtype(
    dtype_policy,
):
    """Documents WHY the policy-parametrized arms need an explicit ``dtype=``:
    Keras resolves ``dtype=None`` through ``floatx()``, so the global policy is
    invisible here. This arm exists so a future reader does not mistake those
    arms for float16 coverage."""
    assert GoodhartAwareLoss().dtype == keras.backend.floatx()


@pytest.mark.parametrize("loss_dtype", ["float32", "mixed_float16", "float64"])
def test_an_explicit_dtype_argument_does_reach_the_computation(loss_dtype):
    """ANTI-VACUITY for the float16 arms: the explicit path must genuinely
    change the compute dtype, unlike the global policy."""
    expected = keras.mixed_precision.Policy(loss_dtype).compute_dtype
    assert GoodhartAwareLoss(dtype=loss_dtype).dtype == expected


@pytest.mark.parametrize("loss_dtype", ["float32", "mixed_float16", "float64"])
def test_the_probability_path_is_finite_at_every_explicit_loss_dtype(loss_dtype):
    """The ``from_logits=False`` path under a REAL float16 compute dtype, on a
    row containing an exact zero -- the case a dtype-blind
    ``clip(y_pred, 1e-8, 1.0)`` floors to 0.0, making ``log`` return ``-inf``."""
    loss = GoodhartAwareLoss(
        from_logits=False, prior_weight=0.5, dtype=loss_dtype
    )
    out = keras.ops.convert_to_numpy(
        loss(np.eye(4, dtype="float32")[:1], _ZERO_PROBS)
    )
    assert np.all(np.isfinite(out)), (
        f"non-finite loss at dtype={loss_dtype} on a row containing an exact "
        "zero probability"
    )


def test_a_genuinely_float16_probability_row_gives_finite_log_probabilities():
    """The direct-call float16 regime, which is the one the D-005 anchor names.
    ``epsilon = 1e-8`` is below float16's smallest normal, so a dtype-blind clip
    returns ``[0, -inf, -inf, -inf]`` here."""
    probs16 = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float16)
    log_probs = keras.ops.convert_to_numpy(
        GoodhartAwareLoss(from_logits=False)._log_probabilities(probs16)
    )
    assert np.all(np.isfinite(log_probs)), (
        f"float16 probabilities produced non-finite log-probabilities: "
        f"{log_probs}"
    )


def test_the_float16_clip_floor_is_the_dtype_tiny_not_the_configured_epsilon():
    """ANTI-VACUITY twin, and the honest statement of what D-005 trades away:
    under float16 the effective floor is ``finfo(float16).tiny`` (6.104e-05),
    NOT the configured ``epsilon`` (1e-8). The finiteness arm above must be
    passing because of THAT substitution and not for some other reason."""
    probs16 = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float16)
    log_probs = keras.ops.convert_to_numpy(
        GoodhartAwareLoss(from_logits=False, epsilon=1e-8)._log_probabilities(
            probs16
        )
    )
    tiny16 = float(np.finfo("float16").tiny)
    # The clipped entries renormalize by ~1, so their log is log(tiny16) to
    # within float16 resolution (~1e-3 relative at magnitude 9.7).
    np.testing.assert_allclose(
        log_probs[0, 1:], np.log(tiny16), atol=1e-2, rtol=0
    )
    assert np.log(tiny16) > np.log(1e-8), (
        "the float16 floor is not above the configured epsilon, so this arm "
        "is not observing the D-005 substitution"
    )
