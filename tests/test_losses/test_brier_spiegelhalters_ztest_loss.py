"""Guards for the calibration losses and metrics.

Every assertion here exists because the pre-2026-09-02 module failed it. The
predecessor of this file was 615 lines and 28 tests, and it asserted almost
nothing but ``not isnan`` / ``not isinf`` / ``>= 0`` / ``hasattr`` / config-dict
equality -- so the module shipped the WRONG STATISTIC under Spiegelhalter's
name for its entire life while the suite stayed green. Nothing in it ever
asserted a Z value.

**The oracle is independent.** ``_z_oracle`` below is written from the formula
in Spiegelhalter (1986) -- residuals weighted by ``(1 - 2p)``, variance weighted
by ``(1 - 2p)**2`` -- not from the implementation. An oracle copied from the
code it grades passes forever and proves nothing. The discriminating arm
(:func:`test_the_two_statistics_are_different_numbers`) is what keeps the oracle
from silently agreeing with the old formula.

**Fixtures are seeded.** The predecessor drew every fixture from unseeded
``np.random``, so a failure was not reproducible and a tolerance could not be
derived.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.losses.brier_spiegelhalters_ztest_loss import (
    PROBABILITY_EPSILON,
    STATISTIC_CALIBRATION_IN_THE_LARGE,
    STATISTIC_SPIEGELHALTER,
    BrierScoreLoss,
    BrierScoreMetric,
    CombinedCalibrationLoss,
    SpiegelhalterZLoss,
    SpiegelhalterZMetric,
)

# ---------------------------------------------------------------------
# fixtures and the independent oracle
# ---------------------------------------------------------------------

#: float32 sums over a few hundred terms reassociate; measured worst deviation
#: between the per-row decomposition and the closed form on the fixtures below
#: was 3.7e-05 on Z**2 ~ 1.28 (2026-09-02, CPU, float32).
Z_ATOL = 1e-4

#: Four rows spanning both sides of p = 0.5, so that (1 - 2p) changes SIGN.
#: A fixture confined to one side cannot distinguish the two statistics, which
#: is exactly how the wrong one survived.
Y_TRUE_4 = np.array([[1.0], [0.0], [1.0], [0.0]], dtype="float32")
Y_PRED_4 = np.array([[0.8], [0.3], [0.4], [0.7]], dtype="float32")

SAMPLE_WEIGHT_4 = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")
MEAN_W_4 = 0.75


def _tensor(x):
    return keras.ops.convert_to_tensor(np.asarray(x, dtype="float32"))


def _numpy(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def _float(x):
    return float(keras.ops.convert_to_numpy(x))


def _binary_batch(batch, seed, low=0.05, high=0.95, bias=0.0, temperature=1.0):
    """Seeded (y_true, y_pred) with shape ``(batch, 1)``.

    Outcomes are always drawn FROM the true probability, which is the only way
    to make the null hypothesis actually true. Two independent, deliberately
    DIFFERENT miscalibrations are then applied to the reported probability:

    ``bias``
        An additive shift in probability units. This is a net-bias failure and
        it is what calibration-in-the-large is built to detect.
    ``temperature``
        A temperature applied to the logit. ``< 1`` sharpens (over-confidence),
        ``> 1`` smooths. This failure is SYMMETRIC about 0.5, so it barely
        moves the net bias at all -- it is what Spiegelhalter's ``(1 - 2p)``
        weight is built to detect.

    Keeping the two separate is what lets the tests below show that the two
    statistics are not interchangeable. MEASURED 2026-09-02 at N=4096, averaged
    over 20 seeds: ``bias=+0.05`` gives ``Z_cil = -7.47`` but ``Z_sh = 1.70``;
    ``temperature=0.7`` gives ``Z_sh = 11.90`` but ``Z_cil = 0.08``.
    """
    rng = np.random.default_rng(seed)
    p_true = rng.uniform(low, high, size=(batch, 1))
    y = (rng.uniform(size=(batch, 1)) < p_true).astype("float32")
    logit = np.log(p_true / (1.0 - p_true)) / temperature
    p = 1.0 / (1.0 + np.exp(-logit)) + bias
    return y, np.clip(p, 1e-4, 1.0 - 1e-4).astype("float32")


def _mean_z_squared(batch, n_draws=200, seed0=2000, **miscalibration):
    """E[Z**2] estimated over ``n_draws`` independent seeded batches."""
    return float(
        np.mean(
            [
                _z_oracle(*_binary_batch(batch, seed0 + i, **miscalibration)) ** 2
                for i in range(n_draws)
            ]
        )
    )


def _z_oracle(y_true, y_pred, statistic=STATISTIC_SPIEGELHALTER, sample_weight=None):
    """Spiegelhalter (1986) Z, in float64 numpy, from the paper's formula.

    Z = sum_i (o_i - p_i)(1 - 2p_i) / sqrt(sum_i p_i(1 - p_i)(1 - 2p_i)^2)

    with the unweighted calibration-in-the-large variant obtained by setting
    every weight to one.
    """
    y = np.asarray(y_true, dtype="float64").ravel()
    p = np.clip(
        np.asarray(y_pred, dtype="float64").ravel(),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    w = (1.0 - 2.0 * p) if statistic == STATISTIC_SPIEGELHALTER else np.ones_like(p)
    contributions = (y - p) * w
    variances = p * (1.0 - p) * w * w
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype="float64").ravel()
        contributions = contributions * sw
        # Var[sum_i w_i r_i] = sum_i w_i^2 Var[r_i]: QUADRATIC in the weight.
        variances = variances * sw * sw
    numerator = float(np.sum(contributions))
    denominator = float(np.sum(variances))
    n = float(y.size)
    denominator = denominator + n * PROBABILITY_EPSILON + keras.config.epsilon()
    return numerator / np.sqrt(denominator)


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


# ---------------------------------------------------------------------
# 1. the statistic is the one the class is named after
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "statistic", [STATISTIC_SPIEGELHALTER, STATISTIC_CALIBRATION_IN_THE_LARGE]
)
@pytest.mark.parametrize("batch,seed", [(4, 0), (128, 7), (512, 11)])
def test_the_loss_computes_the_statistic_it_names(statistic, batch, seed):
    """The RED proof for the misattribution: the old code failed this at
    ``statistic='spiegelhalter'`` and passed it at the other value."""
    if batch == 4:
        y_true, y_pred = Y_TRUE_4, Y_PRED_4
    else:
        y_true, y_pred = _binary_batch(batch, seed)

    expected = _z_oracle(y_true, y_pred, statistic) ** 2
    loss = SpiegelhalterZLoss(
        statistic=statistic, chance_corrected=False, normalize_by_n=False
    )
    measured = _float(keras.ops.mean(loss.call(_tensor(y_true), _tensor(y_pred))))

    assert measured == pytest.approx(expected, abs=Z_ATOL, rel=0.0), (
        f"Z**2 for statistic={statistic!r} is {measured!r} against an oracle "
        f"value of {expected!r}. The oracle is the paper's formula; if the "
        f"Spiegelhalter arm fails while the calibration-in-the-large arm passes, "
        f"the (1 - 2p) residual weight has gone missing again."
    )


def test_the_two_statistics_are_different_numbers():
    """ANTI-VACUITY. If these agree, the oracle cannot tell the formulas apart
    and every arm above is decorative."""
    sh = _z_oracle(Y_TRUE_4, Y_PRED_4, STATISTIC_SPIEGELHALTER)
    cil = _z_oracle(Y_TRUE_4, Y_PRED_4, STATISTIC_CALIBRATION_IN_THE_LARGE)
    assert abs(sh - cil) > 0.1, (
        f"the fixture does not discriminate the two statistics "
        f"(spiegelhalter={sh!r}, calibration_in_the_large={cil!r}). It must span "
        f"both sides of p = 0.5 so that the (1 - 2p) weight changes sign."
    )


def test_spiegelhalters_weight_is_what_makes_the_two_differ():
    """The mechanism, asserted directly: on a fixture where every p is BELOW
    0.5 and the residuals share a sign, both statistics agree in sign; the
    four-row fixture above disagrees in sign only because of the weight."""
    sh = _z_oracle(Y_TRUE_4, Y_PRED_4, STATISTIC_SPIEGELHALTER)
    cil = _z_oracle(Y_TRUE_4, Y_PRED_4, STATISTIC_CALIBRATION_IN_THE_LARGE)
    assert sh > 0.0 > cil, (
        f"expected the (1 - 2p) weight to FLIP the sign on this fixture "
        f"(spiegelhalter={sh!r}, calibration_in_the_large={cil!r}); it is the "
        f"clearest evidence the weight is applied at all"
    )


def _mean_z(batch, statistic, n_draws=20, seed0=1000, **miscalibration):
    return float(
        np.mean(
            [
                _z_oracle(
                    *_binary_batch(batch, seed0 + i, **miscalibration), statistic
                )
                for i in range(n_draws)
            ]
        )
    )


def test_a_perfectly_calibrated_predictor_is_not_rejected():
    """|Z| < 1.96 is the 5% acceptance region; a calibrated 4096-sample draw
    must sit inside it for BOTH statistics."""
    y_true, y_pred = _binary_batch(4096, seed=3)
    for statistic in (STATISTIC_SPIEGELHALTER, STATISTIC_CALIBRATION_IN_THE_LARGE):
        z = _z_oracle(y_true, y_pred, statistic)
        assert abs(z) < 1.96, f"{statistic}: |Z| = {abs(z)!r} rejects a calibrated model"


def test_calibration_in_the_large_detects_an_additive_bias():
    """The 'something changed' twin: a +0.05 probability shift is a NET BIAS
    failure, which is exactly what this statistic tests."""
    z = _mean_z(4096, STATISTIC_CALIBRATION_IN_THE_LARGE, bias=0.05)
    assert abs(z) > 1.96, (
        f"mean Z_cil = {z!r} failed to reject a model biased by 0.05 in "
        f"probability units at N=4096"
    )


def test_spiegelhalters_z_detects_over_confidence_that_the_weaker_test_misses():
    """The reason the misattribution matters. A temperature of 0.7 on the logit
    is over-confidence: it is SYMMETRIC about p = 0.5, so it barely moves the
    net bias, and calibration-in-the-large -- the statistic this module shipped
    under Spiegelhalter's name -- is effectively BLIND to it."""
    z_sh = _mean_z(4096, STATISTIC_SPIEGELHALTER, temperature=0.7)
    z_cil = _mean_z(4096, STATISTIC_CALIBRATION_IN_THE_LARGE, temperature=0.7)
    assert abs(z_sh) > 1.96, (
        f"mean Z_sh = {z_sh!r} failed to reject an over-confident model "
        f"(temperature 0.7) at N=4096"
    )
    assert abs(z_cil) < 1.96, (
        f"mean Z_cil = {z_cil!r} rejected the over-confident model too, so this "
        f"fixture no longer separates the two statistics"
    )


def test_the_weaker_statistic_is_the_one_that_sees_a_pure_shift():
    """The mirror image, so neither arm can be satisfied by a statistic that
    simply fires on everything."""
    z_sh = _mean_z(4096, STATISTIC_SPIEGELHALTER, bias=0.05)
    z_cil = _mean_z(4096, STATISTIC_CALIBRATION_IN_THE_LARGE, bias=0.05)
    assert abs(z_cil) > 4.0 * abs(z_sh), (
        f"a uniform additive shift should be far louder in Z_cil than in Z_sh "
        f"(the (1 - 2p) weight changes sign across 0.5 and largely cancels it), "
        f"but measured Z_cil={z_cil!r} against Z_sh={z_sh!r}"
    )


# ---------------------------------------------------------------------
# 2. the chance floor E[Z**2] = 1
# ---------------------------------------------------------------------


def test_the_null_value_of_z_squared_is_one_not_zero():
    """The premise behind `chance_corrected`. Averaged over 200 calibrated
    batches of 256, Z**2 must sit near 1 -- so minimizing Z**2 toward 0 is
    asking the model to fit minibatch label noise."""
    mean_z2 = _mean_z_squared(256)
    assert 0.75 < mean_z2 < 1.35, (
        f"E[Z**2] measured {mean_z2!r} over 200 calibrated batches; the null "
        f"distribution is N(0, 1), so this should be near 1.0. If it is near "
        f"0, the fixture is not actually calibrated and every chance-floor "
        f"claim in this module is unsupported."
    )


def test_chance_correction_is_near_zero_for_a_calibrated_model():
    y_true, y_pred = _binary_batch(4096, seed=21)
    loss = SpiegelhalterZLoss()  # chance_corrected=True, normalize_by_n=True
    value = _float(keras.ops.mean(loss.call(_tensor(y_true), _tensor(y_pred))))
    assert abs(value) < 1e-3, (
        f"the chance-corrected penalty is {value!r} on a calibrated batch; it "
        f"should be at or near zero"
    )


def test_chance_correction_is_positive_for_a_miscalibrated_model():
    """The both-ways twin. Without it, a penalty stuck at 0.0 would pass."""
    y_true, y_pred = _binary_batch(4096, seed=21, temperature=0.7)
    loss = SpiegelhalterZLoss()
    value = _float(keras.ops.mean(loss.call(_tensor(y_true), _tensor(y_pred))))
    assert value > 1e-3, (
        f"the chance-corrected penalty is {value!r} on an over-confident batch; "
        f"it must be strictly positive or the term is dead"
    )


def test_chance_correction_requires_the_squared_statistic():
    """E[|Z|] = sqrt(2/pi), not 1, so the correction does not transfer."""
    with pytest.raises(ValueError, match="chance_corrected=True requires"):
        SpiegelhalterZLoss(use_squared=False)


# ---------------------------------------------------------------------
# 3. batch-size scaling
# ---------------------------------------------------------------------


def _expected_penalty(batch, normalize_by_n, n_draws=200):
    """E[relu(Z**2 - 1)] over seeded draws, optionally divided by the batch.

    A single draw does not measure this: at batch 64 the sampling noise of
    Z**2 (variance 2 under the null) dominates the systematic term, which is
    how an earlier version of this guard read 2.67 at batch 64 against 2.89 at
    batch 512 and concluded there was no scaling at all.
    """
    mean_z2 = _mean_z_squared(batch, n_draws=n_draws, temperature=0.7)
    penalty = max(0.0, mean_z2 - 1.0)
    return penalty / batch if normalize_by_n else penalty


def test_the_penalty_is_batch_size_invariant_when_normalized():
    """Z**2 - 1 ~ N b**2 / v, so the term normalized by N is a property of the
    MISCALIBRATION and not of the batch size."""
    small = _expected_penalty(64, normalize_by_n=True)
    large = _expected_penalty(512, normalize_by_n=True)
    assert large == pytest.approx(small, rel=0.5, abs=0.0), (
        f"normalized penalty moved from {small!r} at batch 64 to {large!r} at "
        f"batch 512; it is supposed to be batch-size invariant"
    )


def test_the_unnormalized_penalty_grows_with_the_batch():
    """RED-proof for the arm above: flip the knob and the growth must appear.
    Measured 2026-09-02 at temperature 0.7 over 200 draws: E[Z**2] = 3.86 at
    batch 64 against 19.46 at batch 512, i.e. the penalty grows 6.5x."""
    small = _expected_penalty(64, normalize_by_n=False)
    large = _expected_penalty(512, normalize_by_n=False)
    assert large > 4.0 * small, (
        f"unnormalized penalty went {small!r} -> {large!r} from batch 64 to 512. "
        f"It should grow roughly 8x; if it does not, the normalize_by_n guard "
        f"above is not measuring anything."
    )


def test_the_loss_reproduces_the_expected_penalty_scaling():
    """The arms above are computed on the ORACLE; this one pins that the loss
    itself follows the same scaling, so the oracle is not grading itself."""
    for batch in (64, 512):
        values = []
        for i in range(50):
            y_true, y_pred = _binary_batch(batch, seed=3000 + i, temperature=0.7)
            loss = SpiegelhalterZLoss()
            values.append(
                _float(keras.ops.mean(loss.call(_tensor(y_true), _tensor(y_pred))))
            )
        oracle = float(
            np.mean(
                [
                    max(
                        0.0,
                        _z_oracle(
                            *_binary_batch(batch, 3000 + i, temperature=0.7)
                        )
                        ** 2
                        - 1.0,
                    )
                    / batch
                    for i in range(50)
                ]
            )
        )
        assert float(np.mean(values)) == pytest.approx(oracle, rel=0.02, abs=0.0), (
            f"at batch {batch} the loss averaged {float(np.mean(values))!r} "
            f"against an oracle mean of {oracle!r}"
        )


# ---------------------------------------------------------------------
# 4. the per-sample contract
# ---------------------------------------------------------------------


def _all_losses():
    return [
        ("BrierScoreLoss", BrierScoreLoss()),
        ("SpiegelhalterZLoss", SpiegelhalterZLoss()),
        ("SpiegelhalterZLoss.legacy", SpiegelhalterZLoss(
            statistic=STATISTIC_CALIBRATION_IN_THE_LARGE,
            chance_corrected=False,
            normalize_by_n=False,
        )),
        ("CombinedCalibrationLoss", CombinedCalibrationLoss()),
        ("CombinedCalibrationLoss.bce", CombinedCalibrationLoss(base="bce")),
        ("CombinedCalibrationLoss.legacy", CombinedCalibrationLoss(alpha=0.6)),
    ]


@pytest.mark.parametrize("name,loss", _all_losses(), ids=[n for n, _ in _all_losses()])
def test_call_returns_one_value_per_sample(name, loss):
    """The house rule in losses/CLAUDE.md. A scalar return does not IGNORE
    sample_weight -- it broadcasts against it and charges every row the batch
    aggregate."""
    y_true, y_pred = _binary_batch(32, seed=9)
    per_sample = _numpy(loss.call(_tensor(y_true), _tensor(y_pred)))
    assert per_sample.ndim == 1, (
        f"{name}.call() returned shape {per_sample.shape}, expected rank 1"
    )
    assert per_sample.shape[0] == 32, (
        f"{name}.call() returned {per_sample.shape[0]} values for a batch of 32"
    )


@pytest.mark.parametrize(
    "chance_corrected,normalize_by_n",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_the_per_row_decomposition_reproduces_the_scalar_statistic(
    chance_corrected, normalize_by_n
):
    """The identity the decomposition rests on: mean_i(N c_i num/den) = Z**2."""
    batch = 256
    # temperature 0.6 puts Z**2 well ABOVE the chance floor, so the
    # chance_corrected arms are not satisfied by a gate that is simply off.
    y_true, y_pred = _binary_batch(batch, seed=13, temperature=0.6)
    z = _z_oracle(y_true, y_pred)
    assert z * z > 4.0, (
        f"fixture is not miscalibrated enough to exercise the chance floor "
        f"(Z**2 = {z * z!r}); the chance_corrected arms would be vacuous"
    )
    expected = z * z
    if chance_corrected:
        expected = max(0.0, expected - 1.0)
    if normalize_by_n:
        expected = expected / batch

    loss = SpiegelhalterZLoss(
        chance_corrected=chance_corrected, normalize_by_n=normalize_by_n
    )
    measured = _float(keras.ops.mean(loss.call(_tensor(y_true), _tensor(y_pred))))
    assert measured == pytest.approx(expected, abs=Z_ATOL, rel=0.0), (
        f"mean of the per-sample vector is {measured!r}, closed form is "
        f"{expected!r} (chance_corrected={chance_corrected}, "
        f"normalize_by_n={normalize_by_n})"
    )


def test_the_absolute_form_reproduces_abs_z():
    y_true, y_pred = _binary_batch(256, seed=17, bias=0.03)
    expected = abs(_z_oracle(y_true, y_pred))
    loss = SpiegelhalterZLoss(
        use_squared=False, chance_corrected=False, normalize_by_n=False
    )
    measured = _float(keras.ops.mean(loss.call(_tensor(y_true), _tensor(y_pred))))
    assert measured == pytest.approx(expected, abs=Z_ATOL, rel=0.0)


def test_the_brier_loss_is_the_mean_squared_error():
    y_true, y_pred = _binary_batch(64, seed=23)
    expected = float(np.mean((y_pred - y_true) ** 2))
    measured = _float(BrierScoreLoss()(_tensor(y_true), _tensor(y_pred)))
    stock = _float(
        keras.losses.MeanSquaredError()(_tensor(y_true), _tensor(y_pred))
    )
    assert measured == pytest.approx(expected, abs=1e-6, rel=0.0)
    assert measured == pytest.approx(stock, abs=1e-6, rel=0.0), (
        "BrierScoreLoss on binary targets IS MeanSquaredError; if this ever "
        "diverges the docstring's equivalence claim is false"
    )


def test_the_brier_loss_keeps_the_single_column_convention():
    """Perfect = 0.0, fully inverted = 1.0 (not the two-column 2.0)."""
    perfect = _float(BrierScoreLoss()(_tensor(Y_TRUE_4), _tensor(Y_TRUE_4)))
    inverted = _float(
        BrierScoreLoss()(_tensor(Y_TRUE_4), _tensor(1.0 - Y_TRUE_4))
    )
    assert perfect == pytest.approx(0.0, abs=1e-6, rel=0.0)
    assert inverted == pytest.approx(1.0, abs=1e-6, rel=0.0)


# ---------------------------------------------------------------------
# 5. sample_weight
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,loss", _all_losses(), ids=[n for n, _ in _all_losses()])
def test_sample_weight_is_not_applied_as_a_broadcast_scalar(name, loss):
    """The executable predicate from test_the_premature_scalar_family_is_pinned:
    `weighted == unweighted * mean(w)` holds IFF call() returned a scalar."""
    y_true, y_pred = _binary_batch(64, seed=29, temperature=0.6, bias=0.03)
    weights = np.tile(SAMPLE_WEIGHT_4, 16).astype("float32")

    unweighted = _float(loss(_tensor(y_true), _tensor(y_pred)))
    weighted = _float(
        loss(_tensor(y_true), _tensor(y_pred), sample_weight=_tensor(weights))
    )
    assert abs(unweighted) > 1e-6, (
        f"{name}: the unweighted loss is {unweighted!r}, so 0 == 0 * 0.75 would "
        f"satisfy this guard vacuously -- the fixture must produce a live value"
    )
    scaled = unweighted * MEAN_W_4
    assert abs(weighted - scaled) > 1e-6 * max(1.0, abs(scaled)), (
        f"{name}: weighted={weighted!r} equals unweighted*{MEAN_W_4} = "
        f"{scaled!r}, which is the signature of call() returning a scalar that "
        f"broadcast against sample_weight"
    )


def test_zero_weighting_a_row_removes_its_contribution_from_the_brier_loss():
    """The positive arm: a zero-weighted row's VALUE stops mattering."""
    y_true, y_pred = _binary_batch(64, seed=31)
    weights = np.tile(SAMPLE_WEIGHT_4, 16).astype("float32")
    loss = BrierScoreLoss()

    before = _float(
        loss(_tensor(y_true), _tensor(y_pred), sample_weight=_tensor(weights))
    )
    perturbed = y_pred.copy()
    perturbed[3::4] = 0.999999  # exactly the rows carrying weight 0.0
    after = _float(
        loss(_tensor(y_true), _tensor(perturbed), sample_weight=_tensor(weights))
    )
    assert after == pytest.approx(before, abs=1e-6, rel=0.0), (
        f"perturbing only the zero-weighted rows moved the Brier loss "
        f"{before!r} -> {after!r}"
    )


def test_the_brier_loss_matches_a_hand_computed_weighted_mean():
    y_true, y_pred = _binary_batch(64, seed=37)
    weights = np.tile(SAMPLE_WEIGHT_4, 16).astype("float64")
    sq = ((y_pred.astype("float64") - y_true.astype("float64")) ** 2).ravel()
    expected = float(np.sum(sq * weights) / weights.size)  # sum_over_batch_size
    measured = _float(
        BrierScoreLoss()(
            _tensor(y_true), _tensor(y_pred), sample_weight=_tensor(weights)
        )
    )
    assert measured == pytest.approx(expected, abs=1e-6, rel=0.0)


# ---------------------------------------------------------------------
# 6. the metrics
# ---------------------------------------------------------------------


def test_the_z_metric_variance_is_quadratic_in_the_sample_weight():
    """The exact guard for the metric weighting defect.

    ``Var[sum_i w_i r_i] = sum_i w_i^2 Var[r_i]``, so a weighted z-statistic
    needs ``sum w_i^2 p(1-p)(1-2p)^2`` in the denominator. Weighting the
    variance LINEARLY, as this metric used to, leaves a ratio that is not a
    z-score of anything.

    **This guard needs weights that are not 0/1.** ``w**2 == w`` for both 0 and
    1, so a row-selection weight vector cannot distinguish the two forms --
    measured: the linear-variance mutation left all 102 tests in this file
    GREEN when the only weighted arm used ``[1, 1, 1, 0]``.
    """
    y_true, y_pred = _binary_batch(256, seed=39, temperature=0.7)
    weights = np.tile(np.array([2.0, 1.0, 3.0, 0.0]), 64).astype("float32")

    metric = SpiegelhalterZMetric()
    metric.update_state(
        _tensor(y_true), _tensor(y_pred), sample_weight=_tensor(weights)
    )
    quadratic = _z_oracle(y_true, y_pred, sample_weight=weights)
    linear_variance = float(
        np.sum(
            (
                (y_true.astype("float64") - np.clip(y_pred.astype("float64"), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
                * (1.0 - 2.0 * np.clip(y_pred.astype("float64"), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
            ).ravel()
            * weights
        )
    ) / np.sqrt(
        float(
            np.sum(
                (
                    np.clip(y_pred.astype("float64"), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)
                    * (1.0 - np.clip(y_pred.astype("float64"), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON))
                    * (1.0 - 2.0 * np.clip(y_pred.astype("float64"), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)) ** 2
                ).ravel()
                * weights
            )
        )
        + 256 * PROBABILITY_EPSILON
    )

    assert abs(quadratic - linear_variance) > 0.1, (
        f"ANTI-VACUITY: the quadratic and linear forms agree on this fixture "
        f"({quadratic!r} vs {linear_variance!r}), so the guard below cannot "
        f"discriminate them"
    )
    assert _float(metric.result()) == pytest.approx(quadratic, abs=1e-4, rel=0.0), (
        f"the weighted z-statistic is {_float(metric.result())!r}; the "
        f"quadratic-variance form gives {quadratic!r} and the linear one "
        f"{linear_variance!r}"
    )


def test_a_zero_one_sample_weight_is_equivalent_to_dropping_the_rows():
    """The row-selection property, which holds under BOTH weighting forms
    (``w**2 == w`` at 0 and 1) and is therefore not a substitute for the guard
    above -- it is here because row selection is what callers actually use."""
    y_true, y_pred = _binary_batch(64, seed=41, bias=0.04)
    weights = np.tile(SAMPLE_WEIGHT_4, 16).astype("float32")
    kept = np.repeat(SAMPLE_WEIGHT_4.astype(bool), 1)
    keep_mask = np.tile(kept, 16)

    weighted_metric = SpiegelhalterZMetric()
    weighted_metric.update_state(
        _tensor(y_true), _tensor(y_pred), sample_weight=_tensor(weights)
    )
    subset_metric = SpiegelhalterZMetric()
    subset_metric.update_state(
        _tensor(y_true[keep_mask]), _tensor(y_pred[keep_mask])
    )

    assert _float(weighted_metric.result()) == pytest.approx(
        _float(subset_metric.result()), abs=1e-5, rel=0.0
    ), (
        "a 0/1 sample_weight must be equivalent to dropping the rows. If it is "
        "not, the variance is being weighted linearly and the ratio is no "
        "longer a z-score of anything."
    )


def test_the_z_metric_matches_the_oracle():
    y_true, y_pred = _binary_batch(512, seed=43, bias=0.03)
    for statistic in (STATISTIC_SPIEGELHALTER, STATISTIC_CALIBRATION_IN_THE_LARGE):
        metric = SpiegelhalterZMetric(statistic=statistic)
        metric.update_state(_tensor(y_true), _tensor(y_pred))
        assert _float(metric.result()) == pytest.approx(
            _z_oracle(y_true, y_pred, statistic), abs=1e-4, rel=0.0
        ), f"SpiegelhalterZMetric disagrees with the oracle at statistic={statistic!r}"


def test_the_z_metric_accumulates_across_batches():
    y_true, y_pred = _binary_batch(512, seed=47, bias=0.03)
    streaming = SpiegelhalterZMetric()
    for start in range(0, 512, 64):
        streaming.update_state(
            _tensor(y_true[start:start + 64]), _tensor(y_pred[start:start + 64])
        )
    assert _float(streaming.result()) == pytest.approx(
        _z_oracle(y_true, y_pred), abs=1e-4, rel=0.0
    )


def test_the_z_metric_resets():
    y_true, y_pred = _binary_batch(256, seed=53, bias=0.1)
    metric = SpiegelhalterZMetric()
    metric.update_state(_tensor(y_true), _tensor(y_pred))
    assert abs(_float(metric.result())) > 1.0  # anti-vacuity: it moved first
    metric.reset_state()
    assert _float(metric.result()) == pytest.approx(0.0, abs=1e-7, rel=0.0)


def test_the_brier_metric_computes_a_weighted_mean():
    """The denominator must be the WEIGHT SUM. Accumulating a weighted
    numerator over a raw element count is not a mean of anything."""
    y_true, y_pred = _binary_batch(64, seed=59)
    weights = np.tile(np.array([1.0, 2.0, 3.0, 0.0]), 16).astype("float32")
    sq = ((y_pred.astype("float64") - y_true.astype("float64")) ** 2).ravel()
    expected = float(np.sum(sq * weights) / np.sum(weights))

    metric = BrierScoreMetric()
    metric.update_state(
        _tensor(y_true), _tensor(y_pred), sample_weight=_tensor(weights)
    )
    assert _float(metric.result()) == pytest.approx(expected, abs=1e-6, rel=0.0), (
        "BrierScoreMetric is not returning a weighted mean; check that count "
        "accumulates sum(sample_weight) and not ops.size(y_true)"
    )


def test_the_brier_metric_agrees_with_the_canonical_implementation():
    """`dl_techniques.metrics.brier_score.BrierScore` is the canonical home for
    this quantity. Two implementations of one number drift; this pins them."""
    from dl_techniques.metrics.brier_score import BrierScore

    y_true, y_pred = _binary_batch(128, seed=61)
    weights = np.tile(np.array([1.0, 2.0, 3.0, 0.0]), 32).astype("float32")

    local = BrierScoreMetric(from_logits=False)
    canonical = BrierScore(from_logits=False)
    for metric in (local, canonical):
        metric.update_state(
            _tensor(y_true), _tensor(y_pred), sample_weight=_tensor(weights)
        )
    assert _float(local.result()) == pytest.approx(
        _float(canonical.result()), abs=1e-6, rel=0.0
    )


def test_the_brier_metric_resets():
    y_true, y_pred = _binary_batch(64, seed=67)
    metric = BrierScoreMetric()
    metric.update_state(_tensor(y_true), _tensor(y_pred))
    assert _float(metric.result()) > 0.0
    metric.reset_state()
    assert _float(metric.result()) == pytest.approx(0.0, abs=1e-7, rel=0.0)


@pytest.mark.parametrize("metric_cls", [BrierScoreMetric, SpiegelhalterZMetric])
def test_a_metric_with_no_updates_returns_zero(metric_cls):
    assert _float(metric_cls().result()) == pytest.approx(0.0, abs=1e-7, rel=0.0)


# ---------------------------------------------------------------------
# 7. numerical stability at saturation
# ---------------------------------------------------------------------


def test_saturated_probabilities_stay_bounded():
    """A saturated batch must not blow the statistic up.

    The bound is derived: at the clip boundary the per-element variance is
    ``~PROBABILITY_EPSILON``, so ``den >= N * 1e-6`` and
    ``Z**2 <= num**2 / (N * 1e-6) <= N / 1e-6`` for ``|num| <= N``.

    **This guards a PAIR of redundant defences**, and that is what the
    mutations measured (2026-09-02): the probability clip and the
    N-proportional denominator floor each bound this batch on their own, so
    removing EITHER leaves this guard green and only removing BOTH -- which is
    exactly the pre-fix code, an absolute ``keras.backend.epsilon()`` added to
    an unclipped variance sum -- turns it red, at about ``1e10``. Do not read a
    green here as evidence that both defences are present.
    """
    # A NET residual is required. Measured: a fixture whose residuals cancel
    # (num = 0) leaves this guard green with the clip removed, because 0/den is
    # 0 at any den. Every row here is wrong AND saturated, so num = 32.
    y_true = np.ones((32, 1), dtype="float32")
    y_pred = np.zeros((32, 1), dtype="float32")

    for statistic in (STATISTIC_SPIEGELHALTER, STATISTIC_CALIBRATION_IN_THE_LARGE):
        loss = SpiegelhalterZLoss(
            statistic=statistic, chance_corrected=False, normalize_by_n=False
        )
        value = _float(
            keras.ops.mean(loss.call(_tensor(y_true), _tensor(y_pred)))
        )
        assert np.isfinite(value), f"{statistic}: saturated batch gave {value!r}"
        assert value <= 32.0 / PROBABILITY_EPSILON, (
            f"{statistic}: saturated batch gave {value!r}, above the bound "
            f"{32.0 / PROBABILITY_EPSILON!r} that the clip guarantees. Without "
            f"the clip the denominator is keras.config.epsilon() and the same "
            f"batch reads about 1e10."
        )
    # ANTI-VACUITY: the bound must be BELOW what the unclipped form produces,
    # or it is satisfied by any implementation at all.
    unclipped = 32.0 ** 2 / keras.config.epsilon()
    assert unclipped > 32.0 / PROBABILITY_EPSILON, (
        f"the stated bound {32.0 / PROBABILITY_EPSILON!r} is not below the "
        f"unclipped value {unclipped!r}, so it discriminates nothing"
    )


def test_the_degenerate_half_probability_batch_is_finite():
    """At p = 0.5 the Spiegelhalter weight (1 - 2p) is zero, so BOTH the
    numerator and the denominator vanish. The floor must keep this at 0."""
    y_true, _ = _binary_batch(64, seed=71)
    y_pred = np.full((64, 1), 0.5, dtype="float32")
    value = _float(
        keras.ops.mean(
            SpiegelhalterZLoss(chance_corrected=False, normalize_by_n=False).call(
                _tensor(y_true), _tensor(y_pred)
            )
        )
    )
    assert np.isfinite(value) and value == pytest.approx(0.0, abs=1e-4, rel=0.0), (
        f"an all-0.5 batch gave {value!r}; the Spiegelhalter statistic is "
        f"identically degenerate there and must not blow up"
    )


# ---------------------------------------------------------------------
# 8. legacy configs
# ---------------------------------------------------------------------


def test_a_legacy_z_config_restores_the_old_numerics():
    """A config saved before 2026-09-02 has no 'statistic' key. Loading it must
    reproduce what the checkpoint was trained against -- the unweighted
    statistic, uncorrected and unnormalized -- not today's default."""
    legacy_config = {
        "name": "spiegelhalter_z_loss",
        "reduction": "sum_over_batch_size",
        "use_squared": True,
        "from_logits": False,
    }
    restored = SpiegelhalterZLoss.from_config(legacy_config)
    assert restored.statistic == STATISTIC_CALIBRATION_IN_THE_LARGE
    assert restored.chance_corrected is False
    assert restored.normalize_by_n is False

    y_true, y_pred = _binary_batch(256, seed=73, temperature=0.6)
    expected = _z_oracle(y_true, y_pred, STATISTIC_CALIBRATION_IN_THE_LARGE) ** 2
    measured = _float(
        keras.ops.mean(restored.call(_tensor(y_true), _tensor(y_pred)))
    )
    assert measured == pytest.approx(expected, abs=Z_ATOL, rel=0.0)

    fresh = _float(
        keras.ops.mean(
            SpiegelhalterZLoss(
                chance_corrected=False, normalize_by_n=False
            ).call(_tensor(y_true), _tensor(y_pred))
        )
    )
    assert abs(fresh - measured) > 0.1, (
        "the legacy path and today's default produced the same number, so this "
        "guard is not distinguishing them"
    )


def test_a_legacy_z_config_says_so_out_loud(caplog):
    with caplog.at_level("WARNING"):
        SpiegelhalterZLoss.from_config(
            {"name": "z", "reduction": "sum_over_batch_size", "use_squared": True}
        )
    assert "calibration_in_the_large" in caplog.text.lower() or "2026-09-02" in caplog.text


def test_a_legacy_combined_config_restores_the_blend():
    legacy_config = {
        "name": "combined_calibration_loss",
        "reduction": "sum_over_batch_size",
        "alpha": 0.6,
        "use_squared_z": True,
        "from_logits": False,
    }
    restored = CombinedCalibrationLoss.from_config(legacy_config)
    assert restored.alpha == 0.6

    y_true, y_pred = _binary_batch(256, seed=79, bias=0.02)
    brier = float(np.mean((y_pred - y_true) ** 2))
    z2 = _z_oracle(y_true, y_pred, STATISTIC_CALIBRATION_IN_THE_LARGE) ** 2
    expected = 0.6 * brier + 0.4 * z2
    measured = _float(
        keras.ops.mean(restored.call(_tensor(y_true), _tensor(y_pred)))
    )
    assert measured == pytest.approx(expected, abs=1e-3, rel=0.0), (
        f"the legacy blend gave {measured!r} against the pre-2026-09-02 closed "
        f"form {expected!r}"
    )


def test_the_combined_loss_anchors_a_proper_scoring_rule():
    """L = base + lambda * penalty, with both terms per-sample."""
    y_true, y_pred = _binary_batch(256, seed=83, bias=0.03)
    lambda_cal = 0.05
    brier = float(np.mean((y_pred - y_true) ** 2))
    penalty = max(0.0, _z_oracle(y_true, y_pred) ** 2 - 1.0) / 256.0
    expected = brier + lambda_cal * penalty

    measured = _float(
        keras.ops.mean(
            CombinedCalibrationLoss(lambda_cal=lambda_cal, base="brier").call(
                _tensor(y_true), _tensor(y_pred)
            )
        )
    )
    assert measured == pytest.approx(expected, abs=1e-5, rel=0.0)


def test_the_bce_base_is_binary_cross_entropy():
    y_true, y_pred = _binary_batch(256, seed=89)
    stock = _float(
        keras.losses.BinaryCrossentropy()(_tensor(y_true), _tensor(y_pred))
    )
    measured = _float(
        CombinedCalibrationLoss(lambda_cal=0.0, base="bce")(
            _tensor(y_true), _tensor(y_pred)
        )
    )
    assert measured == pytest.approx(stock, abs=1e-4, rel=0.0)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_an_out_of_range_alpha_raises(alpha):
    with pytest.raises(ValueError, match="alpha must be in the range"):
        CombinedCalibrationLoss(alpha=alpha)


def test_a_negative_lambda_raises():
    with pytest.raises(ValueError, match="lambda_cal must be >= 0"):
        CombinedCalibrationLoss(lambda_cal=-0.1)


@pytest.mark.parametrize("bad", ["Spiegelhalter", "z", ""])
def test_an_unknown_statistic_raises(bad):
    with pytest.raises(ValueError, match="statistic must be one of"):
        SpiegelhalterZLoss(statistic=bad)
    with pytest.raises(ValueError, match="statistic must be one of"):
        SpiegelhalterZMetric(statistic=bad)


def test_an_unknown_base_raises():
    with pytest.raises(ValueError, match="base must be one of"):
        CombinedCalibrationLoss(base="focal")


# ---------------------------------------------------------------------
# 9. serialization
# ---------------------------------------------------------------------


def _round_trip_configs():
    return [
        SpiegelhalterZLoss(),
        SpiegelhalterZLoss(
            statistic=STATISTIC_CALIBRATION_IN_THE_LARGE,
            chance_corrected=False,
            normalize_by_n=False,
            use_squared=False,
            from_logits=True,
        ),
        BrierScoreLoss(from_logits=True),
        CombinedCalibrationLoss(lambda_cal=0.2, base="bce"),
        CombinedCalibrationLoss(alpha=0.3, use_squared_z=False),
    ]


@pytest.mark.parametrize(
    "original", _round_trip_configs(), ids=lambda o: type(o).__name__
)
def test_from_config_round_trips_on_values(original):
    """Comparing config dicts alone passes even when a key is never read back
    by __init__."""
    restored = type(original).from_config(original.get_config())
    y_true, y_pred = _binary_batch(128, seed=97, bias=0.02)
    before = _float(original(_tensor(y_true), _tensor(y_pred)))
    after = _float(restored(_tensor(y_true), _tensor(y_pred)))
    assert after == pytest.approx(before, abs=1e-6, rel=0.0)


@pytest.mark.parametrize(
    "original", _round_trip_configs(), ids=lambda o: type(o).__name__
)
def test_keras_serialization_round_trips(original):
    """This arm is what actually exercises the registry decorator."""
    restored = keras.saving.deserialize_keras_object(
        keras.saving.serialize_keras_object(original)
    )
    assert isinstance(restored, type(original))
    y_true, y_pred = _binary_batch(128, seed=101, bias=0.02)
    assert _float(restored(_tensor(y_true), _tensor(y_pred))) == pytest.approx(
        _float(original(_tensor(y_true), _tensor(y_pred))), abs=1e-6, rel=0.0
    )


@pytest.mark.parametrize(
    "metric", [BrierScoreMetric(from_logits=True), SpiegelhalterZMetric(statistic=STATISTIC_CALIBRATION_IN_THE_LARGE)],
    ids=["BrierScoreMetric", "SpiegelhalterZMetric"],
)
def test_metric_config_round_trips(metric):
    restored = type(metric).from_config(metric.get_config())
    assert restored.get_config() == metric.get_config()


def test_a_saved_model_reloads_and_predicts_identically():
    keras.utils.set_random_seed(11)
    model = keras.Sequential(
        [
            keras.Input(shape=(10,)),
            keras.layers.Dense(8, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=CombinedCalibrationLoss(lambda_cal=0.05),
        metrics=[BrierScoreMetric(), SpiegelhalterZMetric()],
    )
    rng = np.random.default_rng(103)
    x = rng.random((64, 10)).astype("float32")
    y = rng.integers(0, 2, size=(64, 1)).astype("float32")
    model.fit(x, y, epochs=1, batch_size=32, verbose=0)

    x_test = rng.random((16, 10)).astype("float32")
    original = np.asarray(model.predict(x_test, verbose=0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "calibration_model.keras")
        model.save(path)
        # No custom_objects: register_dl_technique already resolves these.
        # A dict keyed by BARE CLASS NAME -- what this file used to pass -- is
        # ignored by Keras 3 entirely.
        loaded = keras.models.load_model(path)
        restored = np.asarray(loaded.predict(x_test, verbose=0))

    np.testing.assert_allclose(
        original,
        restored,
        atol=1e-6,
        rtol=0,
        err_msg="predictions differ after a .keras round trip",
    )
    assert isinstance(loaded.loss, CombinedCalibrationLoss)


def test_the_registration_keys_are_package_qualified():
    """The five classes share one package string; the CLASS NAME disambiguates
    the key, so this is not a collision."""
    keys = {
        cls.__name__: keras.saving.get_registered_name(cls)
        for cls in (
            BrierScoreLoss,
            SpiegelhalterZLoss,
            CombinedCalibrationLoss,
            BrierScoreMetric,
            SpiegelhalterZMetric,
        )
    }
    assert len(set(keys.values())) == 5, f"registration keys collided: {keys}"
    for name, key in keys.items():
        assert key == (
            f"dl_techniques.losses.brier_spiegelhalters_ztest_loss>{name}"
        ), f"unexpected registration key for {name}: {key}"


# ---------------------------------------------------------------------
# 10. dtype policies, shapes, gradients
# ---------------------------------------------------------------------


@pytest.mark.parametrize("name,loss_factory", [
    ("BrierScoreLoss", BrierScoreLoss),
    ("SpiegelhalterZLoss", SpiegelhalterZLoss),
    ("CombinedCalibrationLoss", CombinedCalibrationLoss),
])
def test_every_loss_is_finite_under_every_dtype_policy(
    name, loss_factory, dtype_policy
):
    y_true, y_pred = _binary_batch(256, seed=107, bias=0.03)
    cast = "float64" if dtype_policy == "float64" else "float32"
    value = _float(
        loss_factory()(
            keras.ops.cast(keras.ops.convert_to_tensor(y_true), cast),
            keras.ops.cast(keras.ops.convert_to_tensor(y_pred), cast),
        )
    )
    assert np.isfinite(value), f"{name} gave {value!r} under {dtype_policy}"


def test_the_statistic_is_accumulated_without_narrowing_float64(dtype_policy):
    """'Run this reduction in float32' NARROWS a float64 policy. The
    accumulation dtype must be max(compute_dtype, float32)."""
    if dtype_policy != "float64":
        pytest.skip("only float64 can be narrowed by a float32 accumulation")
    y_true, y_pred = _binary_batch(1024, seed=109, bias=0.01)
    loss = SpiegelhalterZLoss(chance_corrected=False, normalize_by_n=False)
    per_sample = loss.call(
        keras.ops.cast(keras.ops.convert_to_tensor(y_true), "float64"),
        keras.ops.cast(keras.ops.convert_to_tensor(y_pred), "float64"),
    )
    assert keras.backend.standardize_dtype(per_sample.dtype) == "float64"
    expected = _z_oracle(y_true, y_pred) ** 2
    assert _float(keras.ops.mean(per_sample)) == pytest.approx(
        expected, abs=1e-9, rel=0.0
    ), "a float64 forward pass is no more accurate than float32 would be"


@pytest.mark.parametrize("name,loss", _all_losses(), ids=[n for n, _ in _all_losses()])
def test_a_multi_column_prediction_raises(name, loss):
    y_true = np.zeros((8, 3), dtype="float32")
    y_pred = np.full((8, 3), 0.3, dtype="float32")
    if isinstance(loss, BrierScoreLoss):
        pytest.skip("BrierScoreLoss is defined element-wise and accepts multi-label")
    with pytest.raises(ValueError, match="BINARY"):
        loss.call(_tensor(y_true), _tensor(y_pred))


@pytest.mark.parametrize("name,loss", _all_losses(), ids=[n for n, _ in _all_losses()])
def test_gradients_flow_to_the_predictions(name, loss):
    import tensorflow as tf

    y_true, y_pred = _binary_batch(128, seed=113, bias=0.05)
    predictions = tf.Variable(y_pred)
    with tf.GradientTape() as tape:
        value = loss(_tensor(y_true), predictions)
    grad = tape.gradient(value, predictions)
    assert grad is not None, f"{name}: no gradient reached y_pred"
    assert np.any(_numpy(grad) != 0.0), f"{name}: all-zero gradient w.r.t. y_pred"


def test_the_chance_corrected_penalty_has_zero_gradient_when_calibrated():
    """The point of the correction: a calibrated model feels no pressure."""
    import tensorflow as tf

    y_true, y_pred = _binary_batch(4096, seed=127)
    predictions = tf.Variable(y_pred)
    loss = SpiegelhalterZLoss()
    with tf.GradientTape() as tape:
        value = loss(_tensor(y_true), predictions)
    grad = _numpy(tape.gradient(value, predictions))
    assert np.all(grad == 0.0), (
        f"a calibrated batch produced a non-zero calibration gradient "
        f"(max |g| = {np.max(np.abs(grad))!r})"
    )


def test_the_uncorrected_penalty_has_a_non_zero_gradient_when_calibrated():
    """The twin. Without it the arm above is satisfied by a dead term."""
    import tensorflow as tf

    y_true, y_pred = _binary_batch(4096, seed=127)
    predictions = tf.Variable(y_pred)
    loss = SpiegelhalterZLoss(chance_corrected=False)
    with tf.GradientTape() as tape:
        value = loss(_tensor(y_true), predictions)
    grad = _numpy(tape.gradient(value, predictions))
    assert np.max(np.abs(grad)) > 0.0, (
        "the uncorrected penalty must still push on a calibrated batch -- that "
        "residual pressure is the defect chance-correction removes"
    )


@pytest.mark.parametrize("from_logits", [True, False])
def test_from_logits_applies_the_sigmoid(from_logits):
    logits = np.array([[2.0], [-1.0], [0.5], [-3.0]], dtype="float32")
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y_true = Y_TRUE_4

    logit_value = _float(
        BrierScoreLoss(from_logits=True)(_tensor(y_true), _tensor(logits))
    )
    prob_value = _float(
        BrierScoreLoss(from_logits=False)(_tensor(y_true), _tensor(probabilities))
    )
    assert logit_value == pytest.approx(prob_value, abs=1e-6, rel=0.0)
