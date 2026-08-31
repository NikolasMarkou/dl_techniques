"""Tests for ``dl_techniques.losses.multistep_loss``.

Every value assertion here is anchored to a **plain-numpy transcription of the
published formula** (``_reference`` below), never to what ``MultistepLoss``
happens to return. The formulas are ADAM's, from Svetunkov (2023) section 11.3:

    MSEh  = mean_t e_{t+h|t}^2
    TMSE  = sum_{j=1..h} mean_t e_{t+j|t}^2
    GTMSE = sum_{j=1..h} log( mean_t e_{t+j|t}^2 )
    MSCE  = mean_t ( sum_{j=1..h} e_{t+j|t} )^2

Measured 2026-08-31 across ``{(7,5), (7,5,3)} x {h=None, h=3} x 4 aggregations``
= 16 cells: all 16 agree with the numpy reference to ``atol=1e-5``.

Two structural claims get their own single-claim guards rather than living here:
``test_the_gtmse_surrogate_matches_the_exact_form.py`` and
``test_the_mseh_starves_other_horizons.py``.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.losses.multistep_loss import (
    MULTISTEP_AGGREGATIONS,
    MultistepLoss,
    create_multistep_loss,
)

# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------

BATCH, HORIZON, FEATURES = 7, 5, 3


def _pair(shape, seed=0):
    """Return a reproducible ``(y_true, y_pred)`` pair of the given shape."""
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=shape).astype("float32"),
        rng.normal(size=shape).astype("float32"),
    )


def _reference(y_true, y_pred, aggregation, h, epsilon=1e-8):
    """Transcribe the published formula in plain numpy.

    This is the oracle. It shares no code with the implementation -- in
    particular it does NOT use the per-sample GTMSE surrogate, but the exact
    batch-global ``sum_j log(mean_i e_ij^2)``.
    """
    error = y_pred - y_true
    horizon = error.shape[1]
    hh = horizon if h is None else h

    squared = error ** 2
    per_step = squared.mean(axis=tuple(range(2, squared.ndim))) if squared.ndim > 2 else squared

    if aggregation == "mseh":
        return per_step[:, hh - 1].mean()
    if aggregation == "tmse":
        return per_step[:, :hh].sum(axis=1).mean()
    if aggregation == "gtmse":
        step_means = per_step[:, :hh].mean(axis=0)
        return np.log(np.maximum(step_means, epsilon)).sum()
    if aggregation == "msce":
        cumulative = error[:, :hh].sum(axis=1) ** 2
        if cumulative.ndim > 1:
            cumulative = cumulative.mean(axis=tuple(range(1, cumulative.ndim)))
        return cumulative.mean()
    raise AssertionError(f"unhandled aggregation {aggregation!r}")


# ---------------------------------------------------------------------
# Test 1 -- the four formulas, against numpy
# ---------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(BATCH, HORIZON), (BATCH, HORIZON, FEATURES)])
@pytest.mark.parametrize("h", [None, 3])
@pytest.mark.parametrize("aggregation", MULTISTEP_AGGREGATIONS)
def test_matches_the_published_formula(shape, h, aggregation):
    y_true, y_pred = _pair(shape)
    loss = MultistepLoss(aggregation, h=h)

    got = float(keras.ops.convert_to_numpy(loss(y_true, y_pred)))
    want = float(_reference(y_true, y_pred, aggregation, h))

    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------
# Test 2 -- MSCE is NOT TMSE: the signs must survive to the aggregation
# ---------------------------------------------------------------------

def test_msce_lets_over_and_under_forecasts_cancel():
    """A +1 then -1 error pair costs MSCE nothing and TMSE two.

    Squaring BEFORE the horizon sum would collapse MSCE onto TMSE and the
    difference is invisible to every shape and finiteness check. This is the
    whole reason MSCE is worth having: a stock position accumulated over a lead
    time nets off, and the loss should say so.
    """
    y_true = np.zeros((4, 2), dtype="float32")
    y_pred = np.tile(np.array([[1.0, -1.0]], dtype="float32"), (4, 1))

    assert float(MultistepLoss("msce")(y_true, y_pred)) == pytest.approx(0.0, abs=1e-6)
    assert float(MultistepLoss("tmse")(y_true, y_pred)) == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------
# Test 3 -- per-sample return, and sample_weight actually selects rows
# ---------------------------------------------------------------------

@pytest.mark.parametrize("aggregation", MULTISTEP_AGGREGATIONS)
def test_call_returns_one_value_per_sample(aggregation):
    """``call()`` must return ``(batch,)``, never a scalar.

    A scalar return does not *ignore* ``sample_weight``; Keras multiplies
    ``values * sample_weight`` before reducing, so it charges every row the batch
    aggregate and quietly kills ``reduction=``. These classes must therefore stay
    OUT of ``test_the_premature_scalar_family_is_pinned.py``.
    """
    y_true, y_pred = _pair((BATCH, HORIZON, FEATURES))
    per_sample = MultistepLoss(aggregation).call(
        keras.ops.convert_to_tensor(y_true), keras.ops.convert_to_tensor(y_pred)
    )
    assert tuple(per_sample.shape) == (BATCH,)


@pytest.mark.parametrize("aggregation", MULTISTEP_AGGREGATIONS)
def test_sample_weight_selects_rows(aggregation):
    """Zeroing one row's weight must change the result by that row's share."""
    y_true, y_pred = _pair((BATCH, HORIZON))
    loss = MultistepLoss(aggregation)

    weights = np.ones((BATCH,), dtype="float32")
    weights[0] = 0.0

    unweighted = float(loss(y_true, y_pred))
    weighted = float(loss(y_true, y_pred, sample_weight=weights))

    per_sample = keras.ops.convert_to_numpy(
        loss.call(keras.ops.convert_to_tensor(y_true), keras.ops.convert_to_tensor(y_pred))
    )
    expected = float((per_sample * weights).sum() / BATCH)

    assert weighted != pytest.approx(unweighted, abs=1e-6)
    np.testing.assert_allclose(weighted, expected, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------
# Test 4 -- constructor validation
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"aggregation": "mse"},          # a real Keras name, but not one of ours
        {"aggregation": "MSEh"},         # the class is case-SENSITIVE; the factory is not
        {"aggregation": "tmse", "h": 0},
        {"aggregation": "tmse", "h": -1},
        {"aggregation": "tmse", "h": 2.0},
        {"aggregation": "tmse", "error_power": 0.0},
        {"aggregation": "tmse", "epsilon": 0.0},
    ],
)
def test_rejects_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        MultistepLoss(**kwargs)


def test_rejects_a_missing_horizon_axis():
    loss = MultistepLoss("tmse")
    with pytest.raises(ValueError, match="horizon axis"):
        loss.call(keras.ops.convert_to_tensor(np.zeros((4,), dtype="float32")),
                  keras.ops.convert_to_tensor(np.ones((4,), dtype="float32")))


def test_rejects_h_beyond_the_horizon():
    y_true, y_pred = _pair((BATCH, HORIZON))
    with pytest.raises(ValueError, match="exceeds the horizon"):
        MultistepLoss("tmse", h=HORIZON + 1)(y_true, y_pred)


# ---------------------------------------------------------------------
# Test 5 -- the factory raises, never filters-and-drops
# ---------------------------------------------------------------------

@pytest.mark.parametrize("aggregation", MULTISTEP_AGGREGATIONS)
def test_factory_builds_each_aggregation(aggregation):
    loss = create_multistep_loss(aggregation.upper(), h=2)
    assert isinstance(loss, MultistepLoss)
    assert loss.aggregation == aggregation
    assert loss.h == 2


def test_factory_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown multistep loss"):
        create_multistep_loss("mse")


@pytest.mark.parametrize("bogus", ["horizon", "power", "aggregation"])
def test_factory_rejects_an_unknown_keyword(bogus):
    """A dropped keyword is a dead knob with no symptom -- it must RAISE.

    ``aggregation`` is in this list deliberately: passing it would silently
    conflict with the positional ``name`` the factory dispatches on.
    """
    with pytest.raises(ValueError, match="unexpected keyword"):
        create_multistep_loss("tmse", **{bogus: 3})


# ---------------------------------------------------------------------
# Test 6 -- serialization round-trip, no custom_objects
# ---------------------------------------------------------------------

@pytest.mark.parametrize("aggregation", MULTISTEP_AGGREGATIONS)
def test_serialization_round_trip(aggregation):
    original = MultistepLoss(aggregation, h=3, error_power=2.0, epsilon=1e-7)
    restored = MultistepLoss.from_config(original.get_config())

    assert restored.aggregation == aggregation
    assert restored.h == 3
    assert restored.epsilon == pytest.approx(1e-7)

    y_true, y_pred = _pair((BATCH, HORIZON))
    np.testing.assert_allclose(
        float(original(y_true, y_pred)), float(restored(y_true, y_pred)),
        rtol=0, atol=1e-6,
    )


def test_survives_model_save_and_load():
    """The registration key must resolve without ``custom_objects``."""
    from tests.optimizer_state import build_optimizer_state

    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(HORIZON)])
    model.compile(optimizer="adam", loss=MultistepLoss("gtmse", h=4))

    x = np.random.default_rng(3).normal(size=(8, 4)).astype("float32")
    y = np.random.default_rng(4).normal(size=(8, HORIZON)).astype("float32")
    model(x)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "multistep.keras")
        build_optimizer_state(model)
        model.save(path)
        loaded = keras.models.load_model(path)  # NO custom_objects

    assert type(loaded.loss).__name__ == "MultistepLoss"
    assert loaded.loss.aggregation == "gtmse"
    assert loaded.loss.h == 4
    np.testing.assert_allclose(
        float(model.loss(y, model(x))), float(loaded.loss(y, loaded(x))),
        rtol=0, atol=1e-6,
    )


# ---------------------------------------------------------------------
# Test 7 -- error_power gives the MAE analogues
# ---------------------------------------------------------------------

def test_error_power_one_gives_the_mae_analogue():
    """``error_power=1`` turns TMSE into ADAM's TMAE: sum of per-step MAEs."""
    y_true, y_pred = _pair((BATCH, HORIZON))
    got = float(MultistepLoss("tmse", error_power=1.0)(y_true, y_pred))
    want = float(np.abs(y_pred - y_true).sum(axis=1).mean())
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)
