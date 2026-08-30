"""Tests for ``dl_techniques.losses.huber_loss``.

This module had ZERO tests before this file.

The point of this file is the **stock-Keras control**: ``keras.losses.Huber``
implements the same textbook definition, so it is an oracle this repo does not
own and cannot accidentally bend to match its own bug. Every value assertion
here is anchored either to stock Keras or to a hand-computed expectation, never
to what ``HuberLoss`` currently happens to return.

Written RED-first for plan ``plan-2026-08-30T203107-30455f66`` step 5. Measured
against the pre-fix tree, with ``sample_weight=[1., 1., 1., 0.]``:

*   this repo:  ``0.1258224``
*   stock Keras: ``0.16704728``

The cause is ``call()`` ending with an axis-less ``ops.mean(...)``: it returns a
SCALAR, so ``keras.losses.Loss.__call__`` broadcasts that scalar against
``sample_weight`` and reduces, yielding exactly ``unweighted * mean(weights)``.
Every row is charged the batch aggregate and WHICH rows were weighted is thrown
away. ``reduction=`` is dead for the same reason. Stock Keras reduces over the
last axis only and returns ``(batch,)``.
"""

import keras
import numpy as np
import pytest

from dl_techniques.losses.huber_loss import HuberLoss

# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------

# Fixed inputs spanning BOTH branches of the piecewise definition at delta=1.0:
# row-wise abs errors are 0.5/2.5/0.1 (mixed), 3.0/0.2/0.4, 0.05/0.05/0.05
# (all quadratic) and 5.0/5.0/5.0 (all linear).
Y_TRUE = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-2.0, -2.0, -2.0],
        [0.5, 0.5, 0.5],
    ],
    dtype="float32",
)
Y_PRED = np.array(
    [
        [0.5, 2.5, -0.1],
        [4.0, 0.8, 1.4],
        [-2.05, -1.95, -2.05],
        [5.5, -4.5, 5.5],
    ],
    dtype="float32",
)
SAMPLE_WEIGHT = np.array([1.0, 1.0, 1.0, 0.0], dtype="float32")


def _tensors():
    return (
        keras.ops.convert_to_tensor(Y_TRUE),
        keras.ops.convert_to_tensor(Y_PRED),
    )


def _f(x):
    return float(keras.ops.convert_to_numpy(x))


def _reference_huber_per_sample(y_true, y_pred, delta):
    """Hand-computed per-sample Huber, in numpy, independent of both impls."""
    e = np.abs(y_true - y_pred)
    per_element = np.where(e <= delta, 0.5 * e**2, delta * (e - 0.5 * delta))
    return per_element.mean(axis=-1)


# ---------------------------------------------------------------------
# the stock-Keras control
# ---------------------------------------------------------------------


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_agrees_with_stock_keras_huber_with_sample_weight(delta):
    """The oracle this repo does not own, exercised where the bug lives.

    RED before the step-5 fix: 0.1258224 here vs 0.16704728 stock at delta=1.0.
    """
    y_true, y_pred = _tensors()
    w = keras.ops.convert_to_tensor(SAMPLE_WEIGHT)

    ours = _f(HuberLoss(delta=delta)(y_true, y_pred, sample_weight=w))
    stock = _f(keras.losses.Huber(delta=delta)(y_true, y_pred, sample_weight=w))

    assert ours == pytest.approx(stock, abs=1e-6, rel=0.0), (
        f"delta={delta}: this repo {ours!r} vs stock keras.losses.Huber {stock!r}"
    )


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_agrees_with_stock_keras_huber_without_sample_weight(delta):
    """Value-unchanged arm: the unweighted number must NOT move with the fix."""
    y_true, y_pred = _tensors()
    ours = _f(HuberLoss(delta=delta)(y_true, y_pred))
    stock = _f(keras.losses.Huber(delta=delta)(y_true, y_pred))
    assert ours == pytest.approx(stock, abs=1e-6, rel=0.0)


@pytest.mark.parametrize("reduction", ["none", "sum", "sum_over_batch_size"])
def test_agrees_with_stock_keras_huber_under_every_reduction(reduction):
    """`reduction` must be a live knob, and mean the same thing stock means."""
    y_true, y_pred = _tensors()
    ours = keras.ops.convert_to_numpy(
        HuberLoss(delta=1.0, reduction=reduction)(y_true, y_pred)
    )
    stock = keras.ops.convert_to_numpy(
        keras.losses.Huber(delta=1.0, reduction=reduction)(y_true, y_pred)
    )
    assert np.shape(ours) == np.shape(stock)
    np.testing.assert_allclose(ours, stock, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------
# the per-sample contract, independent of stock Keras
# ---------------------------------------------------------------------


def test_call_returns_one_value_per_sample():
    y_true, y_pred = _tensors()
    out = HuberLoss(delta=1.0).call(y_true, y_pred)
    assert tuple(keras.ops.shape(out)) == (Y_TRUE.shape[0],)


def test_call_matches_a_hand_computed_per_sample_reference():
    y_true, y_pred = _tensors()
    out = keras.ops.convert_to_numpy(HuberLoss(delta=1.0).call(y_true, y_pred))
    expected = _reference_huber_per_sample(Y_TRUE, Y_PRED, 1.0)
    np.testing.assert_allclose(out, expected, atol=1e-6, rtol=0.0)


def test_reduction_none_returns_one_value_per_sample():
    y_true, y_pred = _tensors()
    out = HuberLoss(delta=1.0, reduction="none")(y_true, y_pred)
    assert tuple(keras.ops.shape(out)) == (Y_TRUE.shape[0],)


def test_sample_weight_weights_the_named_rows():
    """Zeroing row 3 must drop row 3, not scale the batch aggregate by 0.75."""
    y_true, y_pred = _tensors()
    w = keras.ops.convert_to_tensor(SAMPLE_WEIGHT)

    unweighted = _f(HuberLoss(delta=1.0)(y_true, y_pred))
    weighted = _f(HuberLoss(delta=1.0)(y_true, y_pred, sample_weight=w))

    assert weighted != pytest.approx(unweighted * 0.75, abs=1e-9, rel=0.0), (
        "weighted value is exactly unweighted*mean(w): call() returned a scalar "
        "and sample_weight broadcast over it"
    )

    per_sample = _reference_huber_per_sample(Y_TRUE, Y_PRED, 1.0)
    expected = float((per_sample * SAMPLE_WEIGHT).sum() / len(SAMPLE_WEIGHT))
    assert weighted == pytest.approx(expected, abs=1e-6, rel=0.0)


def test_zero_weighting_a_row_makes_that_rows_values_irrelevant():
    """The row-identity property a scalar return cannot have."""
    y_true, y_pred = _tensors()
    w = keras.ops.convert_to_tensor(SAMPLE_WEIGHT)

    perturbed = Y_PRED.copy()
    perturbed[3, :] = 999.0  # the row weighted 0.0
    y_pred_perturbed = keras.ops.convert_to_tensor(perturbed)

    a = _f(HuberLoss(delta=1.0)(y_true, y_pred, sample_weight=w))
    b = _f(HuberLoss(delta=1.0)(y_true, y_pred_perturbed, sample_weight=w))
    assert a == pytest.approx(b, abs=1e-6, rel=0.0)


# ---------------------------------------------------------------------
# knobs, serialization, edges
# ---------------------------------------------------------------------


def test_delta_is_a_live_knob():
    y_true, y_pred = _tensors()
    small = _f(HuberLoss(delta=0.1)(y_true, y_pred))
    large = _f(HuberLoss(delta=5.0)(y_true, y_pred))
    assert small != pytest.approx(large, abs=1e-6, rel=0.0)


def test_name_is_a_live_knob():
    assert HuberLoss(name="custom_huber").name == "custom_huber"
    assert HuberLoss().name == "huber_loss"


def test_config_round_trip_re_evaluates_to_the_same_values():
    y_true, y_pred = _tensors()
    original = HuberLoss(delta=1.7, name="rt_huber", reduction="none")
    restored = HuberLoss.from_config(original.get_config())

    assert restored.delta == 1.7
    assert restored.name == "rt_huber"
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(restored(y_true, y_pred)),
        keras.ops.convert_to_numpy(original(y_true, y_pred)),
        atol=1e-6,
        rtol=0.0,
    )


def test_keras_registry_round_trip_re_evaluates_to_the_same_values():
    y_true, y_pred = _tensors()
    original = HuberLoss(delta=1.7)
    cfg = keras.saving.serialize_keras_object(original)
    restored = keras.saving.deserialize_keras_object(cfg)

    assert isinstance(restored, HuberLoss)
    assert _f(restored(y_true, y_pred)) == pytest.approx(
        _f(original(y_true, y_pred)), abs=1e-6, rel=0.0
    )


def test_batch_of_one():
    y_true = keras.ops.convert_to_tensor(np.array([[0.0, 0.0, 0.0]], dtype="float32"))
    y_pred = keras.ops.convert_to_tensor(np.array([[0.5, 2.5, -0.1]], dtype="float32"))
    out = HuberLoss(delta=1.0, reduction="none")(y_true, y_pred)
    assert tuple(keras.ops.shape(out)) == (1,)
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out),
        _reference_huber_per_sample(
            keras.ops.convert_to_numpy(y_true), keras.ops.convert_to_numpy(y_pred), 1.0
        ),
        atol=1e-6,
        rtol=0.0,
    )


def test_exact_zero_error_is_exactly_zero():
    y = keras.ops.convert_to_tensor(np.ones((4, 3), dtype="float32"))
    assert _f(HuberLoss(delta=1.0)(y, y)) == 0.0


def test_large_errors_stay_finite():
    y_true = keras.ops.convert_to_tensor(np.zeros((4, 3), dtype="float32"))
    y_pred = keras.ops.convert_to_tensor(np.full((4, 3), 1e8, dtype="float32"))
    out = keras.ops.convert_to_numpy(HuberLoss(delta=1.0, reduction="none")(y_true, y_pred))
    assert np.all(np.isfinite(out))


def test_dtype_casting_of_y_true():
    """`call` casts y_true to y_pred's dtype; an int y_true must work."""
    y_true = keras.ops.convert_to_tensor(np.zeros((4, 3), dtype="int32"))
    y_pred = keras.ops.convert_to_tensor(Y_PRED)
    out = HuberLoss(delta=1.0, reduction="none")(y_true, y_pred)
    assert tuple(keras.ops.shape(out)) == (4,)
