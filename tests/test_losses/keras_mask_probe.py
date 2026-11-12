"""Shared arms for a Keras mask and per-timestep weights against the five per-sample losses.

Not a test module (no ``test_`` prefix): the helpers below are imported by
``test_multi_labels_loss.py``, ``test_clustering_loss.py`` and
``test_utilization_loss.py`` (plan-2026-11-11T223825-64d1b401, D-014).

Why these arms exist. ``keras.losses.Loss.__call__`` reads the ``_keras_mask`` of
``y_pred`` and multiplies it into the per-sample losses as a weight. A mask has the
shape of ``y_pred`` minus its last axis, so against a ``(batch,)`` row vector it is
either a shape error (batch != the masked axis) or, when the two lengths coincide,
silent mis-weighting. The same holds for a ``(batch, time)`` ``sample_weight``. The old
scalar-returning losses ignored both without saying so.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf


def with_keras_mask(array, mask):
    """A float32 tensor carrying ``mask`` as ``_keras_mask``, as a masking layer leaves it."""
    tensor = tf.constant(np.asarray(array, dtype="float32"))
    tensor._keras_mask = tf.constant(np.asarray(mask, dtype=bool))
    return tensor


def assert_refuses_a_keras_mask(make_loss, y_true, y_pred, mask_shape):
    """``__call__`` raises a ``ValueError`` that names the Keras mask, at any reduction."""
    mask = np.ones(mask_shape, dtype=bool)
    for reduction in ("sum_over_batch_size", "none"):
        with pytest.raises(ValueError, match="Keras mask"):
            make_loss(reduction=reduction)(y_true, with_keras_mask(y_pred, mask))


def assert_refuses_per_timestep_weights(make_loss, y_true, y_pred, weight_shape):
    """``__call__`` raises a ``ValueError`` that names ``sample_weight`` for a non-row weight."""
    with pytest.raises(ValueError, match="sample_weight"):
        make_loss()(y_true, y_pred, sample_weight=np.ones(weight_shape, "float32"))


def assert_a_per_sample_mask_is_left_to_keras(make_loss, y_true, y_pred):
    """A rank-1 ``(batch,)`` mask is a 0/1 sample weight: ``sum(rows * m) / sum(m)``."""
    batch = np.shape(y_pred)[0]
    mask = np.ones(batch, dtype=bool)
    mask[1] = False
    rows = np.asarray(make_loss(reduction="none")(y_true, y_pred), dtype="float64")
    got = float(make_loss()(y_true, with_keras_mask(y_pred, mask)))
    want = float((rows * mask).sum() / mask.sum())
    assert abs(got - want) < 1e-6, (got, want)


def evaluate_through_a_masking_lstm(loss, x, units):
    """``compile(loss=...)`` on ``Masking(0) -> LSTM(return_sequences)``, then ``evaluate``.

    All-zero timesteps of ``x`` are padding, so the LSTM output carries a
    ``(batch, time)`` ``_keras_mask``. Returns ``(value, prediction, mask)``: the
    evaluated loss, the model output (``(batch, time, units)``) and the mask.
    """
    keras.utils.set_random_seed(1)
    inputs = keras.Input(x.shape[1:])
    hidden = keras.layers.Masking(0.0)(inputs)
    hidden = keras.layers.LSTM(units, return_sequences=True)(hidden)
    model = keras.Model(inputs, hidden)
    model.compile("adam", loss=loss)
    batch = len(x)
    value = model.evaluate(
        x, np.zeros((batch, 1), "float32"), batch_size=batch, verbose=0
    )
    prediction = model.predict(x, batch_size=batch, verbose=0)
    return float(value), prediction.astype("float64"), np.any(x != 0.0, axis=-1)
