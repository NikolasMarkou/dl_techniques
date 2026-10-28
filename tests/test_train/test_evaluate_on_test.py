"""``train.common.evaluation.evaluate_on_test``: the one place a trainer scores its test split.

The trainers that select on a held-out slice of train (BEiT, ET) call it once after ``fit``.
The two datasets below have different targets, so a call that scored the wrong one returns a
different number.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.common.evaluation import evaluate_on_test


def _model():
    model = keras.Sequential([keras.layers.Input((3,)), keras.layers.Dense(1, use_bias=False)])
    model.layers[0].set_weights([np.ones((3, 1), np.float32)])
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    return model


def _dataset(target):
    x = np.ones((8, 3), np.float32)
    y = np.full((8, 1), target, np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(4)


def test_it_scores_the_dataset_it_is_given_and_returns_floats() -> None:
    model = _model()  # predicts 3.0 everywhere
    metrics = evaluate_on_test(model, _dataset(5.0))
    assert "loss" in metrics
    assert all(type(value) is float for value in metrics.values())
    assert metrics["loss"] == pytest.approx(4.0)  # (5 - 3)^2


def test_a_different_dataset_gives_a_different_number() -> None:
    model = _model()
    assert evaluate_on_test(model, _dataset(5.0))["loss"] != pytest.approx(
        evaluate_on_test(model, _dataset(3.0))["loss"])
    assert evaluate_on_test(model, _dataset(3.0))["loss"] == pytest.approx(0.0)


def test_it_does_not_train() -> None:
    model = _model()
    before = [w.copy() for w in model.get_weights()]
    evaluate_on_test(model, _dataset(5.0))
    for a, b in zip(before, model.get_weights()):
        np.testing.assert_array_equal(a, b)
