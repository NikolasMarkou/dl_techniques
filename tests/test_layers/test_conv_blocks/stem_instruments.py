"""Shared instruments for the ``ConvStem`` test files in this directory.

No ``test_`` prefix, so pytest does not collect it. The four stem files import
from here instead of each carrying a copy:

* ``test_stem.py`` (the contract),
* ``test_the_stem_follows_its_own_dtype_policy.py``,
* ``test_a_stem_activation_survives_the_config_round_trip.py``,
* ``test_the_two_convs_of_a_stem_draw_independently.py``.

Interface contract, one entry per name:

* ``B``, ``EPS32`` -- the batch size every stem test uses, and the float32
  machine epsilon the activation comparisons are scaled by.
* ``rng`` -- a pytest fixture (``np.random.default_rng(0)``). A fixture is
  found by name in the module that uses it, so a test module binds it with
  ``from . import stem_instruments`` and ``rng = stem_instruments.rng``
  (an ``import`` of the name itself reads as unused to pyflakes).
* ``random_input(rng, shape)`` -- a float32 standard-normal array;
  ``shape`` defaults to ``(B, 16, 16, 3)``.
* ``to_numpy(tensor)`` -- ``keras.ops.convert_to_numpy``.
* ``randomize_weights(layer, rng)`` -- assigns every weight of a BUILT layer a
  distinct non-trivial value; raises if a weight shape does not accept it.
* ``roundtrip_exact(layer, input_shape, data, name, tmp_path)`` -- saves a
  functional model around ``layer``, reloads it and asserts every weight and
  the inference output are bit-identical (``rtol=0, atol=0``); returns the
  reloaded model.
* ``unlocatable_activation()`` -- a fresh activation Layer instance whose class
  Keras cannot locate by name (see its docstring); used wherever a test needs
  the "cannot clone" refusal.
"""

import os

import keras
import numpy as np
import pytest

B = 2
EPS32 = float(np.finfo(np.float32).eps)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def random_input(rng, shape=(B, 16, 16, 3)):
    return rng.standard_normal(shape).astype("float32")


def to_numpy(tensor):
    return keras.ops.convert_to_numpy(tensor)


def randomize_weights(layer, rng):
    """Give every weight a distinct non-trivial value (BN statistics included).

    A freshly built BatchNormalization is the identity at ``training=False``
    (moving mean 0, variance 1, gamma 1, beta 0), so a comparison made on it
    cannot see a lost norm; variance is kept positive.
    """
    for weight in layer.weights:
        values = rng.uniform(0.5, 1.5, size=tuple(weight.shape)).astype("float32")
        if "moving_variance" not in weight.path and "kernel" not in weight.path:
            values = values - 1.0
        weight.assign(values)


def roundtrip_exact(layer, input_shape, data, name, tmp_path):
    """Save a functional model around ``layer``, reload it, demand bit identity."""
    inputs = keras.Input(shape=input_shape)
    model = keras.Model(inputs, layer(inputs))
    y0 = to_numpy(model(data, training=False))
    path = os.path.join(tmp_path, f"{name}.keras")
    model.save(path)
    loaded = keras.models.load_model(path)
    original = {w.path: to_numpy(w) for w in model.weights}
    restored = {w.path: to_numpy(w) for w in loaded.weights}
    assert set(original) == set(restored)
    for key in original:
        np.testing.assert_allclose(
            original[key], restored[key], rtol=0.0, atol=0.0,
            err_msg=f"weight {key} changed across the .keras round trip",
        )
    y1 = to_numpy(loaded(data, training=False))
    assert np.isfinite(y1).all()
    np.testing.assert_allclose(
        y0, y1, rtol=0.0, atol=0.0,
        err_msg="reloaded model is not bit-identical to the original",
    )
    return loaded


def unlocatable_activation():
    """A custom activation Layer Keras cannot locate by name.

    Defined inside a function on purpose: Keras resolves a serialized class by
    ``module`` + ``class_name``, so a module-level class IS found again (it is
    importable) and would not exercise the failure. A function-local class is
    neither registered nor an attribute of its module.
    """

    class UnlocatableActivation(keras.layers.Layer):
        def call(self, inputs):
            return keras.ops.relu(inputs)

    return UnlocatableActivation()
