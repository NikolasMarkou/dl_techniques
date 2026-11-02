"""The check ``export`` makes of a select-ops ``tflite`` file compares CPU with CPU (D-042).

The ImageNet gate exported ``cliffordnet/nano`` to ``tflite`` on a GPU machine and ``export`` refused
it (exit 2): the Keras side of the check ran on the GPU, the interpreter on the CPU, and the two differ
by about 1e-3 on the logits (4.9e-4 against an allowance of 2.9e-4 on the check batch). The file was
faithful. The Keras side now runs inside ``tf.device("/CPU:0")``; the tolerance is unchanged.

A CPU-only test machine cannot show the device, so the first arm spies on ``tf.device``: the model is
called inside the scope. The others pin what the scope must NOT have loosened: a file that really
differs is refused, at the same boundary.
"""

import contextlib

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.api import exporting
from train.api.errors import ConfigError

SHAPE = (4, 4, 3)
SPEC = tf.TensorSpec((None, *SHAPE), tf.float32, name="input")
CLASSES = 5


def build(bias_shift: float = 0.0, seed: int = 0) -> keras.Model:
    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(SHAPE), keras.layers.Flatten(), keras.layers.Dense(CLASSES)
    ])
    kernel, bias = model.layers[-1].get_weights()
    model.layers[-1].set_weights([kernel, bias + bias_shift])
    return model


def blob_of(model: keras.Model, select: bool = False) -> bytes:
    """The ``.tflite`` bytes of ``model``; ``select`` adds TensorFlow operators (half-precision matmul needs them)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if select:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter.convert()


@pytest.fixture(scope="module")
def reference():
    return build()


@pytest.fixture(scope="module")
def reference_blob(reference):
    return blob_of(reference)


def scale_of(model: keras.Model) -> float:
    batch = np.random.default_rng(0).random((2, *SHAPE)).astype(np.float32)
    return max(1.0, float(np.abs(np.asarray(model(batch, training=False))).max()))


def check(blob, model, tmp_path):
    exporting._check_tflite_against_keras(blob, model, SPEC, tmp_path / "out.tflite")


def test_a_faithful_file_passes(reference, reference_blob, tmp_path):
    """Liveness for the refusals below: the same model and its own file agree."""
    check(reference_blob, reference, tmp_path)


def test_the_keras_side_runs_inside_a_cpu_device_scope(reference, reference_blob, tmp_path, monkeypatch):
    """Mutant ``no tf.device``: on a GPU machine the model would run on the GPU again."""
    real = tf.device
    state = {"depth": 0, "requested": [], "called_inside": []}

    @contextlib.contextmanager
    def spying_device(name):
        state["requested"].append(name)
        state["depth"] += 1
        try:
            with real(name):
                yield
        finally:
            state["depth"] -= 1

    class Spied(keras.layers.Layer):
        def call(self, x):
            # Building the Sequential traces this layer in a graph; only the eager call is the check's.
            if tf.executing_eagerly():
                state["called_inside"].append(state["depth"] > 0)
            return x

    spied = keras.Sequential([keras.layers.Input(SHAPE), Spied(), keras.layers.Flatten(),
                              keras.layers.Dense(CLASSES)])
    spied.layers[-1].set_weights(reference.layers[-1].get_weights())
    monkeypatch.setattr(tf, "device", spying_device)
    check(reference_blob, spied, tmp_path)
    assert state["requested"] == ["/CPU:0"]
    assert state["called_inside"] and all(state["called_inside"])


def test_a_file_of_another_model_is_still_refused(reference, tmp_path):
    other = blob_of(build(seed=1))
    with pytest.raises(ConfigError, match="differs from the Keras model by"):
        check(other, reference, tmp_path)


def test_the_boundary_did_not_move(reference, tmp_path):
    """Mutants ``RTOL x10`` and ``check removed``: 3x the allowance is refused, 0.2x is not.

    The model's logits have scale ``s``; the allowance is ``TFLITE_CHECK_RTOL * s``. A file
    whose bias is shifted by 3x the allowance differs by 3x it, one shifted by 0.2x differs by 0.2x.
    """
    assert exporting.TFLITE_CHECK_RTOL == 1e-4
    allowance = exporting.TFLITE_CHECK_RTOL * scale_of(reference)
    check(blob_of(build(bias_shift=0.2 * allowance)), reference, tmp_path)
    with pytest.raises(ConfigError, match="differs from the Keras model by"):
        check(blob_of(build(bias_shift=3.0 * allowance)), reference, tmp_path)


def test_a_file_with_the_wrong_output_width_is_refused(reference, tmp_path):
    wide = keras.Sequential([keras.layers.Input(SHAPE), keras.layers.Flatten(),
                             keras.layers.Dense(CLASSES + 1)])
    with pytest.raises(ConfigError, match="differs from the Keras model"):
        check(blob_of(wide), reference, tmp_path)


def test_a_mixed_float16_model_still_passes_inside_the_cpu_scope(tmp_path):
    """The scope must not break half-precision (measured on the GPU with ``cliffordnet/nano``: compute dtype
    float16 kept, CPU Keras against the interpreter 0.0). Here a ``mixed_float16`` Dense model, on whatever
    device the test machine has: the check passes and its logits are half precision."""
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        model = build()
        assert model.layers[-1].compute_dtype == "float16"
        batch = np.random.default_rng(0).random((2, *SHAPE)).astype(np.float32)
        assert "float16" in str(model(batch, training=False).dtype)
        check(blob_of(model, select=True), model, tmp_path)
        # and it still refuses a file of another model under that policy
        other = build(seed=1)
        with pytest.raises(ConfigError, match="differs from the Keras model by"):
            check(blob_of(other, select=True), model, tmp_path)
    finally:
        keras.mixed_precision.set_global_policy("float32")
