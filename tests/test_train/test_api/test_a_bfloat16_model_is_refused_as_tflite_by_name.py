"""A ``mixed_bfloat16`` model exported as ``tflite`` is a ``ConfigError`` that names the format (review-iter-2 concern 9).

Reproduced on the GPU with XLA auto-jit off and on (``cliffordnet/nano`` trained with
``precision=mixed_bfloat16``): the converter produced a file and the interpreter of the numeric check
raised ``ValueError: Didn't find op for builtin opcode 'CAST' version '7'`` from ``_tflite_logits``, a raw
traceback and exit 1 for what is a request the format cannot honour (exit 2), with ``<out>.saved_model``
left behind. ``mixed_float16`` and ``float32`` export with exit 0.

The chosen answer is a refusal, not an export. Why not an export: the bfloat16 casts live in the traced
graph, the TFLite runtime of the ``tensorflow`` package has no kernel for them at version 7, and the
variables are float32 whichever policy the layers compute in, so "export it in float32" would be a
different model from the one that was trained and checked. Two layers:

* before anything is written, a model whose compute dtype is ``bfloat16`` is refused, naming the format
  and the formats that work (``export`` writes nothing, leaves no ``.saved_model``);
* when the interpreter itself cannot run a converted file for any other missing operator, the
  ``ValueError`` becomes a ``ConfigError`` that says the file was not written and where the SavedModel
  was kept, instead of a traceback.

Tests write only under ``tmp_path``.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from train.api import exporting
from train.api.errors import ConfigError

SHAPE = (4, 4, 3)


def build(policy):
    keras.utils.set_random_seed(0)
    previous = keras.config.dtype_policy().name
    keras.config.set_dtype_policy(policy)
    try:
        return keras.Sequential([keras.layers.Input(SHAPE), keras.layers.Flatten(), keras.layers.Dense(5)])
    finally:
        keras.config.set_dtype_policy(previous)


def test_a_bfloat16_model_is_refused_before_anything_is_written(tmp_path):
    model = build("mixed_bfloat16")
    assert model.layers[-1].compute_dtype == "bfloat16"
    out = tmp_path / "m.tflite"
    with pytest.raises(ConfigError) as caught:
        exporting.export(model=model, out=out, format="tflite", input_shape=SHAPE)
    text = str(caught.value)
    assert "'format'" in text and "tflite" in text and "bfloat16" in text
    assert "keras, saved_model or onnx" in text
    assert list(tmp_path.iterdir()) == [], "a refusal must leave nothing behind (no file, no .saved_model)"


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
def test_the_other_policies_still_export(tmp_path, policy):
    """Control: the refusal is about bfloat16 only (a refusal that hits every model is not a check)."""
    model = build(policy)
    out = tmp_path / "m.tflite"
    assert exporting.export(model=model, out=out, format="tflite", input_shape=SHAPE) == out
    assert out.is_file()


@pytest.mark.parametrize("format", ["keras", "saved_model"])
def test_a_bfloat16_model_still_exports_to_the_formats_that_carry_it(tmp_path, format):
    model = build("mixed_bfloat16")
    out = tmp_path / ("m.keras" if format == "keras" else "m_dir")
    assert exporting.export(model=model, out=out, format=format, input_shape=SHAPE) == out
    assert out.exists()


def test_an_interpreter_that_lacks_an_operator_is_a_config_error_naming_the_format_and_the_kept_saved_model(
    tmp_path, monkeypatch
):
    """The net under the pre-check: whatever else the interpreter cannot run is not a traceback."""
    model = build("float32")

    def lacks_an_operator(blob, batch):
        raise ValueError("Didn't find op for builtin opcode 'CAST' version '7'\nRegistration failed.")

    monkeypatch.setattr(exporting, "_tflite_logits", lacks_an_operator)
    spec = tf.TensorSpec((None, *SHAPE), tf.float32, name="input")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    blob = converter.convert()
    out = tmp_path / "m.tflite"
    with pytest.raises(ConfigError) as caught:
        exporting._check_tflite_against_keras(blob, model, spec, out)
    text = str(caught.value)
    assert "'format'" in text and "tflite" in text and "Didn't find op for builtin opcode 'CAST'" in text
    assert "was not written" in text and str(out) in text
    assert ".saved_model" in text
    assert not out.exists()
