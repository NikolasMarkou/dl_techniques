"""Behavioural census: no float64 forward of a registered model reaches TensorFlow's
float32-constant GELU or LeakyReLU.

Why this file exists
--------------------
TensorFlow 2.18's ``tf.nn.gelu`` divides by ``cast(1.4142135623730951)`` (a float32
constant widened) and ``tf.nn.leaky_relu`` carries its slope as a float32 op
attribute, so a float64 GELU is off by up to 5.02e-9 and a float64 slope of 0.01
applies as 0.009999999776482582 (D-218). The repo's exact forms
(``layers/activations/exact_activations.py``, D-277) never call either function.

A grep cannot find the routes that reach them: a ``"gelu"`` string handed to a stock
``Activation`` or ``Dense``, a resolved callable called by repo code, a stock
``LeakyReLU`` layer. Executing the forwards can. F-11 measured, at the start of
plan-2026-10-14T062616-81938e81 iteration 3, 25 of the 56 registered packages
calling ``gelu(approximate=False)`` at float64 and 2 calling ``LeakyRelu``.

How it works
------------
Every subject of ``precision_arm_subjects.SUBJECTS`` (resolved through
``float64_subject``, the float64 arm's own resolution) is built and called inside
``float64_scope`` while ``tf.nn.gelu`` and ``tf.nn.leaky_relu`` are replaced by a
recorder. The Keras TensorFlow backend looks both up on ``tf.nn`` at call time, so
every route through ``keras.ops``, ``keras.activations`` and the stock layers is
seen. The recorder keeps the first ``src/`` frame of each float64 call; the
expected list is empty.

Named exclusions
----------------
- ``HARNESS_ABORTS``: subjects whose float64 forward raises inside the harness
  for a reason unrelated to activations. Their calls before the raise are still
  recorded and must still be zero, so for them the census is a lower bound.
"""

import os
import traceback
from typing import Dict, List

import keras
import pytest
import tensorflow as tf

from .precision_arm_oracle import _cast_float_leaves, default_call, float64_scope
from .precision_arm_subjects import SUBJECTS, float64_subject

# ---------------------------------------------------------------------

#: The repository's own ``src/`` (``.venv/.../keras/src`` must not match).
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src") + os.sep

#: name -> reason. The forward raises before it finishes; the calls it made
#: before raising are still held to zero.
HARNESS_ABORTS: Dict[str, str] = {
    "blt": "ByteLatentTransformer.call raises InvalidArgumentError at float64 "
           "(LessEqual: an int32 operand meets an int64 one)",
    "lewm": "SIGRegLayer.call raises InvalidArgumentError at float64 "
            "(Mul: a float32 operand meets a float64 one)",
    "video_jepa": "SIGRegLayer.call raises InvalidArgumentError at float64 "
                  "(Mul: a float32 operand meets a float64 one)",
}


class _Recorder:
    """Stand-in for ``tf.nn.gelu`` / ``tf.nn.leaky_relu`` that records float64 calls."""

    def __init__(self, name, original, sink: List[str]):
        self.name, self.original, self.sink = name, original, sink

    def __call__(self, features, *args, **kwargs):
        if getattr(features, "dtype", None) == tf.float64:
            site = next(
                (f"{os.path.relpath(fr.filename)}:{fr.lineno}"
                 for fr in reversed(traceback.extract_stack()[:-1])
                 if os.path.abspath(fr.filename).startswith(_SRC)),
                "<no src frame>",
            )
            self.sink.append(f"{self.name} {site}")
        return self.original(features, *args, **kwargs)


def _float64_calls(name: str, monkeypatch) -> List[str]:
    sink: List[str] = []
    for fn in ("gelu", "leaky_relu"):
        monkeypatch.setattr(tf.nn, fn, _Recorder(fn, getattr(tf.nn, fn), sink))
    build, make_inputs, kwargs = float64_subject(name)
    call = kwargs.get("call_fn") or default_call
    with float64_scope():
        keras.utils.set_random_seed(0)
        model = build()
        inputs = _cast_float_leaves(make_inputs(), "float64")
        if name in HARNESS_ABORTS:
            try:
                call(model, inputs, kwargs.get("training", False))
            except Exception:  # noqa: BLE001 - the abort is the named exclusion
                pass
        else:
            call(model, inputs, kwargs.get("training", False))
    return sink


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_the_float64_forward_makes_no_float32_constant_activation_call(name, monkeypatch):
    calls = _float64_calls(name, monkeypatch)
    assert calls == [], (
        f"{name}: {len(calls)} float64 call(s) reached TensorFlow's float32-constant "
        f"activation; distinct sites: {sorted(set(calls))}. Route them through "
        f"layers/activations/exact_activations.py (D-277).")


def test_every_harness_abort_is_still_one(monkeypatch):
    """A named exclusion that no longer aborts is stale: remove it."""
    stale = []
    for name in HARNESS_ABORTS:
        build, make_inputs, kwargs = float64_subject(name)
        call = kwargs.get("call_fn") or default_call
        with float64_scope():
            keras.utils.set_random_seed(0)
            try:
                call(build(), _cast_float_leaves(make_inputs(), "float64"),
                     kwargs.get("training", False))
            except Exception:  # noqa: BLE001
                continue
        stale.append(name)
    assert stale == [], f"no longer abort at float64, drop from HARNESS_ABORTS: {stale}"


def test_the_recorder_sees_a_stock_float64_gelu(monkeypatch):
    """Positive control: the patched names are the ones the backend calls."""
    sink: List[str] = []
    for fn in ("gelu", "leaky_relu"):
        monkeypatch.setattr(tf.nn, fn, _Recorder(fn, getattr(tf.nn, fn), sink))
    x = keras.ops.convert_to_tensor([[-1.0, 0.5]], dtype="float64")
    keras.ops.gelu(x, approximate=False)
    keras.layers.Activation("gelu", dtype="float64")(x)
    keras.layers.LeakyReLU(dtype="float64")(x)
    assert [c.split()[0] for c in sink] == ["gelu", "gelu", "leaky_relu"], sink
