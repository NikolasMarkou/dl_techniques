"""Guard: an activation's constants are built in the input's dtype, not float32.

Why this file exists
--------------------
plan-2026-10-14T062616-81938e81 step 7 measured that, at float64, TensorFlow's
``tf.nn.gelu`` divides by ``cast(1.4142135623730951)`` (a float32 constant cast
up, off by up to 5.0e-9) and ``LeakyRelu`` carries its slope as a float32 op
attribute. Step 7.1 (D-218) traced every activation this repository owns for
the same defect with a graph scan (below) over every activation the package
exports, and found these repo-owned sites:

* ``SaturatedMish`` computed its plateau ``mish(alpha)`` in numpy float32, so a
  float64 layer saturated toward a float32 number: output off by 6.7e-8.
* ``gelu_tanh`` delegated to ``keras.ops.gelu(approximate=True)``, whose
  ``0.044715`` is a float32 constant cast up: output off by 9.0e-10.
* ``HardSigmoid`` and ``HardSwish`` built their ReLU6 sub-layer on the default
  float32 policy, so a float64 layer returned float32 (``HardSwish`` raised on
  ``float64 * float32`` under ``tf.function``).
* ``ProbabilityOutput`` built all seven strategy layers that way.
* The ``thresh_max`` function built a fresh float32 ``ThreshMax`` per call,
  which also created a variable inside any ``tf.function`` that called it.

The constants are Python floats now, which Keras converts in the input's
dtype, the sub-layers take the parent's dtype policy, and ``thresh_max`` runs
the shared computation without a layer. The TensorFlow-internal sites stay as
they are (D-218) and are pinned below as positive controls of the scan: if
TensorFlow ever builds them in float64, the control goes red and the record is
stale.

How the scan works
------------------
Each function is traced under ``tf.function`` with a float64 input, and every
constant in the graph is inspected. A float32 constant, or a float64 constant
whose value is exactly a float32 number with more than 12 significant bits,
is a constant computed in float32; a ``LeakyRelu`` node carries a float32
slope. Small dyadic values (``2.0``, ``0.5``) are exact in both dtypes and are
not flagged.
"""

import math

import keras
import numpy as np
import pytest
import tensorflow as tf
from scipy.special import erf

from dl_techniques.layers.activations.factory import (
    ACTIVATION_REGISTRY,
    create_activation_layer,
)
from dl_techniques.layers.activations.gelu_tanh import gelu_tanh
from dl_techniques.layers.activations.mish import SaturatedMish
from dl_techniques.layers.activations.probability_output import ProbabilityOutput
from dl_techniques.layers.activations.sigsoftmax import log_sigsoftmax, sigsoftmax
from dl_techniques.layers.activations.soft_value_range import soft_value_range
from dl_techniques.layers.activations.thresh_max import thresh_max

#: Required constructor arguments for the registry keys that have any.
_REQUIRED = {"output_dim": 4, "min_value": -1.0}

#: Above float64 noise for these magnitudes (|y| <= 6, a few ulps ~1e-15) and
#: far below either measured defect (6.7e-8, 9.0e-10).
_FLOAT64_ATOL = 1e-12


def _significant_bits(value: float) -> int:
    if value == 0 or not math.isfinite(value):
        return 0
    mantissa, _ = math.frexp(abs(value))
    bits = int(mantissa * (1 << 53))
    trailing = (bits & -bits).bit_length() - 1
    return 53 - trailing


def _float32_constants(fn, shape=(4, 8)):
    """Names and values of the constants ``fn`` builds in float32 at float64."""
    graph = tf.function(fn).get_concrete_function(
        tf.TensorSpec(shape, tf.float64)
    ).graph
    found = []
    for op in graph.get_operations():
        if op.type == "LeakyRelu":
            found.append((op.name, op.get_attr("alpha")))
        if op.type != "Const":
            continue
        value = tf.make_ndarray(op.get_attr("value"))
        if value.dtype == np.float32:
            flagged = [float(v) for v in value.ravel() if _significant_bits(float(v)) > 12]
        elif value.dtype == np.float64:
            flagged = [
                float(v) for v in value.ravel()
                if np.isfinite(v) and v != 0 and float(np.float32(v)) == float(v)
                and _significant_bits(float(v)) > 12
            ]
        else:
            continue
        if flagged:
            found.append((op.name, flagged[0]))
    return found


def _registry_layer(key):
    info = ACTIVATION_REGISTRY[key]
    extra = {name: _REQUIRED[name] for name in info["required_params"]}
    return create_activation_layer(key, dtype="float64", **extra)


#: ``ProbabilityOutput`` strategies (one per delegate class) and the
#: constructor arguments each needs.
_PROBABILITY_TYPES = {
    "softmax": {},
    "sparsemax": {},
    "threshmax": {},
    "adaptive": {},
    "harmax": {},
    "routing": {},
    "hierarchical": {"output_dim": 4},
}


def _subjects():
    """Every activation the package exports, as ``name -> builder of fn(x)``."""
    subjects = {
        f"registry:{key}": (lambda key=key: _registry_layer(key))
        for key in ACTIVATION_REGISTRY
    }
    for kind, config in _PROBABILITY_TYPES.items():
        subjects[f"ProbabilityOutput:{kind}"] = (
            lambda kind=kind, config=config: ProbabilityOutput(
                probability_type=kind, type_config=dict(config), dtype="float64"
            )
        )
    subjects["fn:gelu_tanh"] = lambda: gelu_tanh
    subjects["fn:sigsoftmax"] = lambda: sigsoftmax
    subjects["fn:log_sigsoftmax"] = lambda: log_sigsoftmax
    subjects["fn:soft_value_range"] = lambda: (lambda x: soft_value_range(x, -1.0, 1.0))
    subjects["fn:thresh_max"] = lambda: thresh_max
    return subjects


SUBJECTS = _subjects()


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_no_activation_builds_a_float32_constant_at_float64(name):
    fn = SUBJECTS[name]()
    assert _float32_constants(lambda x: fn(x)) == []


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_a_float64_input_comes_back_float64(name):
    """A sub-layer left on the float32 policy narrows the input and returns float32.

    ``HardSigmoid``/``HardSwish`` built their ReLU6 without the parent's dtype
    (``HardSwish`` then raised on ``float64 * float32``), ``ProbabilityOutput``
    built every strategy layer that way, and the ``thresh_max`` function built
    a float32 ``ThreshMax`` (D-218).
    """
    fn = SUBJECTS[name]()
    x = np.random.default_rng(0).normal(size=(4, 8))
    out = fn(keras.ops.convert_to_tensor(x, "float64"))
    assert keras.ops.convert_to_numpy(out).dtype == np.float64


@pytest.mark.parametrize("name, fn, value", [
    ("tf.nn.gelu exact", lambda x: keras.ops.gelu(x, approximate=False),
     float(np.float32(np.sqrt(2.0)))),
    ("tf.nn.gelu tanh", lambda x: keras.ops.gelu(x, approximate=True),
     float(np.float32(0.044715))),
    ("LeakyRelu", lambda x: keras.ops.leaky_relu(x, negative_slope=0.01),
     float(np.float32(0.01))),
])
def test_the_scan_still_sees_the_tensorflow_float32_constants(name, fn, value):
    """Positive control, and the D-218 record: TensorFlow builds these in float32."""
    found = _float32_constants(fn)
    assert [v for _, v in found] == [value], (name, found)


def _mish(x):
    return x * np.tanh(np.log1p(np.exp(x)))


def test_saturated_mish_at_float64_equals_its_float64_reference():
    alpha, beta = 3.0, 0.5
    x = np.linspace(-6.0, 12.0, 3001)
    blend = 1.0 / (1.0 + np.exp(-(x - alpha) / beta))
    expected = _mish(x) * (1.0 - blend) + _mish(alpha) * blend
    layer = SaturatedMish(alpha=alpha, beta=beta, dtype="float64")
    got = keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x, "float64")))
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, rtol=0, atol=_FLOAT64_ATOL)
    assert layer.mish_at_alpha == pytest.approx(_mish(alpha), rel=0, abs=1e-15)


def test_gelu_tanh_at_float64_equals_its_float64_reference():
    x = np.linspace(-6.0, 6.0, 2001)
    expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))
    got = keras.ops.convert_to_numpy(gelu_tanh(keras.ops.convert_to_tensor(x, "float64")))
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, rtol=0, atol=_FLOAT64_ATOL)


def test_gelu_tanh_at_float32_is_still_the_backend_expression():
    """The rewrite changes float64 only: at float32 it is the same op sequence."""
    x = keras.ops.convert_to_tensor(np.linspace(-6.0, 6.0, 2001), "float32")
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(gelu_tanh(x)),
        keras.ops.convert_to_numpy(keras.ops.gelu(x, approximate=True)),
    )


def test_the_exact_gelu_reference_is_what_the_record_says():
    """D-218's measured TensorFlow exact-GELU error at float64, re-derived."""
    x = np.linspace(-6.0, 6.0, 2001)
    got = keras.ops.convert_to_numpy(
        keras.ops.gelu(keras.ops.convert_to_tensor(x, "float64"), approximate=False)
    )
    error = np.abs(got - 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).max()
    assert 1e-9 < error < 1e-8, error
