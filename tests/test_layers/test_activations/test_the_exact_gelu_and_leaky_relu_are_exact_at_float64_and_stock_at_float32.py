"""Guard: the repo's GELU and LeakyReLU are exact at float64 and stock at float32.

Why this file exists
--------------------
TensorFlow 2.18 builds two activation constants in float32 even for a float64
input (D-218, re-measured in plan-2026-10-14T062616-81938e81 F-11):

* ``tf.nn.gelu(approximate=False)`` divides by ``cast(1.4142135623730951)``,
  a float32 constant widened, so a float64 GELU is off by up to 5.02e-9.
* ``LeakyRelu`` carries its slope as a float32 op attribute, so a float64
  slope of 0.01 applies as 0.009999999776482582.

``exact_activations.py`` writes both functions out with Python-float constants,
which the backend converts in the input's dtype. Each arm below holds one claim:

* float64: within 1e-12 of a numpy/scipy reference (GELU; the exact form
  measured 8.9e-16) and equal bit for bit (LeakyReLU).
* float32: bit-identical to stock Keras over 300,000 points, for GELU and for
  the slopes 0.01, 0.2 and 0.3. This is what lets the repo route ``"gelu"``
  and ``"leaky_relu"`` through these forms without moving any float32 result.
* the gradient at 0 equals stock's (``where(x > 0, ...)`` keeps the slope at
  0; ``maximum(x, a * x)`` would give 1 there).
* float16 and bfloat16 run and keep their dtype.
* the functions are registered under names that differ from stock, so a stock
  layer that holds one serializes it as a dict and reloads the repo function.
  A function merely named ``gelu`` would serialize to the string ``"gelu"`` and
  reload as stock, a silent revert.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from scipy.special import erf

from dl_techniques.layers.activations.exact_activations import (
    EXACT_ACTIVATIONS,
    ExactLeakyReLU,
    exact_gelu,
    exact_leaky_relu,
)

from .test_the_activation_constants_follow_the_input_dtype import _float32_constants

# ---------------------------------------------------------------------

#: Above float64 noise for |y| <= 8 (a few ulps, about 1e-15) and four
#: orders below TensorFlow's measured 5.02e-9.
_FLOAT64_ATOL = 1e-12

_SLOPES = (0.01, 0.2, 0.3)


def _points_float32() -> np.ndarray:
    """300,000 finite float32 points: a dense grid, a normal cloud, and
    log-uniform magnitudes of both signs from 1e-30 to 1e3, with 0 and -0."""
    rng = np.random.default_rng(0)
    grid = np.linspace(-8.0, 8.0, 100_000)
    cloud = rng.standard_normal(100_000) * 3.0
    wide = rng.choice([-1.0, 1.0], 100_000) * 10.0 ** rng.uniform(-30, 3, 100_000)
    wide[0], wide[1] = 0.0, -0.0
    x = np.concatenate([grid, cloud, wide]).astype(np.float32)
    assert x.size == 300_000 and np.all(np.isfinite(x))
    return x


def _bits(t) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(t), dtype=np.float32).view(np.uint32)


def _grid64() -> np.ndarray:
    return np.linspace(-8.0, 8.0, 40_001)


# ---------------------------------------------------------------------
# float64: exact
# ---------------------------------------------------------------------


def test_exact_gelu_at_float64_is_within_1e12_of_the_erf_reference():
    x = _grid64()
    got = keras.ops.convert_to_numpy(exact_gelu(keras.ops.convert_to_tensor(x, "float64")))
    assert got.dtype == np.float64
    expected = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    np.testing.assert_allclose(got, expected, rtol=0, atol=_FLOAT64_ATOL)


def test_stock_gelu_at_float64_is_not_within_1e12():
    """Control: the same reference and tolerance reject TensorFlow's gelu."""
    x = _grid64()
    got = keras.ops.convert_to_numpy(
        keras.ops.gelu(keras.ops.convert_to_tensor(x, "float64"), approximate=False))
    error = np.abs(got - 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).max()
    assert error > 100 * _FLOAT64_ATOL, error


@pytest.mark.parametrize("slope", _SLOPES)
def test_exact_leaky_relu_at_float64_equals_numpy_exactly(slope):
    x = _grid64()
    expected = np.where(x > 0, x, slope * x)
    fn_out = keras.ops.convert_to_numpy(
        exact_leaky_relu(keras.ops.convert_to_tensor(x, "float64"), negative_slope=slope))
    layer_out = keras.ops.convert_to_numpy(
        ExactLeakyReLU(negative_slope=slope, dtype="float64")(
            keras.ops.convert_to_tensor(x, "float64")))
    assert fn_out.dtype == np.float64 and layer_out.dtype == np.float64
    np.testing.assert_array_equal(fn_out, expected)
    np.testing.assert_array_equal(layer_out, expected)


def test_stock_leaky_relu_at_float64_is_not_exact():
    """Control: stock's float32 slope attribute misses the float64 product."""
    x = _grid64()
    got = keras.ops.convert_to_numpy(
        keras.ops.leaky_relu(keras.ops.convert_to_tensor(x, "float64"), negative_slope=0.01))
    assert np.abs(got - np.where(x > 0, x, 0.01 * x)).max() > 1e-10


@pytest.mark.parametrize("name, fn", [
    ("exact_gelu", exact_gelu),
    ("exact_leaky_relu", lambda x: exact_leaky_relu(x, negative_slope=0.01)),
    ("ExactLeakyReLU", ExactLeakyReLU(negative_slope=0.01, dtype="float64")),
])
def test_no_float32_constant_is_built_at_float64(name, fn):
    """The graph scan that pins TensorFlow's float32 constants finds none here."""
    assert _float32_constants(lambda x: fn(x)) == []


# ---------------------------------------------------------------------
# float32: bit-identical to stock
# ---------------------------------------------------------------------


def test_exact_gelu_at_float32_is_bit_identical_to_stock_over_300k_points():
    x = keras.ops.convert_to_tensor(_points_float32(), "float32")
    ours = _bits(exact_gelu(x))
    for stock in (keras.ops.gelu(x, approximate=False), keras.activations.gelu(x)):
        mismatch = np.flatnonzero(ours != _bits(stock))
        assert mismatch.size == 0, (mismatch.size, mismatch[:5])


@pytest.mark.parametrize("slope", _SLOPES)
def test_exact_leaky_relu_at_float32_is_bit_identical_to_stock_over_300k_points(slope):
    x = keras.ops.convert_to_tensor(_points_float32(), "float32")
    pairs = (
        (exact_leaky_relu(x, negative_slope=slope),
         keras.ops.leaky_relu(x, negative_slope=slope)),
        (ExactLeakyReLU(negative_slope=slope)(x),
         keras.layers.LeakyReLU(negative_slope=slope)(x)),
    )
    for ours, stock in pairs:
        mismatch = np.flatnonzero(_bits(ours) != _bits(stock))
        assert mismatch.size == 0, (slope, mismatch.size, mismatch[:5])


def test_the_default_slopes_are_stocks():
    """The function default is ``keras.activations.leaky_relu``'s (0.2), the
    layer default is ``keras.layers.LeakyReLU``'s (0.3)."""
    x = keras.ops.convert_to_tensor(np.linspace(-3.0, 3.0, 61), "float32")
    np.testing.assert_array_equal(
        keras.ops.convert_to_numpy(exact_leaky_relu(x)),
        keras.ops.convert_to_numpy(keras.activations.leaky_relu(x)))
    assert ExactLeakyReLU().negative_slope == keras.layers.LeakyReLU().negative_slope


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_gradient_at_zero_equals_stocks(dtype):
    x = tf.constant([0.0, -0.0], dtype=dtype)
    cases = [
        (exact_gelu, lambda v: keras.ops.gelu(v, approximate=False)),
    ] + [
        (lambda v, s=s: exact_leaky_relu(v, negative_slope=s),
         lambda v, s=s: keras.ops.leaky_relu(v, negative_slope=s))
        for s in _SLOPES
    ]
    for ours, stock in cases:
        grads = []
        for fn in (ours, stock):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = fn(x)
            grads.append(tape.gradient(y, x).numpy())
        np.testing.assert_allclose(grads[0], grads[1], rtol=1e-7, atol=0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = exact_leaky_relu(x, negative_slope=0.2)
    assert np.all(tape.gradient(y, x).numpy() == np.asarray(0.2, dtype))


# ---------------------------------------------------------------------
# half precision
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dtype, atol", [("float16", 1e-2), ("bfloat16", 6e-2)])
def test_half_precision_inputs_run_and_keep_their_dtype(dtype, atol):
    x64 = np.linspace(-6.0, 6.0, 1201)
    x = keras.ops.cast(keras.ops.convert_to_tensor(x64, "float32"), dtype)
    xr = np.asarray(keras.ops.convert_to_numpy(keras.ops.cast(x, "float32")), np.float64)
    for name, out, expected in (
        ("exact_gelu", exact_gelu(x), 0.5 * xr * (1.0 + erf(xr / np.sqrt(2.0)))),
        ("exact_leaky_relu", exact_leaky_relu(x, negative_slope=0.2),
         np.where(xr > 0, xr, 0.2 * xr)),
        ("ExactLeakyReLU", ExactLeakyReLU(negative_slope=0.2, dtype=dtype)(x),
         np.where(xr > 0, xr, 0.2 * xr)),
    ):
        assert keras.backend.standardize_dtype(out.dtype) == dtype, name
        got = np.asarray(keras.ops.convert_to_numpy(keras.ops.cast(out, "float32")), np.float64)
        assert np.all(np.isfinite(got)), name
        np.testing.assert_allclose(got, expected, rtol=1e-2, atol=atol, err_msg=name)


# ---------------------------------------------------------------------
# registration and round trips
# ---------------------------------------------------------------------


@pytest.mark.parametrize("fn", [exact_gelu, exact_leaky_relu])
def test_a_registered_function_serializes_by_its_own_name_and_reloads_itself(fn):
    cfg = keras.activations.serialize(fn)
    assert isinstance(cfg, dict), (
        f"{fn.__name__} serialized to {cfg!r}: a plain string reloads as stock")
    assert keras.activations.deserialize(cfg) is fn


def test_the_names_differ_from_stock():
    stock_names = {"gelu", "leaky_relu", "LeakyReLU"}
    assert {exact_gelu.__name__, exact_leaky_relu.__name__,
            ExactLeakyReLU.__name__}.isdisjoint(stock_names)
    assert keras.saving.get_registered_object(
        keras.saving.get_registered_name(ExactLeakyReLU)) is ExactLeakyReLU


def test_the_repo_table_maps_the_two_stock_names():
    assert EXACT_ACTIVATIONS == {"gelu": exact_gelu, "leaky_relu": exact_leaky_relu}


def test_stock_layers_holding_the_exact_forms_survive_a_saved_model_round_trip(tmp_path):
    model = keras.Sequential([
        keras.layers.Input((6,)),
        keras.layers.Activation(exact_gelu),
        keras.layers.Dense(6, activation=exact_leaky_relu),
        ExactLeakyReLU(negative_slope=0.1),
    ])
    x = np.random.default_rng(0).standard_normal((3, 6)).astype("float32")
    before = keras.ops.convert_to_numpy(model(x, training=False))
    path = tmp_path / "exact.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    act, dense, leaky = reloaded.layers
    assert act.activation is exact_gelu
    assert dense.activation is exact_leaky_relu
    assert type(leaky) is ExactLeakyReLU and leaky.negative_slope == 0.1
    np.testing.assert_array_equal(
        before, keras.ops.convert_to_numpy(reloaded(x, training=False)))


def test_the_exact_layer_round_trips_its_config():
    layer = ExactLeakyReLU(negative_slope=0.01, dtype="float64", name="leaky")
    clone = ExactLeakyReLU.from_config(layer.get_config())
    assert type(clone) is ExactLeakyReLU
    assert clone.negative_slope == 0.01
    assert clone.dtype_policy.name == "float64"


# ---------------------------------------------------------------------
# the shared resolvers (step 2.2)
# ---------------------------------------------------------------------


def test_both_shared_resolvers_return_the_exact_forms_for_the_stock_names():
    """``"gelu"`` and ``"leaky_relu"`` resolve to the repo forms; every other
    name, and every callable, is resolved exactly as before."""
    from dl_techniques.layers.activations.common import resolve_activation as common_resolve
    from dl_techniques.layers.activations.gelu_tanh import gelu_tanh
    from dl_techniques.layers.activations.gelu_tanh import resolve_activation as tanh_resolve

    for resolve in (common_resolve, tanh_resolve):
        assert resolve("gelu") is exact_gelu
        assert resolve("leaky_relu") is exact_leaky_relu
        assert resolve("relu") is keras.activations.relu
        assert resolve("silu") is keras.activations.silu
        assert resolve(keras.activations.gelu) is keras.activations.gelu
    assert tanh_resolve("gelu_tanh") is gelu_tanh
    assert common_resolve(None) is keras.activations.linear


def test_the_trio_serializes_an_exact_form_back_to_its_stock_name():
    """``common.serialize_activation`` is the inverse of ``common.resolve_activation``
    on the two shadowed names, so a repo layer that stores the resolved callable
    and serializes it in ``get_config`` keeps writing ``"gelu"`` / ``"leaky_relu"``
    (configs do not change shape), and reloading resolves the exact form again."""
    from dl_techniques.layers.activations.common import resolve_activation as common_resolve
    from dl_techniques.layers.activations.common import serialize_activation

    for name, fn in EXACT_ACTIVATIONS.items():
        assert serialize_activation(common_resolve(name)) == name
        assert common_resolve(serialize_activation(fn)) is fn
    assert serialize_activation(keras.activations.relu) == "relu"
    assert serialize_activation("gelu") == "gelu"
