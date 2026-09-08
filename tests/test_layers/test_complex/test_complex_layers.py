"""
Tests for Complex-Valued Neural Network Layers
============================================

This module provides comprehensive tests for complex-valued neural network layers,
including initialization tests, shape verification, and numerical correctness checks.
"""

import keras
import pytest
import tempfile
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional

from dl_techniques.layers.complex.base import ComplexLayer
from dl_techniques.layers.complex.complex_conv2d import ComplexConv2D
from dl_techniques.layers.complex.complex_dense import ComplexDense
from dl_techniques.layers.complex.complex_relu import ComplexReLU
from dl_techniques.layers.complex.complex_average_pooling2d import (
    ComplexAveragePooling2D,
)
from dl_techniques.layers.complex.complex_dropout import ComplexDropout
from dl_techniques.layers.complex.complex_global_average_pooling2d import (
    ComplexGlobalAveragePooling2D,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def random_complex_input() -> tf.Tensor:
    """Create random complex input tensor."""
    real = tf.random.normal((8, 32, 32, 3))
    imag = tf.random.normal((8, 32, 32, 3))
    return tf.complex(real, imag)


@pytest.fixture
def random_complex_dense_input() -> tf.Tensor:
    """Create random complex input tensor for dense layer."""
    real = tf.random.normal((8, 128))
    imag = tf.random.normal((8, 128))
    return tf.complex(real, imag)


@dataclass
class ComplexModelConfig:
    """Configuration for complex model testing.

    Args:
        batch_size: Number of samples per batch
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        conv_filters: Number of filters in conv layer
        kernel_size: Size of conv kernel
        dense_units: Number of units in dense layer
        kernel_regularizer: Regularization factor for kernel
        kernel_initializer: Initializer for kernel weights
    """
    batch_size: int = 32
    input_shape: Tuple[int, ...] = (32, 32, 3)
    num_classes: int = 10
    learning_rate: float = 0.0001
    num_epochs: int = 10
    conv_filters: int = 32
    kernel_size: int = 3
    dense_units: int = 10
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None
    kernel_initializer: str = 'glorot_uniform'

# ---------------------------------------------------------------------
# Base Layer Tests
# ---------------------------------------------------------------------

class TestComplexLayer:
    """Tests for base ComplexLayer functionality."""

    def test_init_complex_weights(self) -> None:
        """Test complex weight initialization."""
        layer = ComplexLayer()
        shape = (3, 3, 64, 32)
        weights = layer._init_complex_weights(shape)

        # Check shape and dtype
        assert weights.shape == shape
        assert weights.dtype == tf.complex64

        # Check statistical properties
        magnitudes = tf.abs(weights)
        phases = tf.math.angle(weights)

        # Magnitude should follow Rayleigh distribution
        assert tf.reduce_mean(magnitudes) > 0
        assert tf.math.reduce_std(magnitudes) < 1.0

        # Phases should be uniformly distributed in [-π, π]
        assert tf.reduce_min(phases) >= -np.pi
        assert tf.reduce_max(phases) <= np.pi

    def test_epsilon_handling(self) -> None:
        """Test epsilon parameter handling."""
        custom_epsilon = 1e-5
        layer = ComplexLayer(epsilon=custom_epsilon)
        assert layer.epsilon == custom_epsilon

    def test_regularizer_attachment(self) -> None:
        """Test kernel regularizer attachment."""
        regularizer = tf.keras.regularizers.L2(0.01)
        layer = ComplexLayer(kernel_regularizer=regularizer)
        assert layer.kernel_regularizer == regularizer


# ---------------------------------------------------------------------
# Complex Convolution Tests
# ---------------------------------------------------------------------

class TestComplexConv2D:
    """Tests for ComplexConv2D layer."""

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = ComplexConv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding='SAME'
        )
        assert layer.filters == 32
        assert layer.kernel_size == (3, 3)
        assert layer.strides == (1, 1)
        assert layer.padding == 'SAME'

    def test_build(self, random_complex_input: tf.Tensor) -> None:
        """Test layer building and weight creation."""
        layer = ComplexConv2D(filters=32, kernel_size=3)
        layer.build(random_complex_input.shape)

        # Check kernel shape
        assert layer.kernel.shape == (3, 3, 3, 32)
        assert layer.bias.shape == (32,)

        # Check dtypes
        assert layer.kernel.dtype == tf.complex64
        assert layer.bias.dtype == tf.complex64

    def test_forward_pass(self, random_complex_input: tf.Tensor) -> None:
        """Test forward pass computation."""
        layer = ComplexConv2D(filters=32, kernel_size=3)
        output = layer(random_complex_input)

        # Check output shape
        assert output.shape == (8, 32, 32, 32)
        assert output.dtype == tf.complex64

        # Check numerical properties
        assert not tf.reduce_any(tf.math.is_nan(tf.abs(output)))
        assert not tf.reduce_any(tf.math.is_inf(tf.abs(output)))

    def test_padding_modes(self, random_complex_input: tf.Tensor) -> None:
        """Test different padding modes."""
        # Test 'SAME' padding
        layer_same = ComplexConv2D(filters=32, kernel_size=3, padding='SAME')
        output_same = layer_same(random_complex_input)
        assert output_same.shape[1:3] == random_complex_input.shape[1:3]

        # Test 'VALID' padding
        layer_valid = ComplexConv2D(filters=32, kernel_size=3, padding='VALID')
        output_valid = layer_valid(random_complex_input)
        expected_shape = (
            random_complex_input.shape[0],  # batch
            random_complex_input.shape[1] - 2,  # height
            random_complex_input.shape[2] - 2,  # width
            32  # filters
        )
        assert output_valid.shape == expected_shape

    def test_strided_convolution(self, random_complex_input: tf.Tensor) -> None:
        """Test strided convolution."""
        # Test with strides=2
        layer = ComplexConv2D(filters=32, kernel_size=3, strides=2)
        output = layer(random_complex_input)

        expected_shape = (
            random_complex_input.shape[0],  # batch
            random_complex_input.shape[1] // 2,  # height
            random_complex_input.shape[2] // 2,  # width
            32  # filters
        )
        assert output.shape == expected_shape


# ---------------------------------------------------------------------
# Complex Dense Tests
# ---------------------------------------------------------------------

class TestComplexDense:
    """Tests for ComplexDense layer."""

    def test_initialization(self) -> None:
        """Test layer initialization."""
        layer = ComplexDense(units=64)
        assert layer.units == 64

    def test_build(self, random_complex_dense_input: tf.Tensor) -> None:
        """Test layer building and weight creation."""
        layer = ComplexDense(units=64)
        layer.build(random_complex_dense_input.shape)

        # Check shapes
        assert layer.kernel.shape == (128, 64)
        assert layer.bias.shape == (64,)

        # Check dtypes
        assert layer.kernel.dtype == tf.complex64
        assert layer.bias.dtype == tf.complex64

    def test_forward_pass(self, random_complex_dense_input: tf.Tensor) -> None:
        """Test forward pass computation."""
        layer = ComplexDense(units=64)
        output = layer(random_complex_dense_input)

        # Check output properties
        assert output.shape == (8, 64)
        assert output.dtype == tf.complex64
        assert not tf.reduce_any(tf.math.is_nan(tf.abs(output)))
        assert not tf.reduce_any(tf.math.is_inf(tf.abs(output)))

    def test_weight_gradients(self, random_complex_dense_input: tf.Tensor) -> None:
        """Test gradient computation."""
        layer = ComplexDense(units=64)
        with tf.GradientTape() as tape:
            output = layer(random_complex_dense_input)
            loss = tf.reduce_mean(tf.abs(output))

        # Compute gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradient properties
        for grad in grads:
            assert grad is not None
            assert not tf.reduce_any(tf.math.is_nan(tf.abs(grad)))
            assert not tf.reduce_any(tf.math.is_inf(tf.abs(grad)))


# ---------------------------------------------------------------------
# Complex ReLU Tests
# ---------------------------------------------------------------------

class TestComplexReLU:
    """Tests for ComplexReLU activation."""

    def test_forward_pass(self, random_complex_input: tf.Tensor) -> None:
        """Test forward pass computation."""
        layer = ComplexReLU()
        output = layer(random_complex_input)

        # Check shape preservation
        assert output.shape == random_complex_input.shape
        assert output.dtype == tf.complex64

        # Check ReLU properties
        real_part = tf.math.real(output)
        imag_part = tf.math.imag(output)
        assert tf.reduce_all(real_part >= 0)
        assert tf.reduce_all(imag_part >= 0)

    def test_zero_input(self) -> None:
        """Test behavior with zero input."""
        layer = ComplexReLU()
        zero_input = tf.zeros((8, 32, 32, 3), dtype=tf.complex64)
        output = layer(zero_input)
        assert tf.reduce_all(tf.abs(output) == 0)

    def test_negative_input(self) -> None:
        """Test behavior with negative input."""
        layer = ComplexReLU()
        negative_input = tf.complex(
            -tf.ones((8, 32, 32, 3)),
            -tf.ones((8, 32, 32, 3))
        )
        output = layer(negative_input)
        assert tf.reduce_all(tf.abs(output) == 0)

    def test_gradient_flow(self) -> None:
        """Test gradient flow through activation."""
        layer = ComplexReLU()
        input_tensor = tf.complex(
            tf.random.normal((8, 32, 32, 3)),
            tf.random.normal((8, 32, 32, 3))
        )

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            output = layer(input_tensor)
            loss = tf.reduce_mean(tf.abs(output))

        gradient = tape.gradient(loss, input_tensor)
        assert gradient is not None
        assert not tf.reduce_any(tf.math.is_nan(tf.abs(gradient)))


# ---------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------

def test_layer_composition() -> None:
    """Test composition of complex layers."""
    # Create input
    input_tensor = tf.complex(
        tf.random.normal((8, 32, 32, 3)),
        tf.random.normal((8, 32, 32, 3))
    )

    # Create layer stack
    conv1 = ComplexConv2D(32, 3)
    relu1 = ComplexReLU()
    conv2 = ComplexConv2D(64, 3)
    relu2 = ComplexReLU()

    # Forward pass
    x = conv1(input_tensor)
    x = relu1(x)
    x = conv2(x)
    output = relu2(x)

    # Check final output
    assert output.shape == (8, 32, 32, 64)
    assert output.dtype == tf.complex64
    assert not tf.reduce_any(tf.math.is_nan(tf.abs(output)))
    assert not tf.reduce_any(tf.math.is_inf(tf.abs(output)))


def test_numerical_stability() -> None:
    """Test numerical stability with extreme values."""
    # Create input with large values
    large_input = tf.complex(
        1e5 * tf.random.normal((8, 32, 32, 3)),
        1e5 * tf.random.normal((8, 32, 32, 3))
    )

    # Test conv layer
    conv = ComplexConv2D(32, 3)
    conv_out = conv(large_input)
    assert not tf.reduce_any(tf.math.is_nan(tf.abs(conv_out)))
    assert not tf.reduce_any(tf.math.is_inf(tf.abs(conv_out)))

    # Test dense layer with scaled input
    dense = ComplexDense(64)
    dense_input = tf.reshape(large_input, (8, -1))[:, :128]  # Take first 128 features
    dense_out = dense(dense_input)
    assert not tf.reduce_any(tf.math.is_nan(tf.abs(dense_out)))
    assert not tf.reduce_any(tf.math.is_inf(tf.abs(dense_out)))


def create_complex_model(config: ComplexModelConfig) -> keras.Model:
    """Create a model with complex layers.

    Args:
        config: Model configuration parameters

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        ComplexConv2D(
            filters=config.conv_filters,
            kernel_size=config.kernel_size,
            kernel_regularizer=config.kernel_regularizer,
            kernel_initializer=config.kernel_initializer,
        ),
        ComplexReLU(),
        keras.layers.Flatten(),
        ComplexDense(
            units=config.dense_units,
            kernel_regularizer=config.kernel_regularizer,
            kernel_initializer=config.kernel_initializer
        ),
        # Final layer to get real outputs
        keras.layers.Lambda(lambda x: tf.abs(x))
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model



# ---------------------------------------------------------------------
# The `kernel_initializer` dead knob — pinned, per
# plans/plan-2026-09-08T070501-528ded1a/decisions.md D-002
# ---------------------------------------------------------------------

# The AST guards below parse EVERY module of `layers/complex/`, not just the one
# holding `ComplexLayer`. Both knobs are inherited by all six leaf classes, so a
# guard scoped to `base.py` alone is blind to a read added in a leaf -- MEASURED
# at `d148888a7`: `self.kernel_initializer((2, 2))` inside `ComplexDense.call`
# left this suite at 159 passed and the dead-knob suite at 10 passed. The module
# set is enumerated from the package's own `__path__`, never hardcoded, so an
# eighth module is covered the day it lands.
from dl_techniques.layers.complex import base as _complex_layer_module
from tests.complex_dead_knob_ast import (
    BASE_MODULE_NAME,
    EXPECTED_BASE_SITES,
    assert_scan_reaches_the_leaves,
    describe_site_counts,
    expected_dead_knob_sites,
    self_attribute_site_counts,
    self_attribute_sites,
)

_COMPLEX_LAYER_MODULE_NAME = "base.py"


def test_kernel_initializer_is_read_by_exactly_two_ast_nodes_and_neither_computes():
    """The mechanism, asserted rather than described.

    `self.kernel_initializer` appears at exactly two places in the AST of the
    module holding `ComplexLayer` — the assignment in `__init__` and the entry in
    `get_config` — and at ZERO places in each of the six leaf modules that
    inherit it. A site anywhere else means the knob has acquired a consumer and
    D-002's pin-as-documented-dead ruling must be revisited.

    The predicate is AST, deliberately: the DECISION comment placed at the site
    names the attribute, so a `source.count("self.kernel_initializer")` cannot
    tell a consumer from a comment about the absence of consumers. This mirrors
    the `epsilon` guard in
    `tests/test_models/test_the_two_documented_dead_knobs.py`.
    """
    counts = self_attribute_site_counts("kernel_initializer")
    expected = expected_dead_knob_sites()
    assert counts == expected, (
        f"`self.kernel_initializer` site counts across layers/complex/ are "
        f"[{describe_site_counts(counts)}], expected "
        f"[{describe_site_counts(expected)}] — {EXPECTED_BASE_SITES} in "
        f"{BASE_MODULE_NAME}.py (the __init__ assignment and the get_config "
        "entry) and 0 in every leaf. A new site means the knob is no longer "
        "inert and D-002 must be re-decided, not patched."
    )
    # In base.py one is a Store (the assignment), one is a Load (the get_config
    # read). A second Load there is a computation reading the knob.
    contexts = sorted(
        type(node.ctx).__name__
        for node in self_attribute_sites(_complex_layer_module, "kernel_initializer")
    )
    assert contexts == ["Load", "Store"], (
        f"expected one Store and one Load in {_COMPLEX_LAYER_MODULE_NAME}, got "
        f"{contexts} — a second Load is a computation reading the knob"
    )


def test_epsilon_is_still_exactly_two_ast_nodes_across_the_whole_package():
    """Invariant 2 re-checked here, where `kernel_initializer` is edited.

    The two knobs share one assignment block; an edit to one is exactly the kind
    of change that could add a site to the other. Scoped to the whole package for
    the same reason as the guard above.
    """
    counts = self_attribute_site_counts("epsilon")
    expected = expected_dead_knob_sites()
    assert counts == expected, (
        f"`self.epsilon` site counts across layers/complex/ are "
        f"[{describe_site_counts(counts)}], expected "
        f"[{describe_site_counts(expected)}]."
    )


def test_the_ast_scan_reaches_the_leaf_modules_and_can_see_a_live_attribute():
    """LIVENESS for the two guards above — shared with the dead-knob suite.

    The assertion itself lives in `tests/complex_dead_knob_ast.py` because both
    guards depend on it and a copy in each file would be a hand-maintained
    lockstep invariant.
    """
    assert_scan_reaches_the_leaves()


class _SpyInitializer(keras.initializers.Initializer):
    """An initializer that records its own invocations and returns a sentinel.

    RNG-independent by construction. A same-process `max|delta kernel|` probe
    across two initializers CANNOT serve as this oracle: constructing an
    `Initializer` object itself consumes global RNG state, so the glorot-vs-glorot
    control reads ~1.66 too and the assertion would be RED at HEAD for a reason
    unrelated to the knob (MEASURED, decisions.md D-002 § carried correction).
    """

    SENTINEL = 7.0

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def __call__(self, shape, dtype=None):
        self.calls.append(tuple(shape))
        return keras.ops.full(shape, self.SENTINEL, dtype=dtype or "float32")


@pytest.mark.parametrize(
    "factory,build_shape",
    [
        (lambda init: ComplexDense(units=4, kernel_initializer=init), (2, 8)),
        (lambda init: ComplexConv2D(filters=4, kernel_size=3,
                                    kernel_initializer=init), (2, 8, 8, 3)),
    ],
    ids=["ComplexDense", "ComplexConv2D"],
)
def test_kernel_initializer_is_never_invoked_during_build(factory, build_shape):
    """The spy oracle: the knob's own `__call__` never runs, and its value never lands.

    `_init_complex_weights` hardcodes a Rayleigh-magnitude / uniform-phase draw,
    so the passed initializer is dead. This guard goes RED the instant anyone
    makes `_init_complex_weights` call `self.kernel_initializer`.
    """
    spy = _SpyInitializer()
    layer = factory(spy)
    layer.build(build_shape)

    assert spy.calls == [], (
        f"the kernel_initializer was invoked {len(spy.calls)} time(s) during "
        f"build() with shapes {spy.calls} — the knob is no longer dead, so the "
        "D-002 pin-as-documented-dead ruling is stale and must be re-decided"
    )
    # Subordinate belt, not an independent oracle: any wire-up that lands the
    # spy's value in the kernel must first CALL the spy, so this assertion
    # cannot be shown RED while the invocation-count assertion above is green.
    # It is kept because it states the consequence the count is a proxy for.
    kernel = keras.ops.convert_to_numpy(layer.kernel)
    assert not np.allclose(np.real(kernel), _SpyInitializer.SENTINEL), (
        "the kernel carries the spy's sentinel value — the initializer's output "
        "reached the weights"
    )


# ---------------------------------------------------------------------
# `compute_output_shape` pinned against the REAL forward pass
# (plan-2026-09-08T070501-528ded1a Step 3; grid measured at Step 1, decisions.md D-001)
# ---------------------------------------------------------------------

# The full Step-1(a) grid, promoted from a throwaway probe into a real guard:
# {ComplexConv2D, ComplexAveragePooling2D} x padding{SAME,VALID} x stride{1,2,3}
# x input spatial{7,8,9} x kernel|pool{2,3} = 72 cells. The oracle is the shape
# of a REAL forward pass on a complex64 input, never a re-derived formula --
# a formula-vs-formula test would have agreed with the defect it is here to catch.

_SHAPE_GRID = [
    (cls_name, padding, stride, size, k)
    for cls_name in ("ComplexConv2D", "ComplexAveragePooling2D")
    for padding in ("SAME", "VALID")
    for stride in (1, 2, 3)
    for size in (7, 8, 9)
    for k in (2, 3)
]


def _make_shape_grid_layer(cls_name: str, padding: str, stride: int, k: int):
    """Build the layer for one grid cell (unbuilt)."""
    if cls_name == "ComplexConv2D":
        return ComplexConv2D(filters=4, kernel_size=k, strides=stride, padding=padding)
    return ComplexAveragePooling2D(pool_size=k, strides=stride, padding=padding)


def _shape_grid_id(cell) -> str:
    cls_name, padding, stride, size, k = cell
    return f"{cls_name}-{padding}-stride{stride}-in{size}-k{k}"


@pytest.mark.parametrize(
    "cls_name,padding,stride,size,k",
    _SHAPE_GRID,
    ids=[_shape_grid_id(cell) for cell in _SHAPE_GRID],
)
def test_compute_output_shape_agrees_with_forward_pass(cls_name, padding, stride, size, k):
    """`compute_output_shape` must equal `tuple(forward_output.shape)`, built AND unbuilt.

    Guide 3.4 requires `compute_output_shape` to answer from stored config alone,
    so the unbuilt instance is a separate object that is never built or called.

    MEASURED RED at HEAD in exactly the 8 cells where `ComplexConv2D` is under
    `SAME` padding and the stride does not divide the input evenly (independent
    of `k`): `compute_output_shape` floored where `keras.ops.conv(padding='same')`
    ceils. See decisions.md D-001 for the full 72-cell reading.
    """
    inputs = tf.complex(
        tf.random.normal((1, size, size, 3)),
        tf.random.normal((1, size, size, 3)),
    )
    input_shape = tuple(inputs.shape)

    # Unbuilt: answered from stored config only, before any build() or call().
    unbuilt = _make_shape_grid_layer(cls_name, padding, stride, k)
    unbuilt_shape = tuple(unbuilt.compute_output_shape(input_shape))

    built = _make_shape_grid_layer(cls_name, padding, stride, k)
    outputs = built(inputs)
    forward_shape = tuple(outputs.shape)
    built_shape = tuple(built.compute_output_shape(input_shape))

    assert built_shape == forward_shape, (
        f"{cls_name} padding={padding} strides={stride} input={input_shape} k={k}: "
        f"compute_output_shape returned {built_shape} but the real forward pass "
        f"produced {forward_shape}"
    )
    assert unbuilt_shape == forward_shape, (
        f"{cls_name} padding={padding} strides={stride} input={input_shape} k={k}: "
        f"an UNBUILT layer's compute_output_shape returned {unbuilt_shape} but the "
        f"real forward pass produced {forward_shape} (guide 3.4: the answer must "
        "come from stored config alone)"
    )


@pytest.mark.parametrize(
    "cls_name,padding,stride,k",
    [
        (cls_name, padding, stride, k)
        for cls_name in ("ComplexConv2D", "ComplexAveragePooling2D")
        for padding in ("SAME", "VALID")
        for stride in (1, 2, 3)
        for k in (2, 3)
    ],
    ids=[
        f"{cls_name}-{padding}-stride{stride}-k{k}"
        for cls_name in ("ComplexConv2D", "ComplexAveragePooling2D")
        for padding in ("SAME", "VALID")
        for stride in (1, 2, 3)
        for k in (2, 3)
    ],
)
def test_compute_output_shape_propagates_none_dimensions(cls_name, padding, stride, k):
    """Undefined spatial dims and an undefined batch must propagate as `None`, not raise.

    A functional model built on `keras.Input` hands exactly this shape in, so a
    raise here is a graph-construction failure rather than a wrong number.
    """
    layer = _make_shape_grid_layer(cls_name, padding, stride, k)
    output_shape = tuple(layer.compute_output_shape((None, None, None, 3)))

    expected_channels = 4 if cls_name == "ComplexConv2D" else 3
    assert output_shape == (None, None, None, expected_channels), (
        f"{cls_name} padding={padding} strides={stride} k={k}: expected "
        f"(None, None, None, {expected_channels}) from a fully undefined spatial "
        f"input, got {output_shape}"
    )


# ---------------------------------------------------------------------
# The `get_config` / `from_config` pair on `ComplexLayer` --- plan Step 4
# ---------------------------------------------------------------------


def _build_complex_dense_model(regularizer) -> keras.Model:
    """A minimal saveable model whose single complex layer is a `ComplexDense`."""
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(6,), dtype="complex64"),
        ComplexDense(units=4, kernel_regularizer=regularizer),
    ])
    return model


def _first_complex_dense(model: keras.Model) -> ComplexDense:
    for layer in model.layers:
        if isinstance(layer, ComplexDense):
            return layer
    raise AssertionError("no ComplexDense found in the model")


def test_from_config_restores_regularizer_and_initializer_objects(tmp_path):
    """A `.keras` round trip must give back OBJECTS, not their serialized dicts.

    `get_config` writes `kernel_regularizer` / `kernel_initializer` through
    `keras.regularizers.serialize` / `keras.initializers.serialize`, so without a
    matching `from_config` the reloaded layer stores the raw config dict (Keras
    wraps it as a `TrackedDict`). Nothing crashes --- the attribute is simply the
    wrong TYPE, and anything that later calls the regularizer or asks it for a
    config fails far from here. Hence the oracle is `isinstance`, not a crash.

    The `name` / `trainable` assertions are the guide 6.2 base-key guard: a
    `from_config` that pops a base key silently renames the layer and drops its
    trainability on every reload.
    """
    model = _build_complex_dense_model(keras.regularizers.L2(0.01))
    model(tf.complex(tf.random.normal((2, 6)), tf.random.normal((2, 6))))
    original = _first_complex_dense(model)

    path = tmp_path / "complex_dense_roundtrip.keras"
    model.save(path)
    loaded_layer = _first_complex_dense(keras.models.load_model(path))

    assert isinstance(loaded_layer.kernel_regularizer, keras.regularizers.Regularizer), (
        "after a .keras round trip `kernel_regularizer` is "
        f"{type(loaded_layer.kernel_regularizer).__name__} "
        f"({loaded_layer.kernel_regularizer!r}), not a keras Regularizer -- "
        "ComplexLayer.from_config did not deserialize it"
    )
    assert isinstance(loaded_layer.kernel_initializer, keras.initializers.Initializer), (
        "after a .keras round trip `kernel_initializer` is "
        f"{type(loaded_layer.kernel_initializer).__name__} "
        f"({loaded_layer.kernel_initializer!r}), not a keras Initializer -- "
        "ComplexLayer.from_config did not deserialize it"
    )

    assert loaded_layer.name == original.name, (
        f"the reloaded layer is named {loaded_layer.name!r} but the original was "
        f"{original.name!r} -- from_config popped the base `name` key "
        "(the guide 6.2 defect)"
    )
    assert loaded_layer.trainable == original.trainable, (
        f"the reloaded layer has trainable={loaded_layer.trainable} but the "
        f"original had {original.trainable} -- from_config popped the base "
        "`trainable` key (the guide 6.2 defect)"
    )


def test_from_config_accepts_already_deserialized_objects():
    """`from_config` is also reached with live objects, not only with dicts.

    `keras.models.clone_model` and any hand-written `Cls.from_config(layer.get_config())`
    after an in-memory `deserialize_keras_object` pass hand back real objects, so
    this pins `from_config`'s accepted input domain.

    Honest scope: this is a CONTRACT test, not a defect guard. Removing the
    `isinstance(..., dict)` branches in `from_config` leaves it GREEN, because
    MEASURED at keras 3.8 both `deserialize` helpers are idempotent on a live
    object. It goes RED only if a future change makes `from_config` assume a
    dict (e.g. indexing `config["kernel_regularizer"]["class_name"]`).
    """
    config = ComplexDense(units=4, kernel_regularizer=keras.regularizers.L2(0.01)).get_config()
    config["kernel_regularizer"] = keras.regularizers.L2(0.02)
    config["kernel_initializer"] = keras.initializers.HeNormal()

    rebuilt = ComplexDense.from_config(config)

    assert isinstance(rebuilt.kernel_regularizer, keras.regularizers.L2)
    assert isinstance(rebuilt.kernel_initializer, keras.initializers.HeNormal)


def test_from_config_does_not_consume_the_caller_config():
    """`from_config` must not mutate the dict it is handed.

    Keras reuses a config dict across `clone_model` passes; deserializing in place
    turns the second call's input into objects the first call already consumed.
    """
    config = ComplexDense(units=4, kernel_regularizer=keras.regularizers.L2(0.01)).get_config()
    before = dict(config)

    ComplexDense.from_config(config)

    assert config == before, (
        "from_config mutated the caller's config dict; it must work on a copy"
    )
    assert isinstance(config["kernel_regularizer"], dict)


# ---------------------------------------------------------------------
# ComplexDropout sub-layer compliance (guide 3.2)
# ---------------------------------------------------------------------

def test_complex_dropout_names_its_sublayer_explicitly_in_every_instance():
    """Both instances in one process must name their inner Dropout ``"dropout"``.

    Guide 3.2: "Always give sub-layers explicit names, including inside loops.
    Auto-generated names shift when depth changes, and checkpoints stop matching."

    The TWO-instance shape is the whole guard. Keras auto-names per process, so
    without ``name="dropout"`` the first instance's inner layer is still called
    ``dropout`` and a single-instance assertion passes at HEAD -- a guard that
    cannot fail. The SECOND instance is what goes ``dropout_1`` when the explicit
    name is removed, and that is the assertion with power.
    """
    first = ComplexDropout(0.3)
    second = ComplexDropout(0.5)

    assert first.dropout_layer.name == "dropout", (
        f"the first ComplexDropout named its inner Dropout "
        f"{first.dropout_layer.name!r}, not 'dropout'"
    )
    assert second.dropout_layer.name == "dropout", (
        f"the SECOND ComplexDropout in this process named its inner Dropout "
        f"{second.dropout_layer.name!r}, not 'dropout' -- the explicit name= was "
        f"dropped and Keras auto-numbering took over, which is exactly what "
        f"breaks checkpoint name matching (guide 3.2)"
    )


def test_complex_dropout_builds_its_sublayer_in_build():
    """An explicit ``build()`` must materialize the sub-layer tree.

    Guide 1.2's table: ``build`` = "CREATE this layer's weights; MATERIALIZE the
    sub-layer tree", and "ALWAYS in build: build each sub-layer that call() will
    run -- and only those". At HEAD ``ComplexDropout`` defines no ``build()``, so
    nothing builds the inner Dropout before the first ``call()``.

    The oracle is a SPY on ``dropout_layer.build``, not ``dropout_layer.built``.
    MEASURED at keras 3.8: ``keras.layers.Dropout`` defines no ``build`` of its
    own, so ``Layer.__init__`` marks it ``built=True`` at construction --
    ``assert layer.dropout_layer.built is True`` passes at HEAD and is exactly the
    guard-that-cannot-fail this suite refuses to ship. Whether ``build()`` reaches
    the sub-layer is observable only by watching the call.
    """
    layer = ComplexDropout(0.25)
    calls: List[Any] = []
    inner_build = layer.dropout_layer.build

    def spy_build(input_shape):
        calls.append(input_shape)
        return inner_build(input_shape)

    object.__setattr__(layer.dropout_layer, "build", spy_build)

    layer.build((None, 16))

    assert layer.built is True
    assert calls == [(None, 16)], (
        f"ComplexDropout.build() invoked its inner Dropout's build() with {calls} "
        "-- expected exactly one call carrying the input shape. The sub-layer tree "
        "is being materialized lazily inside call() instead (guide 1.2/3.2)."
    )


def test_complex_dropout_round_trips_through_a_keras_model(tmp_path):
    """Adding ``build()`` must not shift a saved model's weight layout or ordering.

    ``build()`` changes WHEN the inner Dropout is materialized, so the falsification
    signal for plan Step 5 is a reload that fails or a weight list whose names or
    order moved. The inner Dropout owns no weights, so the pinned list is
    ``ComplexDense``'s and it must be identical before and after the reload.
    """
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(6,), dtype="complex64"),
        ComplexDense(units=4),
        ComplexDropout(0.3),
    ])
    x = tf.complex(tf.random.normal((2, 6)), tf.random.normal((2, 6)))
    before = model(x, training=False).numpy()
    before_weights = [(w.name, tuple(w.shape)) for w in model.weights]

    path = tmp_path / "complex_dropout_roundtrip.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    after = loaded(x, training=False).numpy()
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)

    assert [(w.name, tuple(w.shape)) for w in loaded.weights] == before_weights, (
        "the reloaded model's weight layout/order differs from the original -- "
        "materializing the ComplexDropout sub-layer in build() shifted it"
    )

    reloaded_dropout = [
        layer for layer in loaded.layers if isinstance(layer, ComplexDropout)
    ]
    assert len(reloaded_dropout) == 1
    assert reloaded_dropout[0].dropout_layer.name == "dropout"



# ---------------------------------------------------------------------
# ComplexAveragePooling2D -- first direct coverage (plan Step 6)
# ---------------------------------------------------------------------

def _complex_from_parts(real: np.ndarray, imag: np.ndarray) -> tf.Tensor:
    """Build a complex64 tensor from two float arrays of the same shape."""
    return tf.complex(
        tf.constant(real, dtype=tf.float32),
        tf.constant(imag, dtype=tf.float32),
    )


def test_complex_average_pooling_forward_values_match_a_hand_computed_reference():
    """Pool a known 4x4 map and compare against arithmetic written out by hand.

    Guide 16.3 forbids a shape-only oracle. The real part is ``arange(16)`` and
    the imaginary part is ``100 - arange(16)``, so the two components carry
    DIFFERENT numbers -- a forward that swapped them, or that pooled one part
    twice, cannot pass.

    2x2 / stride-2 / VALID windows over ``arange(16).reshape(4, 4)``::

        [ 0  1 | 2  3]      (0+1+4+5)/4  = 2.5    (2+3+6+7)/4   = 4.5
        [ 4  5 | 6  7]
        ---------------
        [ 8  9 |10 11]      (8+9+12+13)/4= 10.5   (10+11+14+15)/4 = 12.5
        [12 13 |14 15]
    """
    real = np.arange(16, dtype="float32").reshape(1, 4, 4, 1)
    imag = (100.0 - np.arange(16, dtype="float32")).reshape(1, 4, 4, 1)
    x = _complex_from_parts(real, imag)

    layer = ComplexAveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='VALID')
    y = layer(x).numpy()

    expected_real = np.array([[2.5, 4.5], [10.5, 12.5]], dtype="float32").reshape(1, 2, 2, 1)
    # imag = 100 - real elementwise, and the mean is affine, so each window mean
    # is 100 minus the corresponding real window mean.
    expected_imag = np.array([[97.5, 95.5], [89.5, 87.5]], dtype="float32").reshape(1, 2, 2, 1)

    assert y.shape == (1, 2, 2, 1)
    np.testing.assert_allclose(y.real, expected_real, rtol=0, atol=1e-6)
    np.testing.assert_allclose(y.imag, expected_imag, rtol=0, atol=1e-6)


def test_complex_average_pooling_pools_the_two_components_independently():
    """The imaginary part must be pooled on its own, not derived from the real one.

    The real part is constant (every window mean is 5.0) while the imaginary
    part varies, so a forward that pooled the real part and reused the result
    for both components -- or that pooled ``|z|`` -- produces a constant
    imaginary output and fails here, while the previous test could not tell.
    """
    real = np.full((1, 4, 4, 1), 5.0, dtype="float32")
    imag = np.arange(16, dtype="float32").reshape(1, 4, 4, 1)
    x = _complex_from_parts(real, imag)

    y = ComplexAveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x).numpy()

    np.testing.assert_allclose(
        y.real, np.full((1, 2, 2, 1), 5.0, dtype="float32"), rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        y.imag,
        np.array([[2.5, 4.5], [10.5, 12.5]], dtype="float32").reshape(1, 2, 2, 1),
        rtol=0,
        atol=1e-6,
    )


def test_complex_average_pooling_valid_and_same_branches_have_different_values():
    """Both padding branches, with the SAME edge arithmetic written out.

    MEASURED at keras 3.8 / TF 2.18: ``average_pool(padding='same')`` excludes the
    implicit padding from the denominator (``count_include_pad=False``), so an
    edge window divides by however many REAL elements it saw, not by 4. Over
    ``arange(9).reshape(3, 3)`` with a 2x2 / stride-2 window::

        VALID -> one full window: (0+1+3+4)/4 = 2.0
        SAME  -> [[ (0+1+3+4)/4 , (2+5)/2 ],
                  [ (6+7)/2     , (8)/1   ]]  =  [[2.0, 3.5], [6.5, 8.0]]

    A branch that padded with zeros AND counted them would read
    [[2.0, 1.75], [3.25, 2.0]] instead, so this test also pins the denominator.
    """
    real = np.arange(9, dtype="float32").reshape(1, 3, 3, 1)
    imag = np.zeros((1, 3, 3, 1), dtype="float32")
    x = _complex_from_parts(real, imag)

    y_valid = ComplexAveragePooling2D(pool_size=2, strides=2, padding='VALID')(x).numpy()
    assert y_valid.shape == (1, 1, 1, 1)
    np.testing.assert_allclose(y_valid.real.squeeze(), 2.0, rtol=0, atol=1e-6)

    y_same = ComplexAveragePooling2D(pool_size=2, strides=2, padding='SAME')(x).numpy()
    assert y_same.shape == (1, 2, 2, 1)
    np.testing.assert_allclose(
        y_same.real.squeeze(),
        np.array([[2.0, 3.5], [6.5, 8.0]], dtype="float32"),
        rtol=0,
        atol=1e-6,
    )


def test_complex_average_pooling_rejects_a_padding_that_is_not_same_or_valid():
    """The constructor's own validation branch."""
    with pytest.raises(ValueError, match="padding must be 'SAME' or 'VALID'"):
        ComplexAveragePooling2D(padding='causal')


def test_complex_average_pooling_compute_output_shape_rejects_non_4d():
    """``compute_output_shape`` must refuse a rank it cannot pool."""
    layer = ComplexAveragePooling2D()
    for bad_shape in [(8, 32), (8, 32, 3), (8, 4, 4, 4, 3)]:
        with pytest.raises(ValueError, match="requires 4D input"):
            layer.compute_output_shape(bad_shape)


def test_complex_average_pooling_round_trips_through_a_saved_model(tmp_path):
    """A real ``.keras`` save/load must preserve config and values.

    The complex64 ``InputLayer`` idiom is used deliberately: a
    ``keras.layers.Lambda`` wrapper does not survive safe-mode deserialization.
    """
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(4, 4, 2), dtype="complex64"),
        ComplexAveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='SAME'),
    ])
    x = tf.complex(tf.random.normal((3, 4, 4, 2)), tf.random.normal((3, 4, 4, 2)))
    before = model(x, training=False).numpy()

    path = tmp_path / "complex_avgpool_roundtrip.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    after = loaded(x, training=False).numpy()
    np.testing.assert_allclose(before.real, after.real, rtol=0, atol=1e-6)
    np.testing.assert_allclose(before.imag, after.imag, rtol=0, atol=1e-6)

    reloaded_layer = [
        layer for layer in loaded.layers if isinstance(layer, ComplexAveragePooling2D)
    ]
    assert len(reloaded_layer) == 1
    assert reloaded_layer[0].pool_size == (2, 2)
    assert reloaded_layer[0].strides == (2, 2)
    assert reloaded_layer[0].padding == 'SAME'


# ---------------------------------------------------------------------
# ComplexDropout -- first direct coverage (plan Step 6)
# ---------------------------------------------------------------------

def test_complex_dropout_is_exactly_the_identity_at_inference():
    """``training=False`` must return the input BIT-for-bit, not merely close.

    Dropout's inference path applies no mask and no rescale, so anything other
    than exact equality means a scale factor leaked into the inference branch.
    """
    real = np.random.RandomState(0).randn(4, 32).astype("float32")
    imag = np.random.RandomState(1).randn(4, 32).astype("float32")
    x = _complex_from_parts(real, imag)

    y = ComplexDropout(0.5)(x, training=False).numpy()

    assert np.array_equal(y.real, real), "the inference path altered the real part"
    assert np.array_equal(y.imag, imag), "the inference path altered the imaginary part"


def test_complex_dropout_drops_and_rescales_the_survivors_at_training():
    """Every training-mode output is either exactly 0 or exactly ``z / (1 - rate)``.

    With ``rate=0.5`` the inverted-dropout scale is exactly 2.0. Asserting the
    two-valued ratio pins BOTH halves of the operation: a forward that dropped
    without rescaling fails, and one that rescaled without dropping fails too
    (the drop fraction is checked against the rate).
    """
    keras.utils.set_random_seed(1234)
    real = np.random.RandomState(2).randn(64, 64).astype("float32") + 3.0
    imag = np.random.RandomState(3).randn(64, 64).astype("float32") + 5.0
    x = _complex_from_parts(real, imag)

    y = ComplexDropout(0.5)(x, training=True).numpy()

    dropped = y.real == 0.0
    kept = ~dropped

    np.testing.assert_allclose(y.real[dropped], 0.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(y.imag[dropped], 0.0, rtol=0, atol=1e-7)
    np.testing.assert_allclose(y.real[kept], real[kept] * 2.0, rtol=0, atol=1e-5)
    np.testing.assert_allclose(y.imag[kept], imag[kept] * 2.0, rtol=0, atol=1e-5)

    drop_fraction = dropped.mean()
    assert 0.4 < drop_fraction < 0.6, (
        f"{drop_fraction:.3f} of the units were dropped at rate=0.5 -- the mask is "
        "not being drawn at the configured rate"
    )


def test_complex_dropout_kills_real_and_imaginary_parts_together():
    """The reason this class exists: ONE real mask governs both components.

    Two independent per-component masks would leave, at rate=0.5 over 4096 units,
    roughly a quarter of them with a live real part and a dead imaginary part --
    a complex number whose phase was destroyed by the regulariser. The oracle is
    the elementwise ratio: for a shared mask ``out.real / in.real`` and
    ``out.imag / in.imag`` are the SAME real number at every position.
    """
    keras.utils.set_random_seed(4321)
    rng = np.random.RandomState(7)
    # Every component is bounded away from zero, so a zero in the output can only
    # come from the mask and the ratio below is always well defined.
    real = rng.uniform(1.0, 2.0, size=(64, 64)).astype("float32")
    imag = rng.uniform(3.0, 4.0, size=(64, 64)).astype("float32")
    x = _complex_from_parts(real, imag)

    y = ComplexDropout(0.5)(x, training=True).numpy()

    real_ratio = y.real / real
    imag_ratio = y.imag / imag

    np.testing.assert_allclose(real_ratio, imag_ratio, rtol=0, atol=1e-5)

    real_dead = y.real == 0.0
    imag_dead = y.imag == 0.0
    mismatched = int(np.count_nonzero(real_dead != imag_dead))
    assert mismatched == 0, (
        f"{mismatched} of {real.size} units had exactly one component zeroed -- "
        "the real and imaginary parts are being masked INDEPENDENTLY, which "
        "destroys the phase and is precisely what ComplexDropout exists to avoid"
    )
    # Guard the guard: the mask must actually have killed something, or the two
    # ratios above would agree trivially at 2.0 everywhere.
    assert real_dead.any(), "nothing was dropped, so the drop-together oracle saw no mask"


@pytest.mark.parametrize("bad_rate", [-0.1, 1.0, 1.5])
def test_complex_dropout_rejects_a_rate_outside_the_unit_interval(bad_rate):
    """``rate`` must be in ``[0, 1)`` -- 1.0 would divide by zero when rescaling."""
    with pytest.raises(ValueError, match=r"rate must be in the interval \[0, 1\)"):
        ComplexDropout(bad_rate)


def test_complex_dropout_round_trips_with_its_rate_preserved(tmp_path):
    """A saved ``ComplexDropout`` must come back with the same ``rate``."""
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(8,), dtype="complex64"),
        ComplexDropout(0.35),
    ])
    x = tf.complex(tf.random.normal((2, 8)), tf.random.normal((2, 8)))
    before = model(x, training=False).numpy()

    path = tmp_path / "complex_dropout_rate_roundtrip.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    after = loaded(x, training=False).numpy()
    assert np.array_equal(before, after)

    reloaded = [layer for layer in loaded.layers if isinstance(layer, ComplexDropout)]
    assert len(reloaded) == 1
    assert reloaded[0].rate == 0.35


# ---------------------------------------------------------------------
# ComplexGlobalAveragePooling2D -- first direct coverage (plan Step 6)
# ---------------------------------------------------------------------

def _gap_reference_input() -> Tuple[tf.Tensor, np.ndarray, np.ndarray]:
    """A (2, 2, 2, 3) input whose per-channel spatial means are written out below.

    ``real = arange(24).reshape(2, 2, 2, 3)``. For sample 0 channel c the four
    spatial entries are ``c, c+3, c+6, c+9``, so the mean is ``c + 4.5``; sample 1
    is the same block shifted by 12, so its mean is ``c + 16.5``.
    """
    real = np.arange(24, dtype="float32").reshape(2, 2, 2, 3)
    imag = (100.0 - np.arange(24, dtype="float32")).reshape(2, 2, 2, 3)
    return _complex_from_parts(real, imag), real, imag


def test_complex_global_average_pooling_forward_values_match_a_hand_computed_mean():
    """The mean is over axes [1, 2] -- not [1], not [2], not [1, 2, 3]."""
    x, _, _ = _gap_reference_input()

    y = ComplexGlobalAveragePooling2D(keepdims=False)(x).numpy()

    expected_real = np.array([[4.5, 5.5, 6.5], [16.5, 17.5, 18.5]], dtype="float32")
    # imag = 100 - real elementwise and the mean is affine.
    expected_imag = 100.0 - expected_real

    assert y.shape == (2, 3)
    np.testing.assert_allclose(y.real, expected_real, rtol=0, atol=1e-6)
    np.testing.assert_allclose(y.imag, expected_imag, rtol=0, atol=1e-6)


def test_complex_global_average_pooling_keepdims_true_keeps_the_spatial_axes():
    """``keepdims=True`` must give (B, 1, 1, C) carrying the identical numbers."""
    x, _, _ = _gap_reference_input()

    y = ComplexGlobalAveragePooling2D(keepdims=True)(x).numpy()

    expected_real = np.array(
        [[4.5, 5.5, 6.5], [16.5, 17.5, 18.5]], dtype="float32"
    ).reshape(2, 1, 1, 3)

    assert y.shape == (2, 1, 1, 3)
    np.testing.assert_allclose(y.real, expected_real, rtol=0, atol=1e-6)
    np.testing.assert_allclose(y.imag, 100.0 - expected_real, rtol=0, atol=1e-6)


@pytest.mark.parametrize("keepdims", [False, True])
def test_complex_global_average_pooling_compute_output_shape_matches_the_forward(keepdims):
    """Both ``keepdims`` branches pinned against the real forward output.

    Asserted on an UNBUILT layer as well, because guide 3.4 requires
    ``compute_output_shape`` to work from stored config alone.
    """
    x, _, _ = _gap_reference_input()

    unbuilt = ComplexGlobalAveragePooling2D(keepdims=keepdims)
    predicted_unbuilt = unbuilt.compute_output_shape((2, 2, 2, 3))

    layer = ComplexGlobalAveragePooling2D(keepdims=keepdims)
    y = layer(x)

    assert tuple(predicted_unbuilt) == tuple(y.shape)
    assert tuple(layer.compute_output_shape((2, 2, 2, 3))) == tuple(y.shape)
    assert tuple(layer.compute_output_shape((None, 2, 2, 3)))[0] is None


def test_complex_global_average_pooling_compute_output_shape_rejects_non_4d():
    """``compute_output_shape`` must refuse a rank that has no [1, 2] to reduce."""
    layer = ComplexGlobalAveragePooling2D()
    for bad_shape in [(8, 32), (8, 32, 3), (8, 4, 4, 4, 3)]:
        with pytest.raises(ValueError, match="requires 4D input"):
            layer.compute_output_shape(bad_shape)


@pytest.mark.parametrize("keepdims", [False, True])
def test_complex_global_average_pooling_round_trips_through_a_saved_model(tmp_path, keepdims):
    """A real ``.keras`` save/load must preserve ``keepdims`` and the values."""
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(4, 4, 3), dtype="complex64"),
        ComplexGlobalAveragePooling2D(keepdims=keepdims),
    ])
    x = tf.complex(tf.random.normal((2, 4, 4, 3)), tf.random.normal((2, 4, 4, 3)))
    before = model(x, training=False).numpy()

    path = tmp_path / f"complex_gap_roundtrip_{keepdims}.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    after = loaded(x, training=False).numpy()
    np.testing.assert_allclose(before.real, after.real, rtol=0, atol=1e-6)
    np.testing.assert_allclose(before.imag, after.imag, rtol=0, atol=1e-6)

    reloaded = [
        layer for layer in loaded.layers
        if isinstance(layer, ComplexGlobalAveragePooling2D)
    ]
    assert len(reloaded) == 1
    assert reloaded[0].keepdims is keepdims


# ---------------------------------------------------------------------
# Sequence-typed config keys must be round-trip-closed (plan Step 6.1)
# ---------------------------------------------------------------------
#
# `get_config()` emits a tuple; the `.keras` archive stores it as a JSON LIST;
# `__init__` receives that list back. A normalization written as
# `x if isinstance(x, tuple) else (x, x)` is therefore NOT closed under a round
# trip -- it re-wraps `[3, 3]` into `([3, 3], [3, 3])`. These guards assert the
# reloaded layer's sequence attributes are tuples of ints, not merely that the
# model loads.

@pytest.mark.parametrize(
    "kernel_size,strides,expected_kernel,expected_strides",
    [
        (3, None, (3, 3), (1, 1)),          # int form
        ((3, 3), None, (3, 3), (1, 1)),     # tuple form
        (3, (2, 2), (3, 3), (2, 2)),        # explicit strides
    ],
    ids=["kernel_int", "kernel_tuple", "explicit_strides"],
)
def test_complex_conv2d_round_trips_its_sequence_config_keys(
    tmp_path, kernel_size, strides, expected_kernel, expected_strides
):
    """A saved ``ComplexConv2D`` must reload with tuple-of-int shape config."""
    kwargs = {"filters": 4, "kernel_size": kernel_size, "padding": "SAME"}
    if strides is not None:
        kwargs["strides"] = strides

    model = keras.Sequential([
        keras.layers.InputLayer(shape=(8, 8, 2), dtype="complex64"),
        ComplexConv2D(**kwargs),
    ])
    x = tf.complex(tf.random.normal((2, 8, 8, 2)), tf.random.normal((2, 8, 8, 2)))
    before = model(x, training=False).numpy()

    path = tmp_path / f"complex_conv_roundtrip_{kernel_size}_{strides}.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    after = loaded(x, training=False).numpy()
    np.testing.assert_allclose(before.real, after.real, rtol=0, atol=1e-6)
    np.testing.assert_allclose(before.imag, after.imag, rtol=0, atol=1e-6)

    reloaded = [
        layer for layer in loaded.layers if isinstance(layer, ComplexConv2D)
    ]
    assert len(reloaded) == 1
    layer = reloaded[0]
    assert layer.kernel_size == expected_kernel, (
        f"after a .keras round trip kernel_size is {layer.kernel_size!r} "
        f"(type {type(layer.kernel_size).__name__}), expected {expected_kernel!r}"
    )
    assert layer.strides == expected_strides, (
        f"after a .keras round trip strides is {layer.strides!r} "
        f"(type {type(layer.strides).__name__}), expected {expected_strides!r}"
    )
    assert all(isinstance(v, int) for v in layer.kernel_size)
    assert all(isinstance(v, int) for v in layer.strides)


@pytest.mark.parametrize(
    "pool_size,strides,expected_pool,expected_strides",
    [
        (2, None, (2, 2), (2, 2)),          # int form, strides default to pool_size
        ((2, 2), None, (2, 2), (2, 2)),     # tuple form
        (2, (1, 1), (2, 2), (1, 1)),        # explicit strides
    ],
    ids=["pool_int", "pool_tuple", "explicit_strides"],
)
def test_complex_average_pooling_round_trips_its_sequence_config_keys(
    tmp_path, pool_size, strides, expected_pool, expected_strides
):
    """The same closure property for ``ComplexAveragePooling2D``."""
    kwargs = {"pool_size": pool_size, "padding": "SAME"}
    if strides is not None:
        kwargs["strides"] = strides

    model = keras.Sequential([
        keras.layers.InputLayer(shape=(4, 4, 2), dtype="complex64"),
        ComplexAveragePooling2D(**kwargs),
    ])
    x = tf.complex(tf.random.normal((2, 4, 4, 2)), tf.random.normal((2, 4, 4, 2)))
    before = model(x, training=False).numpy()

    path = tmp_path / f"complex_avgpool_seqcfg_{pool_size}_{strides}.keras"
    model.save(path)
    loaded = keras.models.load_model(path)

    after = loaded(x, training=False).numpy()
    np.testing.assert_allclose(before.real, after.real, rtol=0, atol=1e-6)
    np.testing.assert_allclose(before.imag, after.imag, rtol=0, atol=1e-6)

    reloaded = [
        layer for layer in loaded.layers
        if isinstance(layer, ComplexAveragePooling2D)
    ]
    assert len(reloaded) == 1
    layer = reloaded[0]
    assert layer.pool_size == expected_pool, (
        f"after a .keras round trip pool_size is {layer.pool_size!r} "
        f"(type {type(layer.pool_size).__name__}), expected {expected_pool!r}"
    )
    assert layer.strides == expected_strides, (
        f"after a .keras round trip strides is {layer.strides!r} "
        f"(type {type(layer.strides).__name__}), expected {expected_strides!r}"
    )
    assert all(isinstance(v, int) for v in layer.pool_size)
    assert all(isinstance(v, int) for v in layer.strides)


@pytest.mark.parametrize(
    "cls,kwargs,attrs",
    [
        (ComplexConv2D, {"filters": 4, "kernel_size": [3, 3], "strides": [2, 2]},
         {"kernel_size": (3, 3), "strides": (2, 2)}),
        (ComplexAveragePooling2D, {"pool_size": [2, 2], "strides": [2, 2]},
         {"pool_size": (2, 2), "strides": (2, 2)}),
    ],
    ids=["conv2d", "average_pooling2d"],
)
def test_sequence_config_normalization_is_closed_over_lists(cls, kwargs, attrs):
    """``from_config`` receives LISTS from JSON; normalization must accept them.

    This is the unit-level statement of the same defect the two save/load
    guards above exercise end to end: it needs no filesystem and pins the
    normalization itself rather than its downstream symptom.
    """
    layer = cls(**kwargs)
    for name, expected in attrs.items():
        actual = getattr(layer, name)
        assert actual == expected, (
            f"{cls.__name__}.{name} normalized the list {kwargs[name]!r} to "
            f"{actual!r}; a list from a JSON config must coerce to {expected!r}"
        )
        assert isinstance(actual, tuple)


# ---------------------------------------------------------------------
# Registration keys (the one-class-per-module split)
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "cls,expected_key",
    [
        (ComplexLayer,
         "dl_techniques.layers.complex.base>ComplexLayer"),
        (ComplexConv2D,
         "dl_techniques.layers.complex.complex_conv2d>ComplexConv2D"),
        (ComplexDense,
         "dl_techniques.layers.complex.complex_dense>ComplexDense"),
        (ComplexReLU,
         "dl_techniques.layers.complex.complex_relu>ComplexReLU"),
        (ComplexAveragePooling2D,
         "dl_techniques.layers.complex.complex_average_pooling2d>ComplexAveragePooling2D"),
        (ComplexDropout,
         "dl_techniques.layers.complex.complex_dropout>ComplexDropout"),
        (ComplexGlobalAveragePooling2D,
         "dl_techniques.layers.complex.complex_global_average_pooling2d>"
         "ComplexGlobalAveragePooling2D"),
    ],
    ids=[
        "base", "conv2d", "dense", "relu",
        "average_pooling2d", "dropout", "global_average_pooling2d",
    ],
)
def test_registration_key_is_the_modules_own_dotted_path(cls, expected_key):
    """Pin all seven keys by literal ``==`` on the full registered name.

    A save/load round trip can never validate a ``package=`` string: the write
    and the read share one in-process registry, so a mistyped key is
    self-consistent and loads green. Only a literal comparison against the
    module's own dotted path can see it. These keys were rewritten by the
    one-class-per-module split with no legacy alias for the single shared
    key that the now-deleted multi-class module claimed for all seven.
    """
    assert keras.saving.get_registered_name(cls) == expected_key


# ---------------------------------------------------------------------
# The complex product itself, pinned to hand-computed values (plan Step 6.2)
# ---------------------------------------------------------------------
#
# The four-real-product expansion is the entire reason `ComplexConv2D` and
# `ComplexDense` exist, and it had NO value oracle: MEASURED at `d148888a7`,
# inverting the sign in the real branch (`I_r*K_r - I_i*K_i` -> `+`) of EITHER
# class left this suite at 159 passed. `test_forward_pass` checks shape, dtype
# and NaN/Inf and nothing else, and shape is blind to the algebra.
#
# The oracle is arithmetic written out by hand below, never a re-derivation of
# the same four-product form in numpy -- a formula-vs-formula test agrees with
# the sign it is here to catch. `rtol=0` throughout: `assert_allclose`'s default
# `rtol=1e-7` would otherwise contribute a silent second tolerance.

_PRODUCT_ATOL = 1e-6


def _assert_complex_allclose(actual, expected_real, expected_imag, message: str) -> None:
    """Both components of a complex tensor against hand-computed reals."""
    actual = np.asarray(keras.ops.convert_to_numpy(actual))
    np.testing.assert_allclose(
        actual.real, expected_real, rtol=0, atol=_PRODUCT_ATOL,
        err_msg=f"{message} — REAL part",
    )
    np.testing.assert_allclose(
        actual.imag, expected_imag, rtol=0, atol=_PRODUCT_ATOL,
        err_msg=f"{message} — IMAGINARY part",
    )


def _assign_complex(variable, value) -> None:
    """Overwrite a complex64 weight with an explicit literal."""
    variable.assign(tf.constant(np.asarray(value, dtype=np.complex64)))


@pytest.mark.parametrize(
    "bias,expected_real,expected_imag",
    [
        (0 + 0j, 2.0, 14.0),
        (1 - 1j, 3.0, 13.0),
    ],
    ids=["zero_bias", "nonzero_bias"],
)
def test_complex_dense_computes_the_hand_computed_complex_product(
    bias, expected_real, expected_imag
):
    """`z @ w` for a 2-in / 1-out kernel, every number written out.

        z = [1+2i, 2-1i]        w = [3+4i, 2+3i]^T

        (1+2i)(3+4i) = (1*3 - 2*4) + i(1*4 + 2*3) = -5 + 10i
        (2-1i)(2+3i) = (2*2 - -1*3) + i(2*3 + -1*2) =  7 +  4i
                                                sum =  2 + 14i

    The real part is the discriminating one: with the subtraction inverted to
    `I_r*W_r + I_i*W_i` the two terms read `3+8 = 11` and `4-3 = 1`, i.e. `12`
    rather than `2`. The imaginary part is asserted for the mirror-image flip of
    the `+` in the imaginary branch (`10` and `4` become `-2` and `-8`).
    """
    layer = ComplexDense(units=1)
    layer.build((1, 2))
    _assign_complex(layer.kernel, [[3 + 4j], [2 + 3j]])
    _assign_complex(layer.bias, [bias])

    inputs = tf.constant(np.array([[1 + 2j, 2 - 1j]], dtype=np.complex64))
    outputs = layer(inputs)

    assert outputs.dtype == tf.complex64
    assert tuple(outputs.shape) == (1, 1)
    _assert_complex_allclose(
        outputs, [[expected_real]], [[expected_imag]],
        f"ComplexDense with bias {bias} did not compute z @ w",
    )


def test_complex_dense_matches_an_independent_numpy_complex_matmul():
    """Breadth arm: many entries, oracle = numpy's own complex `@`.

    Numpy implements the product in ITS OWN complex dtype, not as four real
    matmuls, so this is an independent implementation rather than a restatement
    of `call`. It exists to catch component transpositions the 2x1 cell above is
    too small to separate; the hand-computed test is still the primary oracle.
    """
    rng = np.random.RandomState(0)
    kernel = (rng.randn(4, 3) + 1j * rng.randn(4, 3)).astype(np.complex64)
    bias = (rng.randn(3) + 1j * rng.randn(3)).astype(np.complex64)
    x = (rng.randn(5, 4) + 1j * rng.randn(5, 4)).astype(np.complex64)

    layer = ComplexDense(units=3)
    layer.build((5, 4))
    _assign_complex(layer.kernel, kernel)
    _assign_complex(layer.bias, bias)

    expected = x @ kernel + bias
    _assert_complex_allclose(
        layer(tf.constant(x)), expected.real, expected.imag,
        "ComplexDense disagrees with numpy's complex matmul",
    )


def test_complex_conv2d_computes_the_hand_computed_complex_product():
    """A 1x1 kernel over 2 input channels — the same arithmetic as the dense cell.

        z = [1+2i, 2-1i] (two channels of one pixel)
        k = [3+4i, 2+3i]

        (1+2i)(3+4i) = -5 + 10i
        (2-1i)(2+3i) =  7 +  4i
                 sum =  2 + 14i

    With the real branch's subtraction inverted the sum reads `12`, not `2`.
    The bias is assigned to exactly zero so the expected value is the product
    alone.
    """
    layer = ComplexConv2D(filters=1, kernel_size=1, strides=1, padding="VALID")
    layer.build((1, 1, 1, 2))
    _assign_complex(layer.kernel, [[[[3 + 4j], [2 + 3j]]]])   # (1, 1, 2, 1)
    _assign_complex(layer.bias, [0 + 0j])

    inputs = tf.constant(np.array([[[[1 + 2j, 2 - 1j]]]], dtype=np.complex64))
    outputs = layer(inputs)

    assert outputs.dtype == tf.complex64
    assert tuple(outputs.shape) == (1, 1, 1, 1)
    _assert_complex_allclose(
        outputs, [[[[2.0]]]], [[[[14.0]]]],
        "ComplexConv2D did not compute the 1x1 complex product",
    )


def test_complex_conv2d_slides_a_two_tap_kernel_over_hand_computed_values():
    """A 1x2 kernel over a 1x3 map: the sliding sum, written out per position.

        z = [1+0i, 0+1i, 2+2i]      k = [1+1i, 2-1i]

    `keras.ops.conv` is a CROSS-correlation (no kernel flip), so

        out[0] = z0*k0 + z1*k1 = (1+1i) + (1+2i) = 2+3i
        out[1] = z1*k0 + z2*k1 = (-1+1i) + (6+2i) = 5+3i

    With the real branch's subtraction inverted, `out[0]`'s real part reads
    `1 + (-1) = 0` rather than `2`. This cell also pins the kernel ORIENTATION:
    a flipped kernel would swap `k0` and `k1` and give `out[0] = 0+2i`.
    """
    layer = ComplexConv2D(filters=1, kernel_size=(1, 2), strides=1, padding="VALID")
    layer.build((1, 1, 3, 1))
    _assign_complex(layer.kernel, [[[[1 + 1j]], [[2 - 1j]]]])   # (1, 2, 1, 1)
    _assign_complex(layer.bias, [0 + 0j])

    inputs = tf.constant(
        np.array([[[[1 + 0j], [0 + 1j], [2 + 2j]]]], dtype=np.complex64)
    )
    outputs = layer(inputs)

    assert tuple(outputs.shape) == (1, 1, 2, 1)
    _assert_complex_allclose(
        outputs, [[[[2.0], [5.0]]]], [[[[3.0], [3.0]]]],
        "ComplexConv2D's two-tap sliding product is wrong",
    )


def test_complex_conv2d_bias_is_added_to_both_components():
    """The bias is complex and must reach BOTH components, not just the real one."""
    layer = ComplexConv2D(filters=1, kernel_size=1, strides=1, padding="VALID")
    layer.build((1, 1, 1, 1))
    _assign_complex(layer.kernel, [[[[1 + 0j]]]])
    _assign_complex(layer.bias, [10 - 20j])

    inputs = tf.constant(np.array([[[[1 + 2j]]]], dtype=np.complex64))
    _assert_complex_allclose(
        layer(inputs), [[[[11.0]]]], [[[[-18.0]]]],
        "ComplexConv2D did not add the complex bias to both components",
    )


# ---------------------------------------------------------------------
# ASYMMETRIC shape cells — the 72-cell grid above is square-only (Step 6.2)
# ---------------------------------------------------------------------
#
# Every cell of `_SHAPE_GRID` uses input `(1, S, S, 3)` with a SCALAR kernel and
# a SCALAR stride, so H == W, kernel[0] == kernel[1] and strides[0] == strides[1]
# in 72/72 cells and a height/width transposition is invisible. MEASURED at
# `d148888a7`: swapping `strides[0]`/`strides[1]` and `kernel_size[0]`/
# `kernel_size[1]` between the two branches of `ComplexConv2D.compute_output_shape`
# left this suite at 159 passed; the same swap in `ComplexAveragePooling2D._compute_dim`
# also left it at 159 passed. `kernel_size`/`pool_size`/`strides` are documented as
# `Union[int, Tuple[int, int]]`, so the asymmetric form is a supported API surface
# that had zero coverage. This is the repo's own "a layout decision needs a layout
# guard" lesson.

_ASYM_SHAPE_GRID = [
    (cls_name, padding, kernel, strides, size)
    for cls_name in ("ComplexConv2D", "ComplexAveragePooling2D")
    for padding in ("SAME", "VALID")
    for kernel in ((2, 3), (3, 2))
    for strides in ((1, 2), (2, 1), (2, 3))
    for size in ((7, 9), (9, 7))
]


def _make_asym_shape_layer(cls_name: str, padding: str, kernel, strides):
    """Build one asymmetric grid cell (unbuilt)."""
    if cls_name == "ComplexConv2D":
        return ComplexConv2D(
            filters=4, kernel_size=kernel, strides=strides, padding=padding
        )
    return ComplexAveragePooling2D(
        pool_size=kernel, strides=strides, padding=padding
    )


def _asym_shape_id(cell) -> str:
    cls_name, padding, kernel, strides, size = cell
    return (
        f"{cls_name}-{padding}-k{kernel[0]}x{kernel[1]}"
        f"-s{strides[0]}x{strides[1]}-in{size[0]}x{size[1]}"
    )


@pytest.mark.parametrize(
    "cls_name,padding,kernel,strides,size",
    _ASYM_SHAPE_GRID,
    ids=[_asym_shape_id(cell) for cell in _ASYM_SHAPE_GRID],
)
def test_compute_output_shape_agrees_with_forward_pass_on_asymmetric_cells(
    cls_name, padding, kernel, strides, size
):
    """Non-square input, non-square kernel, non-equal strides — built AND unbuilt.

    Every cell has `kernel[0] != kernel[1]`, `strides[0] != strides[1]` and
    `H != W`, so a height/width transposition anywhere in either shape method
    changes the answer. The oracle is the shape of a REAL forward pass, never a
    re-derived formula.
    """
    height, width = size
    inputs = tf.complex(
        tf.random.normal((1, height, width, 3)),
        tf.random.normal((1, height, width, 3)),
    )
    input_shape = tuple(inputs.shape)

    unbuilt = _make_asym_shape_layer(cls_name, padding, kernel, strides)
    unbuilt_shape = tuple(unbuilt.compute_output_shape(input_shape))

    built = _make_asym_shape_layer(cls_name, padding, kernel, strides)
    forward_shape = tuple(built(inputs).shape)
    built_shape = tuple(built.compute_output_shape(input_shape))

    assert built_shape == forward_shape, (
        f"{cls_name} padding={padding} kernel={kernel} strides={strides} "
        f"input={input_shape}: compute_output_shape returned {built_shape} but "
        f"the real forward pass produced {forward_shape}"
    )
    assert unbuilt_shape == forward_shape, (
        f"{cls_name} padding={padding} kernel={kernel} strides={strides} "
        f"input={input_shape}: an UNBUILT layer's compute_output_shape returned "
        f"{unbuilt_shape} but the real forward pass produced {forward_shape}"
    )


# ---------------------------------------------------------------------
# `compute_output_shape` for the three remaining classes (Step 6.2)
# ---------------------------------------------------------------------
#
# The grid above covers `ComplexConv2D` and `ComplexAveragePooling2D` only, and
# `ComplexGlobalAveragePooling2D` has its own pin. The other three were
# unguarded: MEASURED at `d148888a7`, replacing `ComplexDropout.compute_output_shape`
# with `lambda self, s: (s[0],)` left this suite at 159 passed, as did the same
# mutation on `ComplexReLU`, as did `ComplexDense.compute_output_shape` ignoring
# `units` entirely. Every case below therefore uses `units != input_shape[-1]`,
# so a method that echoes its argument is separable from a correct one.

_FORWARD_SHAPE_CASES = [
    ("ComplexDense-2D", lambda: ComplexDense(units=5), (2, 3)),
    ("ComplexDense-3D", lambda: ComplexDense(units=5), (2, 4, 3)),
    ("ComplexDense-narrowing", lambda: ComplexDense(units=1), (3, 7)),
    ("ComplexReLU-2D", lambda: ComplexReLU(), (2, 3)),
    ("ComplexReLU-4D", lambda: ComplexReLU(), (2, 5, 5, 3)),
    ("ComplexDropout-2D", lambda: ComplexDropout(rate=0.3), (2, 3)),
    ("ComplexDropout-4D", lambda: ComplexDropout(rate=0.0), (2, 5, 5, 3)),
]


@pytest.mark.parametrize(
    "factory,input_shape",
    [(factory, shape) for _, factory, shape in _FORWARD_SHAPE_CASES],
    ids=[case_id for case_id, _, _ in _FORWARD_SHAPE_CASES],
)
def test_compute_output_shape_agrees_with_the_forward_pass(factory, input_shape):
    """`compute_output_shape` must equal `tuple(forward_output.shape)`, built AND unbuilt.

    Guide 3.4 requires the answer to come from stored config alone, so the
    unbuilt instance is a separate object that is never built or called.
    """
    inputs = tf.complex(
        tf.random.normal(input_shape), tf.random.normal(input_shape)
    )

    unbuilt = factory()
    unbuilt_shape = tuple(unbuilt.compute_output_shape(input_shape))

    built = factory()
    forward_shape = tuple(built(inputs, training=False).shape)
    built_shape = tuple(built.compute_output_shape(input_shape))

    assert built_shape == forward_shape, (
        f"{type(built).__name__} on input {input_shape}: compute_output_shape "
        f"returned {built_shape} but the real forward pass produced {forward_shape}"
    )
    assert unbuilt_shape == forward_shape, (
        f"{type(unbuilt).__name__} on input {input_shape}: an UNBUILT layer's "
        f"compute_output_shape returned {unbuilt_shape} but the real forward pass "
        f"produced {forward_shape} (guide 3.4: the answer must come from stored "
        "config alone)"
    )


# ---------------------------------------------------------------------
# Constructor / build validation on ComplexConv2D (Step 6.2)
# ---------------------------------------------------------------------
#
# Both raises below survived deletion at 159 passed, MEASURED at `d148888a7`.
#
# The carried gap noted here at step 6.2 -- `ComplexConv2D.compute_output_shape`
# alone among the shape methods did no rank check, so a rank-2 shape raised
# `IndexError` from `input_shape[2]` instead of the class's own `ValueError` --
# is CLOSED at step 2.2 by the guard below plus the matching `src/` check.

@pytest.mark.parametrize("filters", [0, -1, -32], ids=["zero", "minus_one", "minus_32"])
def test_complex_conv2d_rejects_a_non_positive_filter_count(filters):
    """A `filters <= 0` kernel shape fails late and obscurely inside `add_weight`."""
    with pytest.raises(ValueError, match="filters must be positive"):
        ComplexConv2D(filters=filters, kernel_size=3)


@pytest.mark.parametrize(
    "input_shape",
    [(8,), (4, 8), (2, 8, 3), (2, 8, 8, 8, 3)],
    ids=["rank1", "rank2", "rank3", "rank5"],
)
def test_complex_conv2d_build_rejects_a_non_4d_input_shape(input_shape):
    """`build` must name the rank error itself, not let `keras.ops.conv` raise later."""
    layer = ComplexConv2D(filters=4, kernel_size=3)
    with pytest.raises(ValueError, match="requires 4D input"):
        layer.build(input_shape)


def test_complex_conv2d_accepts_a_4d_input_shape():
    """ANTI-VACUITY for the guard above: the rank it does accept must still build."""
    layer = ComplexConv2D(filters=4, kernel_size=3)
    layer.build((2, 8, 8, 3))
    assert layer.built is True
    assert tuple(layer.kernel.shape) == (3, 3, 3, 4)


@pytest.mark.parametrize(
    "input_shape",
    [(8,), (4, 8), (2, 8, 3), (2, 8, 8, 8, 3)],
    ids=["rank1", "rank2", "rank3", "rank5"],
)
def test_complex_conv2d_compute_output_shape_rejects_non_4d(input_shape):
    """`compute_output_shape` must raise the class's OWN error, not `IndexError`.

    `pytest.raises(ValueError)` is load-bearing: at HEAD before this guard the
    rank-2 and rank-1 cells raised `IndexError` out of `input_shape[2]`, which is
    the shape method crashing rather than judging. The two pooling classes both
    already raise `ValueError` here, so this closes the one inconsistent method.
    """
    layer = ComplexConv2D(filters=4, kernel_size=3)
    with pytest.raises(ValueError, match="requires 4D input"):
        layer.compute_output_shape(input_shape)


def test_complex_conv2d_compute_output_shape_still_accepts_rank_4():
    """ANTI-VACUITY for the guard above: rank 4 must still return a shape."""
    layer = ComplexConv2D(filters=4, kernel_size=3, strides=2, padding="SAME")
    assert layer.compute_output_shape((2, 8, 8, 3)) == (2, 4, 4, 4)


# ---------------------------------------------------------------------
# ABSOLUTE strides oracles (Step 2.3)
# ---------------------------------------------------------------------
#
# The 48-cell asymmetric grid above is a RELATIVE oracle: it asserts
# `compute_output_shape == forward.shape`, which a CONSISTENTLY wrong pair
# satisfies. MEASURED at `0dc3bc1ed`: transposing `self.strides` in BOTH
# `ComplexConv2D.call` (all four `keras.ops.conv` calls) AND
# `ComplexConv2D.compute_output_shape` left the three suites at 263 passed --
# exactly baseline. The same both-sides transposition of `pool_size` + `strides`
# in `ComplexAveragePooling2D` also left them at 263. `kernel_size` asymmetry is
# already pinned ABSOLUTELY by the two-tap value test; `strides` asymmetry was
# not pinned at all. The two cells below fix that: each states a hand-computed
# output SHAPE and the hand-computed VALUES at that shape, so a transposition
# changes WHICH input elements are sampled and no amount of self-consistency
# saves it.

def _ramp_complex_2x4() -> tf.Tensor:
    """A `(1, 2, 4, 1)` complex map whose every element is distinguishable.

    real = [[1, 2, 3, 4], [5, 6, 7, 8]], imag = 10 * real. With a `1 + 0i`
    kernel of size 1x1 the output IS the set of sampled inputs, so the values
    name the sampled positions directly.
    """
    real = np.array([[[[1.0], [2.0], [3.0], [4.0]],
                      [[5.0], [6.0], [7.0], [8.0]]]], dtype=np.float32)
    return _complex_from_parts(real, real * 10.0)


def test_complex_conv2d_asymmetric_strides_sample_the_hand_computed_positions():
    """`strides=(1, 2)` on a 2x4 map: an ABSOLUTE shape AND value oracle.

    A 1x1 identity kernel with `padding='VALID'` and `strides=(1, 2)` keeps every
    row and every SECOND column, so the output is exactly

        [[1, 3],
         [5, 7]]   (+ 10i each)

    Transposing the strides to `(2, 1)` -- even consistently, in both `call` and
    `compute_output_shape` -- keeps every second ROW and every column instead,
    giving `(1, 1, 4, 1)` and `[[1, 2, 3, 4]]`. Both assertions below therefore
    fire, which is what the relative grid could not do.
    """
    layer = ComplexConv2D(filters=1, kernel_size=1, strides=(1, 2), padding="VALID")
    layer.build((1, 2, 4, 1))
    _assign_complex(layer.kernel, [[[[1 + 0j]]]])
    _assign_complex(layer.bias, [0 + 0j])

    outputs = layer(_ramp_complex_2x4())

    assert tuple(outputs.shape) == (1, 2, 2, 1), (
        "ComplexConv2D with strides=(1, 2) on a (1, 2, 4, 1) input must produce "
        f"(1, 2, 2, 1) -- height undivided, width halved -- but produced "
        f"{tuple(outputs.shape)}; a height/width transposition of `strides` "
        "gives (1, 1, 4, 1)"
    )
    assert layer.compute_output_shape((1, 2, 4, 1)) == (1, 2, 2, 1), (
        "compute_output_shape disagrees with the hand-computed (1, 2, 2, 1) for "
        f"strides=(1, 2): {layer.compute_output_shape((1, 2, 4, 1))}"
    )
    _assert_complex_allclose(
        outputs, [[[[1.0], [3.0]], [[5.0], [7.0]]]],
        [[[[10.0], [30.0]], [[50.0], [70.0]]]],
        "ComplexConv2D with strides=(1, 2) sampled the wrong input elements -- "
        "it kept every second ROW instead of every second COLUMN",
    )


def test_complex_average_pooling_asymmetric_window_averages_the_hand_computed_pairs():
    """`pool_size=(1, 2)`, `strides=(1, 2)` on a 2x4 map: ABSOLUTE shape AND values.

    A 1-high, 2-wide window stepping 2 across averages HORIZONTAL pairs:

        [[ (1+2)/2, (3+4)/2 ],     [[1.5, 3.5],
         [ (5+6)/2, (7+8)/2 ]]  =   [5.5, 7.5]]   (imag = 10x)

    Transposing `pool_size` and `strides` together to `(2, 1)` -- consistently,
    in both `call` and `compute_output_shape` -- averages VERTICAL pairs instead
    and gives `(1, 1, 4, 1)` = `[[3, 4, 5, 6]]`. Note 1.5 != 3 and 3.5 != 4, so
    the value assertion is not satisfiable by the transposed layout even where
    the shapes happen to coincide.
    """
    layer = ComplexAveragePooling2D(pool_size=(1, 2), strides=(1, 2), padding="VALID")

    outputs = layer(_ramp_complex_2x4())

    assert tuple(outputs.shape) == (1, 2, 2, 1), (
        "ComplexAveragePooling2D with pool_size=strides=(1, 2) on a "
        f"(1, 2, 4, 1) input must produce (1, 2, 2, 1) but produced "
        f"{tuple(outputs.shape)}; transposing the window gives (1, 1, 4, 1)"
    )
    assert layer.compute_output_shape((1, 2, 4, 1)) == (1, 2, 2, 1), (
        "compute_output_shape disagrees with the hand-computed (1, 2, 2, 1) for "
        f"pool_size=strides=(1, 2): {layer.compute_output_shape((1, 2, 4, 1))}"
    )
    _assert_complex_allclose(
        outputs, [[[[1.5], [3.5]], [[5.5], [7.5]]]],
        [[[[15.0], [35.0]], [[55.0], [75.0]]]],
        "ComplexAveragePooling2D with a 1x2 window averaged VERTICAL pairs -- "
        "`pool_size`/`strides` are transposed",
    )


# ---------------------------------------------------------------------
# D-002 evidence pin: the Initializer determinism trap (Step 2.3)
# ---------------------------------------------------------------------

def test_one_initializer_instance_replays_its_draw_at_a_fixed_shape():
    """The trap that made the D-002 anchor wrong TWICE, pinned so it cannot recur.

    The anchor at `base.py:ComplexLayer.__init__` describes the wire-up it is
    refusing. Two successive wordings described it as "draw real and imag
    separately from the passed initializer" using ONE instance -- which is
    degenerate, because a keras 3.8 `Initializer` instance is DETERMINISTIC PER
    SHAPE. This test asserts the mechanism directly in both directions, so a
    third wrong wording cannot ship under a green suite:

    * one instance called twice at one shape -> bit-identical arrays, hence
      `imag == real` and a "complex" weight with exactly 2 distinct phases;
    * two DISTINCT-SEED instances -> different arrays, hence a real phase spread.
    """
    shape = (5, 5, 3, 20)          # CoShNet's first-conv kernel

    for name, ctor in (
        ("GlorotUniform", keras.initializers.GlorotUniform),
        ("HeNormal", keras.initializers.HeNormal),
    ):
        one = ctor()
        first, second = np.array(one(shape)), np.array(one(shape))
        assert np.array_equal(first, second), (
            f"a single {name} instance returned DIFFERENT arrays on two calls at "
            "the same shape. keras Initializer determinism-per-shape is the "
            "premise of the D-002 anchor's TRAP paragraph; if it no longer "
            "holds, reword the anchor rather than deleting this test"
        )

        degenerate = first + 1j * second
        assert len(np.unique(np.round(np.angle(degenerate), 4))) == 2, (
            f"the one-instance {name} 'complex' draw did not collapse to 2 "
            "phases; the anchor's stated tell (it is a real kernel times (1+i)) "
            "is stale"
        )

        spread = np.array(ctor(seed=1)(shape)) + 1j * np.array(ctor(seed=2)(shape))
        assert len(np.unique(np.round(np.angle(spread), 4))) > 1000, (
            f"two DISTINCT-SEED {name} instances produced only "
            f"{len(np.unique(np.round(np.angle(spread), 4)))} distinct phases -- "
            "the wire-up the anchor describes as workable is not workable, so "
            "the anchor is wrong again"
        )


if __name__ == '__main__':
    pytest.main([__file__])
