"""
Tests for Complex-Valued Neural Network Layers
============================================

This module provides comprehensive tests for complex-valued neural network layers,
including initialization tests, shape verification, and numerical correctness checks.
"""

import ast
import keras
import pytest
import inspect
import tempfile
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from typing import Any, Tuple, List, Optional

from dl_techniques.layers.complex.complex_layers import (
    ComplexLayer,
    ComplexConv2D,
    ComplexDense,
    ComplexReLU,
    ComplexAveragePooling2D,
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

# The module physically holding `ComplexLayer`. The AST guard below parses THIS
# object's source, so when `ComplexLayer` is relocated, repointing this single
# import is the whole change.
from dl_techniques.layers.complex import complex_layers as _complex_layer_module

_COMPLEX_LAYER_MODULE_NAME = "complex_layers.py"


def test_kernel_initializer_is_read_by_exactly_two_ast_nodes_and_neither_computes():
    """The mechanism, asserted rather than described.

    `self.kernel_initializer` appears at exactly two places in the AST of the
    module holding `ComplexLayer`: the assignment in `__init__` and the entry in
    `get_config`. A third site means the knob has acquired a consumer and D-002's
    pin-as-documented-dead ruling must be revisited.

    The predicate is AST, deliberately: the DECISION comment placed at the site
    names the attribute, so a `source.count("self.kernel_initializer")` cannot
    tell a consumer from a comment about the absence of consumers. This mirrors
    the `epsilon` guard in
    `tests/test_models/test_the_two_documented_dead_knobs.py`.
    """
    tree = ast.parse(inspect.getsource(_complex_layer_module))
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "kernel_initializer"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ]
    assert len(nodes) == 2, (
        f"`self.kernel_initializer` now appears at {len(nodes)} AST sites in "
        f"{_COMPLEX_LAYER_MODULE_NAME} (expected exactly 2: the __init__ "
        "assignment and the get_config entry). A new site means the knob is no "
        "longer inert and D-002 must be re-decided, not patched."
    )
    # One is a Store (the assignment), one is a Load (the get_config read).
    contexts = sorted(type(node.ctx).__name__ for node in nodes)
    assert contexts == ["Load", "Store"], (
        f"expected one Store and one Load, got {contexts} — a second Load is a "
        "computation reading the knob"
    )


def test_epsilon_is_still_exactly_two_ast_nodes_in_the_same_module():
    """Invariant 2 re-checked here, where `kernel_initializer` is edited.

    The two knobs share one assignment block; an edit to one is exactly the kind
    of change that could add a site to the other.
    """
    tree = ast.parse(inspect.getsource(_complex_layer_module))
    nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "epsilon"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ]
    assert len(nodes) == 2, (
        f"`self.epsilon` now appears at {len(nodes)} AST sites in "
        f"{_COMPLEX_LAYER_MODULE_NAME} (expected exactly 2)."
    )


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


if __name__ == '__main__':
    pytest.main([__file__])
