"""
Test suite for orthogonal regularization implementations.

This module provides comprehensive tests for:
- SoftOrthogonalConstraintRegularizer
- SoftOrthonormalConstraintRegularizer
- Helper functions
- Edge cases and numerical stability
- Serialization/deserialization
"""

import pytest
import tensorflow as tf
import numpy as np
from typing import Any, Dict, Tuple

from dl_techniques.regularizers.soft_orthogonal import (
    SoftOrthogonalConstraintRegularizer,
    SoftOrthonormalConstraintRegularizer,
    DEFAULT_SOFTORTHOGONAL_LAMBDA,
    DEFAULT_SOFTORTHOGONAL_L1,
    DEFAULT_SOFTORTHOGONAL_L2
)


# Test fixtures
@pytest.fixture
def random_weights_2d() -> tf.Tensor:
    """Generate random 2D weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((10, 5))


@pytest.fixture
def random_weights_4d() -> tf.Tensor:
    """Generate random 4D weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 64, 32))


@pytest.fixture
def orthogonal_matrix() -> tf.Tensor:
    """Generate a known orthogonal matrix for testing."""
    # Create a simple 2x2 rotation matrix (orthogonal)
    theta = np.pi / 4  # 45 degrees
    matrix = tf.constant([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=tf.float32)
    return matrix


# SoftOrthogonalConstraintRegularizer tests
def test_soft_orthogonal_default_params() -> None:
    """Test SoftOrthogonalConstraintRegularizer with default parameters."""
    regularizer = SoftOrthogonalConstraintRegularizer()
    assert regularizer._lambda_coefficient == DEFAULT_SOFTORTHOGONAL_LAMBDA
    assert regularizer._l1_coefficient == DEFAULT_SOFTORTHOGONAL_L1
    assert regularizer._l2_coefficient == DEFAULT_SOFTORTHOGONAL_L2


def test_soft_orthogonal_custom_params() -> None:
    """Test SoftOrthogonalConstraintRegularizer with custom parameters."""
    params = {
        "lambda_coefficient": 0.1,
        "l1_coefficient": 0.2,
        "l2_coefficient": 0.3
    }
    regularizer = SoftOrthogonalConstraintRegularizer(**params)
    assert regularizer._lambda_coefficient == params["lambda_coefficient"]
    assert regularizer._l1_coefficient == params["l1_coefficient"]
    assert regularizer._l2_coefficient == params["l2_coefficient"]


def test_soft_orthogonal_zero_for_orthogonal_matrix(orthogonal_matrix: tf.Tensor) -> None:
    """Test that regularization is zero for perfectly orthogonal matrix."""
    regularizer = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(orthogonal_matrix)
    assert tf.abs(penalty) < 1e-5


def test_soft_orthogonal_nonzero_for_nonorthogonal(random_weights_2d: tf.Tensor) -> None:
    """Test that regularization is nonzero for non-orthogonal matrix."""
    regularizer = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(random_weights_2d)
    assert penalty > 1e-5


# SoftOrthonormalConstraintRegularizer tests
def test_soft_orthonormal_default_params() -> None:
    """Test SoftOrthonormalConstraintRegularizer with default parameters."""
    regularizer = SoftOrthonormalConstraintRegularizer()
    assert regularizer._lambda_coefficient == DEFAULT_SOFTORTHOGONAL_LAMBDA
    assert regularizer._l1_coefficient == DEFAULT_SOFTORTHOGONAL_L1
    assert regularizer._l2_coefficient == DEFAULT_SOFTORTHOGONAL_L2


def test_soft_orthonormal_custom_params() -> None:
    """Test SoftOrthonormalConstraintRegularizer with custom parameters."""
    params = {
        "lambda_coefficient": 0.1,
        "l1_coefficient": 0.2,
        "l2_coefficient": 0.3
    }
    regularizer = SoftOrthonormalConstraintRegularizer(**params)
    assert regularizer._lambda_coefficient == params["lambda_coefficient"]
    assert regularizer._l1_coefficient == params["l1_coefficient"]
    assert regularizer._l2_coefficient == params["l2_coefficient"]


def test_soft_orthonormal_zero_for_orthonormal_matrix(orthogonal_matrix: tf.Tensor) -> None:
    """Test that regularization is zero for perfectly orthonormal matrix."""
    regularizer = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(orthogonal_matrix)
    assert tf.abs(penalty) < 1e-5


def test_soft_orthonormal_nonzero_for_nonorthonormal(random_weights_2d: tf.Tensor) -> None:
    """Test that regularization is nonzero for non-orthonormal matrix."""
    regularizer = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0
    )
    penalty = regularizer(random_weights_2d)
    assert penalty > 1e-5


# Serialization tests
def test_soft_orthogonal_serialization() -> None:
    """Test serialization/deserialization of SoftOrthogonalConstraintRegularizer."""
    original = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=0.1,
        l1_coefficient=0.2,
        l2_coefficient=0.3
    )
    config = original.get_config()
    reconstructed = SoftOrthogonalConstraintRegularizer.from_config(config)
    assert original._lambda_coefficient == reconstructed._lambda_coefficient
    assert original._l1_coefficient == reconstructed._l1_coefficient
    assert original._l2_coefficient == reconstructed._l2_coefficient


def test_soft_orthonormal_serialization() -> None:
    """Test serialization/deserialization of SoftOrthonormalConstraintRegularizer."""
    original = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=0.1,
        l1_coefficient=0.2,
        l2_coefficient=0.3
    )
    config = original.get_config()
    reconstructed = SoftOrthonormalConstraintRegularizer.from_config(config)
    assert original._lambda_coefficient == reconstructed._lambda_coefficient
    assert original._l1_coefficient == reconstructed._l1_coefficient
    assert original._l2_coefficient == reconstructed._l2_coefficient


# Edge cases and numerical stability tests
@pytest.mark.parametrize("shape", [
    (1, 1),
    (100, 100)
])
def test_valid_shapes(shape: Tuple[int, int]) -> None:
    """Test regularizers with valid shapes."""
    weights = tf.random.normal(shape)
    regularizer = SoftOrthogonalConstraintRegularizer()
    penalty = regularizer(weights)
    assert not tf.math.is_nan(penalty)
    assert not tf.math.is_inf(penalty)


def test_numerical_stability_large_values() -> None:
    """Test numerical stability with large values."""
    weights = tf.random.normal((10, 5)) * 1e6
    regularizer = SoftOrthogonalConstraintRegularizer()
    penalty = regularizer(weights)
    assert not tf.math.is_nan(penalty)
    assert not tf.math.is_inf(penalty)


def test_numerical_stability_small_values() -> None:
    """Test numerical stability with small values."""
    weights = tf.random.normal((10, 5)) * 1e-6
    regularizer = SoftOrthogonalConstraintRegularizer()
    penalty = regularizer(weights)
    assert not tf.math.is_nan(penalty)
    assert not tf.math.is_inf(penalty)


# Integration tests
def test_integration_with_keras_layer() -> None:
    """Test regularizer integration with Keras layer."""
    # Enable eager execution for this test
    tf.config.run_functions_eagerly(True)

    try:
        regularizer = SoftOrthogonalConstraintRegularizer()

        # Suppress warnings about input_shape by using Input layer
        inputs = tf.keras.Input(shape=(5,))
        outputs = tf.keras.layers.Dense(
            10,
            kernel_regularizer=regularizer
        )(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer='adam', loss='mse')

        # Generate random data
        x = tf.random.normal((100, 5))
        y = tf.random.normal((100, 10))

        # Check that training works without errors
        with tf.keras.utils.custom_object_scope({
            'SoftOrthogonalConstraintRegularizer': SoftOrthogonalConstraintRegularizer
        }):
            history = model.fit(x, y, epochs=1, verbose=0)
            assert len(history.history['loss']) == 1

    finally:
        # Restore original execution mode
        tf.config.run_functions_eagerly(False)