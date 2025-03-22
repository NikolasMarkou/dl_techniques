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


@pytest.fixture
def conv_small() -> tf.Tensor:
    """Generate small random 4D conv weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 8, 16))  # [h, w, in_channels, out_channels]


@pytest.fixture
def conv_medium() -> tf.Tensor:
    """Generate medium random 4D conv weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 16, 32))  # [h, w, in_channels, out_channels]


@pytest.fixture
def conv_large() -> tf.Tensor:
    """Generate large random 4D conv weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((3, 3, 32, 64))  # [h, w, in_channels, out_channels]


@pytest.fixture
def dense_small() -> tf.Tensor:
    """Generate small random 2D dense weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((64, 32))  # [input_dim, output_dim]


@pytest.fixture
def dense_large() -> tf.Tensor:
    """Generate large random 2D dense weights for testing."""
    tf.random.set_seed(42)
    return tf.random.normal((256, 128))  # [input_dim, output_dim]


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


# ---------------------------------------------------------------------


def test_scaling_effect_on_regularization_strength(conv_small: tf.Tensor, conv_large: tf.Tensor) -> None:
    """
    Test 1: Verify that matrix scaling affects regularization values as expected.

    This test confirms that:
    1. Regularization values differ when scaling is enabled
    2. Without scaling, larger matrices have much higher regularization values
    3. With scaling, regularization values are more proportional to matrix size
    """
    # Create regularizers with and without scaling
    ortho_no_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    ortho_with_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    # Calculate regularization values
    small_no_scaling = ortho_no_scaling(conv_small)
    small_with_scaling = ortho_with_scaling(conv_small)
    large_no_scaling = ortho_no_scaling(conv_large)
    large_with_scaling = ortho_with_scaling(conv_large)

    # Calculate ratios of large to small
    ratio_no_scaling = large_no_scaling / small_no_scaling
    ratio_with_scaling = large_with_scaling / small_with_scaling

    # Print values for diagnostic purposes
    print(f"Small Conv - No scaling: {small_no_scaling:.6f}, With scaling: {small_with_scaling:.6f}")
    print(f"Large Conv - No scaling: {large_no_scaling:.6f}, With scaling: {large_with_scaling:.6f}")
    print(f"Ratio - No scaling: {ratio_no_scaling:.6f}, With scaling: {ratio_with_scaling:.6f}")

    # Assert expected behavior
    assert small_no_scaling != small_with_scaling, "Scaling should affect regularization values"
    assert ratio_no_scaling > ratio_with_scaling, "Scaling should reduce the ratio between large and small"

    # The exact ratio depends on implementation, but should be significantly reduced
    assert ratio_with_scaling < ratio_no_scaling * 0.5, "Scaling should substantially reduce the large/small ratio"


def test_scaling_across_filter_sizes() -> None:
    """
    Test 2: Verify scaling behavior across different filter sizes.

    This test confirms that:
    1. Without scaling, regularization grows rapidly with filter count
    2. With scaling, regularization grows more slowly and consistently
    """
    # Define filter sizes to test
    filter_sizes = [8, 16, 32, 64, 128]
    results_no_scaling = []
    results_with_scaling = []

    # Create regularizers
    ortho_no_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    ortho_with_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    # Test with each filter size
    for filters in filter_sizes:
        # Create test weights with fixed kernel size but different filter count
        tf.random.set_seed(42)  # For reproducibility
        weights = tf.random.normal((3, 3, 3, filters))

        # Calculate regularization values
        val_no_scaling = ortho_no_scaling(weights).numpy()
        val_with_scaling = ortho_with_scaling(weights).numpy()

        results_no_scaling.append(val_no_scaling)
        results_with_scaling.append(val_with_scaling)

        print(f"Filters: {filters}, No scaling: {val_no_scaling:.6f}, With scaling: {val_with_scaling:.6f}")

    # Calculate growth rates between consecutive filter sizes
    growth_rates_no_scaling = [results_no_scaling[i] / results_no_scaling[i - 1]
                               for i in range(1, len(results_no_scaling))]

    growth_rates_with_scaling = [results_with_scaling[i] / results_with_scaling[i - 1]
                                 for i in range(1, len(results_with_scaling))]

    print(f"Growth rates without scaling: {growth_rates_no_scaling}")
    print(f"Growth rates with scaling: {growth_rates_with_scaling}")

    # Without scaling, growth rate should be higher than with scaling
    assert all(g_no > g_with for g_no, g_with in zip(growth_rates_no_scaling, growth_rates_with_scaling)), \
        "Growth rate should be lower with scaling enabled"

    # Check scaling effect between smallest and largest filter counts
    ratio_no_scaling = results_no_scaling[-1] / results_no_scaling[0]
    ratio_with_scaling = results_with_scaling[-1] / results_with_scaling[0]

    print(
        f"Ratio from smallest to largest - No scaling: {ratio_no_scaling:.6f}, With scaling: {ratio_with_scaling:.6f}")
    assert ratio_with_scaling < ratio_no_scaling, "Scaling should reduce the ratio between largest and smallest"


def test_scaling_consistency_between_dense_and_conv(
        dense_small: tf.Tensor,
        dense_large: tf.Tensor,
        conv_small: tf.Tensor,
        conv_large: tf.Tensor
) -> None:
    """
    Test 3: Verify scaling consistency between dense and convolutional layers.

    This test confirms that:
    1. Scaling behavior is consistent between different layer types
    2. Ratios between large and small layers are similar with scaling enabled
    """
    # Create regularizers
    ortho_with_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    # Calculate regularization values
    dense_small_val = ortho_with_scaling(dense_small)
    dense_large_val = ortho_with_scaling(dense_large)
    conv_small_val = ortho_with_scaling(conv_small)
    conv_large_val = ortho_with_scaling(conv_large)

    # Calculate ratios
    dense_ratio = dense_large_val / dense_small_val
    conv_ratio = conv_large_val / conv_small_val

    print(f"Dense small: {dense_small_val:.6f}, large: {dense_large_val:.6f}, ratio: {dense_ratio:.6f}")
    print(f"Conv small: {conv_small_val:.6f}, large: {conv_large_val:.6f}, ratio: {conv_ratio:.6f}")

    # The ratios should be reasonably close with scaling enabled
    # We use a relatively loose bound since different layer types have
    # different characteristics
    ratio_difference = abs(dense_ratio.numpy() - conv_ratio.numpy())
    print(f"Ratio difference: {ratio_difference:.6f}")

    # Assert the ratio difference is within reasonable bounds
    assert ratio_difference < max(dense_ratio.numpy(), conv_ratio.numpy()) * 0.5, \
        "Ratios between dense and conv should be reasonably similar with scaling"


def test_orthogonal_vs_orthonormal_scaling(conv_medium: tf.Tensor) -> None:
    """
    Test 4: Compare scaling behavior between orthogonal and orthonormal regularizers.

    This test confirms that:
    1. Both orthogonal and orthonormal regularizers are affected by scaling
    2. The scaling implementation is appropriate for each regularizer type
    """
    # Create regularizers - both with and without scaling
    orthogonal_no_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    orthogonal_with_scaling = SoftOrthogonalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    orthonormal_no_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=False
    )

    orthonormal_with_scaling = SoftOrthonormalConstraintRegularizer(
        lambda_coefficient=1.0,
        l1_coefficient=0.0,
        l2_coefficient=0.0,
        use_matrix_scaling=True
    )

    # Calculate regularization values
    orthogonal_no_scale_val = orthogonal_no_scaling(conv_medium)
    orthogonal_with_scale_val = orthogonal_with_scaling(conv_medium)
    orthonormal_no_scale_val = orthonormal_no_scaling(conv_medium)
    orthonormal_with_scale_val = orthonormal_with_scaling(conv_medium)

    print(f"Orthogonal - No scaling: {orthogonal_no_scale_val:.6f}, With scaling: {orthogonal_with_scale_val:.6f}")
    print(f"Orthonormal - No scaling: {orthonormal_no_scale_val:.6f}, With scaling: {orthonormal_with_scale_val:.6f}")

    # Calculate scaling effect ratios
    orthogonal_scale_ratio = orthogonal_no_scale_val / orthogonal_with_scale_val
    orthonormal_scale_ratio = orthonormal_no_scale_val / orthonormal_with_scale_val

    print(
        f"Scaling effect ratio - Orthogonal: {orthogonal_scale_ratio:.6f}, Orthonormal: {orthonormal_scale_ratio:.6f}")

    # Both regularizers should have their values reduced by scaling
    assert orthogonal_with_scale_val < orthogonal_no_scale_val, "Scaling should reduce orthogonal regularization value"
    assert orthonormal_with_scale_val < orthonormal_no_scale_val, "Scaling should reduce orthonormal regularization value"

    # The scaling effect should be substantial for both types
    assert orthogonal_scale_ratio > 2.0, "Scaling should substantially reduce orthogonal regularization"
    assert orthonormal_scale_ratio > 2.0, "Scaling should substantially reduce orthonormal regularization"

    # The scaling effect might differ between types due to diagonal vs. off-diagonal elements
    # but should be in a reasonable range
    ratio_difference = abs(orthogonal_scale_ratio.numpy() - orthonormal_scale_ratio.numpy())
    print(f"Scaling effect ratio difference: {ratio_difference:.6f}")

    # The difference in scaling effect should not be extreme
    max_ratio = max(orthogonal_scale_ratio.numpy(), orthonormal_scale_ratio.numpy())
    assert ratio_difference < max_ratio * 0.5, "Scaling effect should be reasonably similar for both regularizer types"
