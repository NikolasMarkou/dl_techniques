"""
Tests for Kolmogorov-Arnold Network (KAN) Implementation

This module contains comprehensive tests for the KAN implementation:
- KANLinear layer functionality
- KAN model functionality
- Initialization, forward pass, serialization, and integration tests
- Edge cases and numerical stability tests

Tests cover initialization, forward pass, serialization, edge cases, and integration scenarios.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Any, Dict, List

from dl_techniques.utils.logger import logger
from dl_techniques.layers.kan_linear import KANLinear


class TestKANLinear:
    """Test suite for KANLinear layer implementation."""

    @pytest.fixture
    def input_tensor(self) -> keras.KerasTensor:
        """Create a test input tensor.

        Returns:
            Random input tensor for testing.
        """
        return keras.random.normal([16, 10])

    @pytest.fixture
    def small_input_tensor(self) -> keras.KerasTensor:
        """Create a small test input tensor.

        Returns:
            Small random input tensor for testing.
        """
        return keras.random.normal([4, 5])

    @pytest.fixture
    def layer_instance(self) -> KANLinear:
        """Create a default KANLinear layer instance for testing.

        Returns:
            KANLinear layer with default parameters.
        """
        return KANLinear(in_features=10, out_features=8)

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        layer = KANLinear(in_features=10, out_features=5)

        # Check default values
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.grid_size == 5
        assert layer.spline_order == 3
        assert layer.activation_name == 'swish'
        assert layer.regularization_factor == 0.01
        assert layer.grid_range == (-1.0, 1.0)
        assert layer.epsilon == 1e-7
        assert layer.clip_value == 1e3
        assert layer.use_residual is False  # Different in/out features

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        layer = KANLinear(
            in_features=8,
            out_features=8,  # Same as input for residual
            grid_size=10,
            spline_order=4,
            activation='relu',
            regularization_factor=0.05,
            grid_range=(-2.0, 2.0),
            epsilon=1e-6,
            clip_value=500.0,
            use_residual=True
        )

        # Check custom values
        assert layer.in_features == 8
        assert layer.out_features == 8
        assert layer.grid_size == 10
        assert layer.spline_order == 4
        assert layer.activation_name == 'relu'
        assert layer.regularization_factor == 0.05
        assert layer.grid_range == (-2.0, 2.0)
        assert layer.epsilon == 1e-6
        assert layer.clip_value == 500.0
        assert layer.use_residual is True

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise appropriate errors."""
        # Negative features
        with pytest.raises(ValueError, match="Features must be positive integers"):
            KANLinear(in_features=-1, out_features=5)

        with pytest.raises(ValueError, match="Features must be positive integers"):
            KANLinear(in_features=5, out_features=0)

        # Grid size < spline order
        with pytest.raises(ValueError, match="Grid size must be >= spline order"):
            KANLinear(in_features=5, out_features=3, grid_size=2, spline_order=3)

        # Invalid grid range
        with pytest.raises(ValueError, match="Invalid grid range"):
            KANLinear(in_features=5, out_features=3, grid_range=(1.0, -1.0))

    def test_build_process(self, input_tensor: keras.KerasTensor) -> None:
        """Test that the layer builds properly."""
        layer = KANLinear(in_features=10, out_features=8)
        layer(input_tensor)  # Forward pass triggers build

        # Check that weights were created
        assert layer.built is True
        assert len(layer.weights) > 0
        assert hasattr(layer, "base_weight")
        assert hasattr(layer, "spline_weight")
        assert hasattr(layer, "spline_scaler")

        # Check weight shapes
        assert layer.base_weight.shape == (10, 8)
        assert layer.spline_weight.shape == (10, 8, 7)  # grid_size + spline_order - 1
        assert layer.spline_scaler.shape == (10, 8)

    def test_build_with_wrong_input_shape(self) -> None:
        """Test build process with incorrect input dimensions."""
        layer = KANLinear(in_features=10, out_features=8)

        # Test with 1D input (should fail)
        with pytest.raises(ValueError, match="Input must be at least 2D"):
            layer.build((10,))

        # Test with wrong last dimension
        wrong_input = keras.random.normal([16, 5])  # Should be 10, not 5
        with pytest.raises(ValueError, match="Input dimension .* doesn't match in_features"):
            layer(wrong_input)

    def test_output_shapes(self, input_tensor: keras.KerasTensor) -> None:
        """Test that output shapes are computed correctly."""
        layer_configs = [
            {"in_features": 10, "out_features": 8},
            {"in_features": 10, "out_features": 16},
            {"in_features": 10, "out_features": 1},
        ]

        for config in layer_configs:
            layer = KANLinear(**config)
            output = layer(input_tensor)

            # Check output shape
            expected_shape = (16, config["out_features"])
            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == expected_shape

    def test_forward_pass(self, input_tensor: keras.KerasTensor) -> None:
        """Test that forward pass produces expected values."""
        layer = KANLinear(in_features=10, out_features=8)
        output = layer(input_tensor)

        # Basic sanity checks
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))
        assert output.shape == (16, 8)

    def test_different_activations(self, small_input_tensor: keras.KerasTensor) -> None:
        """Test layer with different activation functions."""
        activations = ["relu", "swish", "gelu", "tanh", "sigmoid"]

        for activation in activations:
            layer = KANLinear(
                in_features=5,
                out_features=3,
                activation=activation
            )
            output = layer(small_input_tensor)

            # Check output is valid
            assert not ops.any(ops.isnan(output))
            assert output.shape == (4, 3)

    def test_residual_connections(self) -> None:
        """Test residual connections when enabled."""
        # Test with same input/output dimensions (residual possible)
        layer_with_residual = KANLinear(
            in_features=8,
            out_features=8,
            use_residual=True
        )

        # Test with different dimensions (residual disabled)
        layer_without_residual = KANLinear(
            in_features=8,
            out_features=6,
            use_residual=True  # Should be ignored
        )

        assert layer_with_residual.use_residual is True
        assert layer_without_residual.use_residual is False

    def test_serialization(self) -> None:
        """Test serialization and deserialization of the layer."""
        original_layer = KANLinear(
            in_features=6,
            out_features=4,
            grid_size=8,
            spline_order=4,
            activation='relu',
            regularization_factor=0.02
        )

        # Build the layer
        input_shape = (None, 6)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = KANLinear.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.in_features == original_layer.in_features
        assert recreated_layer.out_features == original_layer.out_features
        assert recreated_layer.grid_size == original_layer.grid_size
        assert recreated_layer.spline_order == original_layer.spline_order
        assert recreated_layer.activation_name == original_layer.activation_name

    def test_model_integration(self, input_tensor: keras.KerasTensor) -> None:
        """Test the layer in a model context."""
        # Create a simple model with the KANLinear layer
        inputs = keras.Input(shape=(10,))
        x = KANLinear(in_features=10, out_features=16)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = KANLinear(in_features=16, out_features=8)(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(3)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (16, 3)

    def test_grid_building(self) -> None:
        """Test grid building functionality."""
        layer = KANLinear(
            in_features=5,
            out_features=3,
            grid_size=6,
            grid_range=(-2.0, 3.0)
        )

        # Build layer to initialize grid
        test_input = keras.random.normal([4, 5])
        layer(test_input)

        # Check grid properties
        assert hasattr(layer, '_cached_grid')
        assert layer._cached_grid is not None
        assert ops.shape(layer._cached_grid)[0] == 6

    def test_normalize_inputs(self) -> None:
        """Test input normalization functionality."""
        layer = KANLinear(
            in_features=3,
            out_features=2,
            grid_range=(-1.0, 1.0)
        )

        # Build layer
        test_input = keras.random.normal([2, 3])
        layer(test_input)

        # Test normalization with known values
        test_values = ops.convert_to_tensor([[-1.0, 0.0, 1.0]])
        normalized = layer._normalize_inputs(test_values)

        # Should be in [0, 1] range
        assert ops.all(normalized >= 0.0)
        assert ops.all(normalized <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])