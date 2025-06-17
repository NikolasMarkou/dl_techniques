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
from dl_techniques.layers.kan import KANLinear, KAN


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


class TestKAN:
    """Test suite for KAN model implementation."""

    @pytest.fixture
    def simple_config(self) -> List[Dict[str, Any]]:
        """Create a simple KAN configuration.

        Returns:
            List of layer configurations for a simple KAN.
        """
        return [
            {"in_features": 10, "out_features": 8, "grid_size": 5},
            {"in_features": 8, "out_features": 4, "grid_size": 4},
            {"in_features": 4, "out_features": 2, "grid_size": 3}
        ]

    @pytest.fixture
    def input_tensor(self) -> keras.KerasTensor:
        """Create a test input tensor.

        Returns:
            Random input tensor for testing.
        """
        return keras.random.normal([16, 10])

    def test_initialization_simple(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test initialization with simple configuration."""
        model = KAN(layers_configurations=simple_config)

        assert len(model.kan_layers) == 3
        assert model.enable_debugging is False
        assert model.layers_configurations == simple_config

    def test_initialization_with_debugging(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test initialization with debugging enabled."""
        model = KAN(
            layers_configurations=simple_config,
            enable_debugging=True,
            name="test_kan"
        )

        assert model.enable_debugging is True
        assert model.name == "test_kan"

    def test_invalid_configurations(self) -> None:
        """Test that invalid configurations raise appropriate errors."""
        # Empty configuration
        with pytest.raises(ValueError, match="Empty layer configurations"):
            KAN(layers_configurations=[])

        # Missing required keys
        with pytest.raises(ValueError, match="missing required"):
            KAN(layers_configurations=[{"grid_size": 5}])

        # Incompatible layer dimensions
        incompatible_config = [
            {"in_features": 10, "out_features": 8},
            {"in_features": 6, "out_features": 4}  # Should be in_features=8
        ]
        with pytest.raises(ValueError, match="don't match"):
            KAN(layers_configurations=incompatible_config)

    def test_forward_pass(
        self,
        simple_config: List[Dict[str, Any]],
        input_tensor: keras.KerasTensor
    ) -> None:
        """Test forward pass through KAN model."""
        model = KAN(layers_configurations=simple_config)

        output = model(input_tensor)

        # Check output shape and validity
        assert output.shape == (16, 2)  # Final layer has 2 outputs
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_debugging_mode(
        self,
        simple_config: List[Dict[str, Any]],
        input_tensor: keras.KerasTensor
    ) -> None:
        """Test forward pass with debugging enabled."""
        model = KAN(
            layers_configurations=simple_config,
            enable_debugging=True
        )

        # Should not raise any errors and should log information
        output = model(input_tensor)
        assert output.shape == (16, 2)

    def test_model_compilation_and_training(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test KAN model compilation and basic training."""
        model = KAN(layers_configurations=simple_config)

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=keras.losses.MeanSquaredError(),
            metrics=['mae']
        )

        # Create dummy data
        x_train = keras.random.normal([32, 10])
        y_train = keras.random.normal([32, 2])

        # Test training (should not crash)
        history = model.fit(
            x_train, y_train,
            epochs=2,
            batch_size=16,
            verbose=0
        )

        assert 'loss' in history.history
        assert len(history.history['loss']) == 2

    def test_serialization(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test KAN model serialization."""
        original_model = KAN(
            layers_configurations=simple_config,
            enable_debugging=True,
            name="original_kan"
        )

        # Get config and recreate
        config = original_model.get_config()
        recreated_model = KAN.from_config(config)

        # Check configuration matches
        assert recreated_model.layers_configurations == original_model.layers_configurations
        assert recreated_model.enable_debugging == original_model.enable_debugging
        assert recreated_model.name == original_model.name

    def test_different_architectures(self) -> None:
        """Test KAN with different architectures."""
        architectures = [
            # Single layer
            [{"in_features": 5, "out_features": 3}],

            # Deep network
            [
                {"in_features": 10, "out_features": 20},
                {"in_features": 20, "out_features": 15},
                {"in_features": 15, "out_features": 10},
                {"in_features": 10, "out_features": 5},
                {"in_features": 5, "out_features": 1}
            ],

            # Wide network
            [
                {"in_features": 8, "out_features": 32},
                {"in_features": 32, "out_features": 4}
            ]
        ]

        for config in architectures:
            model = KAN(layers_configurations=config)

            # Test with appropriate input
            input_size = config[0]["in_features"]
            output_size = config[-1]["out_features"]

            test_input = keras.random.normal([8, input_size])
            output = model(test_input)

            assert output.shape == (8, output_size)
            assert not ops.any(ops.isnan(output))


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_empty_tensors(self) -> None:
        """Test with empty tensors."""
        layer = KANLinear(in_features=5, out_features=3)

        # Create empty tensor
        empty_input = ops.zeros((0, 5))
        output = layer(empty_input)

        # Should handle gracefully
        assert output.shape == (0, 3)

    def test_extreme_values(self) -> None:
        """Test with extreme input values."""
        layer = KANLinear(
            in_features=4,
            out_features=2,
            grid_range=(-10.0, 10.0),
            clip_value=1e2
        )

        # Test with large values
        large_input = ops.ones((2, 4)) * 100.0
        output_large = layer(large_input)

        # Test with small values
        small_input = ops.ones((2, 4)) * 1e-6
        output_small = layer(small_input)

        # Should handle without NaN/Inf
        assert not ops.any(ops.isnan(output_large))
        assert not ops.any(ops.isnan(output_small))
        assert not ops.any(ops.isinf(output_large))
        assert not ops.any(ops.isinf(output_small))

    def test_single_sample(self) -> None:
        """Test with single sample inputs."""
        layer = KANLinear(in_features=6, out_features=4)

        single_input = keras.random.normal([1, 6])
        output = layer(single_input)

        assert output.shape == (1, 4)
        assert not ops.any(ops.isnan(output))

    def test_large_batch(self) -> None:
        """Test with large batch sizes."""
        layer = KANLinear(in_features=8, out_features=5)

        large_batch_input = keras.random.normal([1000, 8])
        output = layer(large_batch_input)

        assert output.shape == (1000, 5)
        assert not ops.any(ops.isnan(output))

    def test_numerical_stability_different_ranges(self) -> None:
        """Test numerical stability with different grid ranges."""
        ranges_to_test = [
            (-1.0, 1.0),
            (-10.0, 10.0),
            (-0.1, 0.1),
            (0.0, 1.0),
            (-5.0, 0.0)
        ]

        for grid_range in ranges_to_test:
            layer = KANLinear(
                in_features=3,
                out_features=2,
                grid_range=grid_range
            )

            test_input = keras.random.normal([4, 3])
            output = layer(test_input)

            assert not ops.any(ops.isnan(output))
            assert not ops.any(ops.isinf(output))


class TestModelSaveLoad:
    """Test model saving and loading with KAN layers."""

    def test_model_save_load_with_kan(self) -> None:
        """Test saving and loading a model with KAN layers."""
        # Create a model with KAN layers
        kan_config = [
            {"in_features": 8, "out_features": 12, "grid_size": 6},
            {"in_features": 12, "out_features": 6, "grid_size": 5},
            {"in_features": 6, "out_features": 3, "grid_size": 4}
        ]

        inputs = keras.Input(shape=(8,))
        x = inputs

        # Add KAN layers
        for config in kan_config:
            kan_layer = KANLinear(**config)
            x = kan_layer(x)

        # Add final dense layer
        outputs = keras.layers.Dense(1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Generate dummy data and train briefly
        x_train = keras.random.normal([100, 8])
        y_train = keras.random.uniform([100, 1]) > 0.5
        y_train = ops.cast(y_train, dtype="float32")

        model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Generate predictions before saving
        x_test = keras.random.normal([20, 8])
        original_predictions = model.predict(x_test, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "kan_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "KANLinear": KANLinear
                }
            )

            # Generate predictions with loaded model
            loaded_predictions = loaded_model.predict(x_test, verbose=0)

            # Predictions should match
            np.testing.assert_allclose(
                original_predictions,
                loaded_predictions,
                rtol=1e-5
            )

            logger.info("KAN model save/load test passed successfully")

    def test_kan_model_save_load(self) -> None:
        """Test saving and loading KAN model directly."""
        config = [
            {"in_features": 5, "out_features": 8, "grid_size": 6},
            {"in_features": 8, "out_features": 3, "grid_size": 4}
        ]

        original_model = KAN(
            layers_configurations=config,
            enable_debugging=True,
            name="test_kan_model"
        )

        # Generate test data
        x_test = keras.random.normal([10, 5])
        original_output = original_model(x_test)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "kan_direct.keras")

            original_model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "KAN": KAN,
                    "KANLinear": KANLinear
                }
            )

            loaded_output = loaded_model(x_test)

            # Check that the loaded model produces valid outputs
            assert loaded_output.shape == original_output.shape
            assert not ops.any(ops.isnan(loaded_output))
            assert not ops.any(ops.isinf(loaded_output))

            # Check that the model architecture is preserved
            assert len(loaded_model.kan_layers) == len(original_model.kan_layers)
            assert loaded_model.enable_debugging == original_model.enable_debugging

            # Check that the layers have the correct configurations
            for orig_layer, loaded_layer in zip(original_model.kan_layers, loaded_model.kan_layers):
                assert orig_layer.in_features == loaded_layer.in_features
                assert orig_layer.out_features == loaded_layer.out_features
                assert orig_layer.grid_size == loaded_layer.grid_size

            logger.info("KAN model save/load test passed successfully")


class TestPerformance:
    """Test performance characteristics of KAN layers."""

    def test_memory_usage_different_grid_sizes(self) -> None:
        """Test memory usage with different grid sizes."""
        grid_sizes = [3, 5, 10, 20]

        for grid_size in grid_sizes:
            layer = KANLinear(
                in_features=10,
                out_features=5,
                grid_size=grid_size
            )

            test_input = keras.random.normal([32, 10])
            output = layer(test_input)

            # Should complete without memory errors
            assert output.shape == (32, 5)
            assert not ops.any(ops.isnan(output))

    def test_computational_complexity(self) -> None:
        """Test computational behavior with increasing complexity."""
        complexities = [
            {"in_features": 5, "out_features": 5, "grid_size": 5},
            {"in_features": 20, "out_features": 15, "grid_size": 8},
            {"in_features": 50, "out_features": 30, "grid_size": 12},
        ]

        for config in complexities:
            layer = KANLinear(**config)

            batch_size = 64
            test_input = keras.random.normal([batch_size, config["in_features"]])
            output = layer(test_input)

            expected_shape = (batch_size, config["out_features"])
            assert output.shape == expected_shape
            assert not ops.any(ops.isnan(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])