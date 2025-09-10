"""
Comprehensive test suite for the refined KAN model implementation.

Tests cover:
- Basic functionality and initialization
- Enhanced validation and error handling
- New utility methods and features
- Build system and serialization
- Edge cases and numerical stability
- Performance characteristics
- Model saving and loading
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import List, Dict, Any

from dl_techniques.layers.kan_linear import KANLinear
from dl_techniques.models.kan.model import KAN, create_kan_from_sizes
from dl_techniques.utils.logger import logger


class TestKAN:
    """Test suite for refined KAN model implementation."""

    @pytest.fixture
    def simple_config(self) -> List[Dict[str, Any]]:
        """Create a simple KAN configuration.

        Returns:
            List of layer configurations for a simple KAN.
        """
        return [
            {"in_features": 10, "out_features": 8, "grid_size": 5, "activation": "swish"},
            {"in_features": 8, "out_features": 4, "grid_size": 4, "activation": "gelu"},
            {"in_features": 4, "out_features": 2, "grid_size": 3, "activation": "linear"}
        ]

    @pytest.fixture
    def complex_config(self) -> List[Dict[str, Any]]:
        """Create a more complex KAN configuration.

        Returns:
            List of layer configurations for a complex KAN.
        """
        return [
            {
                "in_features": 15, "out_features": 20, "grid_size": 8,
                "activation": "swish", "spline_order": 3
            },
            {
                "in_features": 20, "out_features": 12, "grid_size": 6,
                "activation": "gelu", "spline_order": 2
            },
            {
                "in_features": 12, "out_features": 5, "grid_size": 4,
                "activation": "linear"
            }
        ]

    @pytest.fixture
    def input_tensor(self) -> keras.KerasTensor:
        """Create a test input tensor.

        Returns:
            Random input tensor for testing.
        """
        return keras.random.normal([16, 10])

    @pytest.fixture
    def complex_input_tensor(self) -> keras.KerasTensor:
        """Create a test input tensor for complex config.

        Returns:
            Random input tensor for complex configuration testing.
        """
        return keras.random.normal([16, 15])

    def test_initialization_simple(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test initialization with simple configuration."""
        model = KAN(layers_configurations=simple_config)

        assert model.built is True
        assert model.enable_debugging is False
        assert model.layers_configurations == simple_config
        assert model.name == "kan_model"  # Default name

        kan_layers = [l for l in model.layers if isinstance(l, KANLinear)]
        assert len(kan_layers) == 3

    def test_initialization_with_debugging(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test initialization with debugging enabled."""
        model = KAN(
            layers_configurations=simple_config,
            enable_debugging=True,
            name="test_kan"
        )

        assert model.enable_debugging is True
        assert model.name == "test_kan"
        assert len(model.layers_configurations) == 3

    def test_initialization_with_complex_config(self, complex_config: List[Dict[str, Any]]) -> None:
        """Test initialization with complex configuration."""
        model = KAN(layers_configurations=complex_config, enable_debugging=True)

        assert len(model.layers_configurations) == 3
        assert model.layers_configurations[0]["spline_order"] == 3
        assert model.layers_configurations[1]["spline_order"] == 2

    def test_build_method(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test that the model is built correctly on initialization."""
        model = KAN(layers_configurations=simple_config)

        # The model is built on initialization with the Functional API
        assert model.built

        kan_layers = [l for l in model.layers if isinstance(l, KANLinear)]
        assert len(kan_layers) == 3

        # Check that layers are properly created
        assert kan_layers[0].in_features == 10
        assert kan_layers[0].out_features == 8
        assert kan_layers[1].in_features == 8
        assert kan_layers[1].out_features == 4
        assert kan_layers[2].in_features == 4
        assert kan_layers[2].out_features == 2

    def test_auto_build_on_call(self, simple_config: List[Dict[str, Any]], input_tensor: keras.KerasTensor) -> None:
        """Test that model is built on init and callable."""
        model = KAN(layers_configurations=simple_config)

        # Model should be built after initialization
        assert model.built

        # Calling the model should work and produce the correct output shape
        output = model(input_tensor)

        kan_layers = [l for l in model.layers if isinstance(l, KANLinear)]
        assert len(kan_layers) == 3
        assert output.shape == (16, 2)

    def test_invalid_configurations(self) -> None:
        """Test that invalid configurations raise appropriate errors."""
        # Empty configuration
        with pytest.raises(ValueError, match="Layer configurations cannot be empty"):
            KAN(layers_configurations=[])

        # Non-list configuration
        with pytest.raises(ValueError, match="Layer configurations must be a list"):
            KAN(layers_configurations="invalid")

        # Missing required keys
        with pytest.raises(ValueError, match="Layer .* missing required key"):
            KAN(layers_configurations=[{"grid_size": 5}])

        # Invalid feature values
        with pytest.raises(ValueError, match="must be a positive integer"):
            KAN(layers_configurations=[{"in_features": -1, "out_features": 5}])

        with pytest.raises(ValueError, match="must be a positive integer"):
            KAN(layers_configurations=[{"in_features": 5, "out_features": 0}])

        # Invalid grid_size
        with pytest.raises(ValueError, match="'grid_size' must be an integer >= 3"):
            KAN(layers_configurations=[{"in_features": 5, "out_features": 3, "grid_size": 2}])

        # Invalid spline_order
        with pytest.raises(ValueError, match="'spline_order' must be a positive integer"):
            KAN(layers_configurations=[{"in_features": 5, "out_features": 3, "spline_order": 0}])

        # Incompatible layer dimensions
        incompatible_config = [
            {"in_features": 10, "out_features": 8},
            {"in_features": 6, "out_features": 4}  # Should be in_features=8
        ]
        with pytest.raises(ValueError, match="Incompatible layer dimensions"):
            KAN(layers_configurations=incompatible_config)

    def test_call_with_invalid_input_shape(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test calling the model with an invalid input shape."""
        model = KAN(layers_configurations=simple_config)

        # Incompatible input features
        with pytest.raises(ValueError, match="incompatible with the layer"):
            invalid_input = keras.random.normal((16, 5))  # Expected 10 features
            model(invalid_input)

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

    def test_forward_pass_complex(
            self,
            complex_config: List[Dict[str, Any]],
            complex_input_tensor: keras.KerasTensor
    ) -> None:
        """Test forward pass with complex configuration."""
        model = KAN(layers_configurations=complex_config)

        output = model(complex_input_tensor)

        # Check output shape and validity
        assert output.shape == (16, 5)  # Final layer has 5 outputs
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
        output = model(input_tensor, training=True)
        assert output.shape == (16, 2)

        # Debugging only logs to console, no stats are returned
        stats = model.get_layer_statistics()
        assert stats == {}

    def test_empty_batch_handling(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test handling of empty batches."""
        model = KAN(layers_configurations=simple_config)

        # Create empty batch
        empty_input = ops.zeros((0, 10))
        output = model(empty_input)

        # Should return appropriately shaped empty tensor
        assert output.shape == (0, 2)
        assert output.dtype == empty_input.dtype

    def test_compute_output_shape(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test output shape computation."""
        model = KAN(layers_configurations=simple_config)

        # Test various input shapes
        input_shapes = [
            (None, 10),
            (32, 10),
            (1, 10),
            (100, 10)
        ]

        for input_shape in input_shapes:
            output_shape = model.compute_output_shape(input_shape)
            expected_shape = input_shape[:-1] + (2,)  # Last layer has 2 outputs
            assert output_shape == expected_shape

    def test_get_architecture_summary(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test architecture summary functionality."""
        model = KAN(layers_configurations=simple_config, enable_debugging=True)

        summary = model.get_architecture_summary()

        assert "KAN Model Architecture:" in summary
        assert "Layer  0:" in summary
        assert "Layer  1:" in summary
        assert "Layer  2:" in summary
        assert "Total parameters" in summary
        assert "Number of layers: 3" in summary
        assert "Debugging enabled: True" in summary

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

    def test_serialization_config_only(self, simple_config: List[Dict[str, Any]]) -> None:
        """Test KAN model get_config and from_config methods."""
        original_model = KAN(
            layers_configurations=simple_config,
            enable_debugging=True,
        )

        # Get config and recreate
        config = original_model.get_config()
        recreated_model = KAN.from_config(config)

        # Check configuration matches
        assert recreated_model.layers_configurations == original_model.layers_configurations
        assert recreated_model.enable_debugging == original_model.enable_debugging
        assert recreated_model.built == original_model.built

        # Test that recreated model has the same architecture properties
        assert recreated_model.input_shape == original_model.input_shape
        assert recreated_model.output_shape == original_model.output_shape

    def test_different_architectures(self) -> None:
        """Test KAN with different architectures."""
        architectures = [
            # Single layer
            [{"in_features": 5, "out_features": 3}],

            # Deep network
            [
                {"in_features": 10, "out_features": 20, "activation": "swish"},
                {"in_features": 20, "out_features": 15, "activation": "gelu"},
                {"in_features": 15, "out_features": 10, "activation": "relu"},
                {"in_features": 10, "out_features": 5, "activation": "tanh"},
                {"in_features": 5, "out_features": 1, "activation": "linear"}
            ],

            # Wide network
            [
                {"in_features": 8, "out_features": 32, "grid_size": 10},
                {"in_features": 32, "out_features": 4, "grid_size": 8}
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


class TestCreateKANFromSizes:
    """Test the create_kan_from_sizes convenience function."""

    def test_create_simple_model(self) -> None:
        """Test creating a simple KAN model with uniform configuration."""
        layer_sizes = [10, 20, 15, 5]
        model = create_kan_from_sizes(
            layer_sizes=layer_sizes,
            grid_size=8,
            activation='gelu',
            enable_debugging=True
        )

        assert len(model.layers_configurations) == 3  # 4 sizes = 3 layers
        assert model.enable_debugging is True

        # Check layer configurations
        assert model.layers_configurations[0]["in_features"] == 10
        assert model.layers_configurations[0]["out_features"] == 20
        assert model.layers_configurations[0]["grid_size"] == 8
        assert model.layers_configurations[0]["activation"] == 'gelu'

        assert model.layers_configurations[1]["in_features"] == 20
        assert model.layers_configurations[1]["out_features"] == 15

        assert model.layers_configurations[2]["in_features"] == 15
        assert model.layers_configurations[2]["out_features"] == 5

    def test_create_model_with_additional_kwargs(self) -> None:
        """Test creating KAN model with additional KANLinear arguments."""
        layer_sizes = [5, 8, 3]
        model = create_kan_from_sizes(
            layer_sizes=layer_sizes,
            grid_size=6,
            spline_order=4,
            use_residual=False
        )

        # Check that additional kwargs were passed through
        config = model.layers_configurations[0]
        assert config["spline_order"] == 4
        assert config["use_residual"] is False

    def test_create_model_invalid_layer_sizes(self) -> None:
        """Test creating KAN model with invalid layer sizes."""
        # Too few layer sizes
        with pytest.raises(ValueError, match="layer_sizes must have at least 2 elements"):
            create_kan_from_sizes(layer_sizes=[10])

        # Empty layer sizes
        with pytest.raises(ValueError, match="layer_sizes must have at least 2 elements"):
            create_kan_from_sizes(layer_sizes=[])

    def test_create_model_functional_test(self) -> None:
        """Test that created model works functionally."""
        model = create_kan_from_sizes(
            layer_sizes=[784, 128, 64, 10],
            grid_size=8,
            activation='swish'
        )

        # Test forward pass
        test_input = keras.random.normal([32, 784])
        output = model(test_input)

        assert output.shape == (32, 10)
        assert not ops.any(ops.isnan(output))


class TestEdgeCases:
    """Test edge cases and numerical stability for refined KAN model."""

    def test_single_sample(self) -> None:
        """Test with single sample inputs."""
        config = [{"in_features": 6, "out_features": 4}]
        model = KAN(layers_configurations=config)

        single_input = keras.random.normal([1, 6])
        output = model(single_input)

        assert output.shape == (1, 4)
        assert not ops.any(ops.isnan(output))

    def test_large_batch(self) -> None:
        """Test with large batch sizes."""
        config = [{"in_features": 8, "out_features": 5}]
        model = KAN(layers_configurations=config)

        large_batch_input = keras.random.normal([1000, 8])
        output = model(large_batch_input)

        assert output.shape == (1000, 5)
        assert not ops.any(ops.isnan(output))

    def test_extreme_values(self) -> None:
        """Test with extreme input values."""
        config = [
            {
                "in_features": 4, "out_features": 2,
            }
        ]
        model = KAN(layers_configurations=config)

        # Test with large values
        large_input = ops.ones((2, 4)) * 100.0
        output_large = model(large_input)

        # Test with small values
        small_input = ops.ones((2, 4)) * 1e-6
        output_small = model(small_input)

        # Should handle without NaN/Inf
        assert not ops.any(ops.isnan(output_large))
        assert not ops.any(ops.isnan(output_small))
        assert not ops.any(ops.isinf(output_large))
        assert not ops.any(ops.isinf(output_small))

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
            config = [
                {
                    "in_features": 3, "out_features": 2,
                    "grid_range": grid_range
                }
            ]
            model = KAN(layers_configurations=config)

            test_input = keras.random.normal([4, 3])
            output = model(test_input)

            assert not ops.any(ops.isnan(output))
            assert not ops.any(ops.isinf(output))

    def test_debugging_with_problematic_values(self) -> None:
        """Test debugging mode with problematic input values."""
        config = [{"in_features": 3, "out_features": 2}]
        model = KAN(layers_configurations=config, enable_debugging=True)

        # Create input with some extreme values
        problematic_input = keras.random.normal([8, 3])
        # Add some large values
        problematic_input = ops.where(
            keras.random.uniform([8, 3]) > 0.9,
            ops.ones_like(problematic_input) * 1000.0,
            problematic_input
        )

        # Should handle gracefully and log warnings
        output = model(problematic_input, training=True)
        assert output.shape == (8, 2)

        # Check that statistics are empty as they are not collected
        stats = model.get_layer_statistics()
        assert stats == {}


class TestModelSaveLoad:
    """Test model saving and loading with refined KAN layers."""

    def test_model_save_load_with_kan(self) -> None:
        """Test saving and loading a model with KAN layers."""
        # Create a model with KAN layers
        kan_config = [
            {"in_features": 8, "out_features": 12, "grid_size": 6, "activation": "swish"},
            {"in_features": 12, "out_features": 6, "grid_size": 5, "activation": "gelu"},
            {"in_features": 6, "out_features": 3, "grid_size": 4, "activation": "relu"}
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

    def test_kan_model_save_load_direct(self) -> None:
        """Test saving and loading KAN model directly."""
        config = [
            {"in_features": 5, "out_features": 8, "grid_size": 6, "activation": "swish"},
            {"in_features": 8, "out_features": 3, "grid_size": 4, "activation": "linear"}
        ]

        original_model = KAN(
            layers_configurations=config,
            enable_debugging=True,
            name="test_kan_model"
        )

        # Generate test data and build model
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
            original_kan_layers = [l for l in original_model.layers if isinstance(l, KANLinear)]
            loaded_kan_layers = [l for l in loaded_model.layers if isinstance(l, KANLinear)]
            assert len(loaded_kan_layers) == len(original_kan_layers)
            assert loaded_model.enable_debugging == original_model.enable_debugging

            # Check that the layers have the correct configurations
            for orig_layer, loaded_layer in zip(original_kan_layers, loaded_kan_layers):
                assert orig_layer.in_features == loaded_layer.in_features
                assert orig_layer.out_features == loaded_layer.out_features
                assert orig_layer.grid_size == loaded_layer.grid_size

            logger.info("KAN model direct save/load test passed successfully")

    def test_create_kan_model_save_load(self) -> None:
        """Test saving and loading model created with create_kan_from_sizes."""
        model = create_kan_from_sizes(
            layer_sizes=[10, 15, 8, 3],
            grid_size=7,
            activation='gelu',
            enable_debugging=True
        )

        # Test data
        x_test = keras.random.normal([5, 10])
        original_output = model(x_test)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "created_kan.keras")

            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    "KAN": KAN,
                    "KANLinear": KANLinear
                }
            )

            loaded_output = loaded_model(x_test)

            # Check outputs
            assert loaded_output.shape == original_output.shape
            assert not ops.any(ops.isnan(loaded_output))

            # Check architecture is preserved
            assert len(loaded_model.layers_configurations) == 3
            assert loaded_model.enable_debugging is True


class TestPerformance:
    """Test performance characteristics of refined KAN model."""

    def test_memory_usage_different_grid_sizes(self) -> None:
        """Test memory usage with different grid sizes."""
        grid_sizes = [3, 5, 10, 20]

        for grid_size in grid_sizes:
            config = [
                {
                    "in_features": 10, "out_features": 5,
                    "grid_size": grid_size
                }
            ]
            model = KAN(layers_configurations=config)

            test_input = keras.random.normal([32, 10])
            output = model(test_input)

            # Should complete without memory errors
            assert output.shape == (32, 5)
            assert not ops.any(ops.isnan(output))

    def test_computational_complexity_scaling(self) -> None:
        """Test computational behavior with increasing complexity."""
        complexities = [
            [{"in_features": 5, "out_features": 5, "grid_size": 5}],
            [
                {"in_features": 20, "out_features": 15, "grid_size": 8},
                {"in_features": 15, "out_features": 10, "grid_size": 6}
            ],
            [
                {"in_features": 50, "out_features": 40, "grid_size": 12},
                {"in_features": 40, "out_features": 30, "grid_size": 10},
                {"in_features": 30, "out_features": 20, "grid_size": 8}
            ],
        ]

        for config in complexities:
            model = KAN(layers_configurations=config, enable_debugging=True)

            batch_size = 64
            input_features = config[0]["in_features"]
            output_features = config[-1]["out_features"]

            test_input = keras.random.normal([batch_size, input_features])
            output = model(test_input, training=True)

            expected_shape = (batch_size, output_features)
            assert output.shape == expected_shape
            assert not ops.any(ops.isnan(output))

            # Check that statistics were collected in debugging mode
            stats = model.get_layer_statistics()
            assert stats == {}

    def test_deep_network_stability(self) -> None:
        """Test stability with very deep KAN networks."""
        # Create a deep network
        layer_sizes = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
        model = create_kan_from_sizes(
            layer_sizes=layer_sizes,
            grid_size=5,
            activation='swish',
        )

        test_input = keras.random.normal([16, 20])
        output = model(test_input)

        assert output.shape == (16, 2)
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

        # Check architecture summary for deep network
        summary = model.get_architecture_summary()
        assert "Number of layers: 9" in summary

    def test_wide_network_performance(self) -> None:
        """Test performance with wide KAN networks."""
        # Create a wide network
        config = [
            {"in_features": 100, "out_features": 200, "grid_size": 8},
            {"in_features": 200, "out_features": 150, "grid_size": 6},
            {"in_features": 150, "out_features": 50, "grid_size": 5}
        ]
        model = KAN(layers_configurations=config)

        test_input = keras.random.normal([32, 100])
        output = model(test_input)

        assert output.shape == (32, 50)
        assert not ops.any(ops.isnan(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
