"""
Comprehensive test suite for the modern KAN model implementation.

Tests cover:
- Basic functionality and initialization using modern interface
- Enhanced validation and error handling
- New utility methods and variant creation
- Functional API architecture and serialization
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
from dl_techniques.models.kan.model import KAN, create_kan_model
from dl_techniques.utils.logger import logger


class TestKAN:
    """Test suite for modern KAN model implementation."""

    @pytest.fixture
    def simple_layer_configs(self) -> List[Dict[str, Any]]:
        """Create simple KAN layer configurations."""
        return [
            {"features": 8, "grid_size": 5, "activation": "swish"},
            {"features": 4, "grid_size": 4, "activation": "gelu"},
            {"features": 2, "grid_size": 3, "activation": "linear"}
        ]

    @pytest.fixture
    def complex_layer_configs(self) -> List[Dict[str, Any]]:
        """Create complex KAN layer configurations."""
        return [
            {
                "features": 20, "grid_size": 8,
                "activation": "swish", "spline_order": 3
            },
            {
                "features": 12, "grid_size": 6,
                "activation": "gelu", "spline_order": 2
            },
            {
                "features": 5, "grid_size": 4,
                "activation": "linear"
            }
        ]

    @pytest.fixture
    def simple_input_tensor(self) -> keras.KerasTensor:
        """Create test input tensor for simple config."""
        return keras.random.normal([16, 10])

    @pytest.fixture
    def complex_input_tensor(self) -> keras.KerasTensor:
        """Create test input tensor for complex config."""
        return keras.random.normal([16, 15])

    def test_initialization_simple(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test initialization with simple configuration."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        assert model.built is True  # Modern functional API builds immediately
        assert model.layer_configs == simple_layer_configs
        assert model.input_features == 10
        assert model.num_layers == 3
        assert "kan_model" in model.name

        # Check that KANLinear layers were created
        kan_layers = [l for l in model.layers if isinstance(l, KANLinear)]
        assert len(kan_layers) == 3

    def test_initialization_with_complex_config(self, complex_layer_configs: List[Dict[str, Any]]) -> None:
        """Test initialization with complex configuration."""
        model = KAN(
            layer_configs=complex_layer_configs,
            input_features=15
        )

        assert len(model.layer_configs) == 3
        assert model.layer_configs[0]["spline_order"] == 3
        assert model.layer_configs[1]["spline_order"] == 2

    def test_model_built_on_initialization(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test that model is built during initialization (functional API)."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        # Model should be built immediately with functional API
        assert model.built
        assert model.input_shape == (None, 10)
        assert model.output_shape == (None, 2)

        # Check that KANLinear layers exist and are built
        kan_layers = [l for l in model.layers if isinstance(l, KANLinear)]
        assert len(kan_layers) == 3
        for layer in kan_layers:
            assert layer.built

    def test_forward_pass_immediate(
        self,
        simple_layer_configs: List[Dict[str, Any]],
        simple_input_tensor: keras.KerasTensor
    ) -> None:
        """Test that model works immediately after creation."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        # Should work immediately since model is built
        output = model(simple_input_tensor)
        assert output.shape == (16, 2)

    def test_invalid_configurations(self) -> None:
        """Test that invalid configurations raise appropriate errors."""
        # Empty configuration
        with pytest.raises(ValueError, match="must be a non-empty list"):
            KAN(layer_configs=[], input_features=10)

        # Non-list configuration
        with pytest.raises(ValueError, match="must be a non-empty list"):
            KAN(layer_configs="invalid", input_features=10)

        # Missing input_features
        with pytest.raises(ValueError, match="must be positive integer"):
            KAN(layer_configs=[{"features": 5}], input_features=0)

        # Invalid input_features type
        with pytest.raises(ValueError, match="must be positive integer"):
            KAN(layer_configs=[{"features": 5}], input_features="invalid")

        # Missing required 'features' key
        with pytest.raises(ValueError, match="missing required 'features' key"):
            KAN(layer_configs=[{"grid_size": 5}], input_features=10)

        # Invalid features value
        # Note: This error comes from KANLinear validation during build
        with pytest.raises(ValueError, match="Features must be a positive integer"):
            KAN(layer_configs=[{"features": -1}], input_features=10)

        # Invalid grid_size (caught by KANLinear)
        with pytest.raises((ValueError, Exception)):
            KAN(layer_configs=[{"features": 5, "grid_size": 0}], input_features=10)

    def test_call_with_wrong_input_shape(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test calling model with wrong input shape."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        # Wrong number of features
        with pytest.raises((ValueError, Exception)):
            invalid_input = keras.random.normal((16, 5))  # Expected 10 features
            model(invalid_input)

    def test_forward_pass(
        self,
        simple_layer_configs: List[Dict[str, Any]],
        simple_input_tensor: keras.KerasTensor
    ) -> None:
        """Test forward pass through KAN model."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        output = model(simple_input_tensor)

        # Check output shape and validity
        assert output.shape == (16, 2)  # Final layer has 2 features
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_forward_pass_complex(
        self,
        complex_layer_configs: List[Dict[str, Any]],
        complex_input_tensor: keras.KerasTensor
    ) -> None:
        """Test forward pass with complex configuration."""
        model = KAN(
            layer_configs=complex_layer_configs,
            input_features=15
        )

        output = model(complex_input_tensor)

        # Check output shape and validity
        assert output.shape == (16, 5)  # Final layer has 5 features
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_empty_batch_handling(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test handling of empty batches."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        # Create empty batch
        empty_input = ops.zeros((0, 10))
        output = model(empty_input)

        # Should return appropriately shaped empty tensor
        assert output.shape == (0, 2)
        assert output.dtype == empty_input.dtype

    def test_compute_output_shape(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test output shape computation."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        # Test various input shapes
        input_shapes = [
            (None, 10),
            (32, 10),
            (1, 10),
            (100, 10)
        ]

        for input_shape in input_shapes:
            output_shape = model.compute_output_shape(input_shape)
            expected_shape = input_shape[:-1] + (2,)  # Last layer has 2 features
            assert output_shape == expected_shape

    def test_get_architecture_summary(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test architecture summary functionality."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

        summary = model.get_architecture_summary()

        assert "KAN Model Architecture Summary" in summary
        assert "Layer  0:" in summary
        assert "Layer  1:" in summary
        assert "Layer  2:" in summary
        assert "Est. Parameters:" in summary
        assert "Total layers: 3" in summary
        # Check flow string
        assert "Flow: 10 -> 8 -> 4 -> 2" in summary

    def test_model_compilation_and_training(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test KAN model compilation and basic training."""
        model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10
        )

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

    def test_serialization_config_only(self, simple_layer_configs: List[Dict[str, Any]]) -> None:
        """Test KAN model get_config and from_config methods."""
        original_model = KAN(
            layer_configs=simple_layer_configs,
            input_features=10,
        )

        # Get config and recreate
        config = original_model.get_config()
        recreated_model = KAN.from_config(config)

        # Check configuration matches
        assert recreated_model.layer_configs == original_model.layer_configs
        assert recreated_model.input_features == original_model.input_features
        assert recreated_model.built == original_model.built

        # Test that recreated model has the same architecture properties
        assert recreated_model.input_shape == original_model.input_shape
        assert recreated_model.output_shape == original_model.output_shape

    def test_different_architectures(self) -> None:
        """Test KAN with different architectures."""
        architectures = [
            # Single layer
            {"layer_configs": [{"features": 3}], "input_features": 5},

            # Deep network
            {
                "layer_configs": [
                    {"features": 20, "activation": "swish"},
                    {"features": 15, "activation": "gelu"},
                    {"features": 10, "activation": "relu"},
                    {"features": 5, "activation": "tanh"},
                    {"features": 1, "activation": "linear"}
                ],
                "input_features": 10
            },

            # Wide network
            {
                "layer_configs": [
                    {"features": 32, "grid_size": 10},
                    {"features": 4, "grid_size": 8}
                ],
                "input_features": 8
            }
        ]

        for arch_config in architectures:
            model = KAN(**arch_config)

            # Test with appropriate input
            input_features = arch_config["input_features"]
            output_features = arch_config["layer_configs"][-1]["features"]

            test_input = keras.random.normal([8, input_features])
            output = model(test_input)

            assert output.shape == (8, output_features)
            assert not ops.any(ops.isnan(output))


class TestKANVariants:
    """Test KAN variant creation methods."""

    def test_from_variant_classification(self) -> None:
        """Test creating KAN for classification (multiclass)."""
        model = KAN.from_variant(
            variant="small",
            input_features=784,
            output_features=10
        )

        assert model.input_features == 784
        assert model.layer_configs[-1]["features"] == 10
        # Should default to softmax for >1 output features
        assert model.layer_configs[-1]["activation"] == "softmax"
        assert len(model.layer_configs) == 4  # [64, 32, 16] + [10]

        # Test forward pass
        test_input = keras.random.normal([16, 784])
        output = model(test_input)
        assert output.shape == (16, 10)

    def test_from_variant_regression_defaults(self) -> None:
        """Test that single output defaults to linear activation (regression)."""
        model = KAN.from_variant(
            variant="micro",
            input_features=10,
            output_features=1
        )

        # Check defaults for scalar output
        assert model.layer_configs[-1]["features"] == 1
        assert model.layer_configs[-1]["activation"] == "linear"

    def test_from_variant_binary_classification(self) -> None:
        """Test explicit binary classification (sigmoid)."""
        model = KAN.from_variant(
            variant="micro",
            input_features=10,
            output_features=1,
            output_activation="sigmoid"
        )

        assert model.layer_configs[-1]["features"] == 1
        assert model.layer_configs[-1]["activation"] == "sigmoid"

    def test_from_variant_with_overrides(self) -> None:
        """Test variant creation with configuration overrides."""
        model = KAN.from_variant(
            variant="medium",
            input_features=256,
            output_features=50,
            override_config={"activation": "relu", "grid_size": 12}
        )

        # Check that overrides were applied to hidden layers
        for layer_config in model.layer_configs[:-1]:
            assert layer_config["grid_size"] == 12

        assert model.input_features == 256
        assert model.layer_configs[-1]["features"] == 50

    def test_from_variant_invalid(self) -> None:
        """Test invalid variant name."""
        with pytest.raises(ValueError, match="Unknown variant"):
            KAN.from_variant(
                variant="invalid_variant",
                input_features=100,
                output_features=10
            )

    def test_from_layer_sizes(self) -> None:
        """Test creating KAN from layer sizes."""
        layer_sizes = [784, 128, 64, 10]
        model = KAN.from_layer_sizes(
            layer_sizes=layer_sizes,
            grid_size=8,
            activation='gelu'
        )

        assert model.input_features == 784
        assert len(model.layer_configs) == 3  # 4 sizes = 3 layers

        # Check configurations
        assert model.layer_configs[0]["features"] == 128
        assert model.layer_configs[0]["grid_size"] == 8
        assert model.layer_configs[0]["activation"] == 'gelu'

        # Last layer logic (multiclass default)
        assert model.layer_configs[-1]["features"] == 10
        assert model.layer_configs[-1]["activation"] == "softmax"

        # Test forward pass
        test_input = keras.random.normal([16, 784])
        output = model(test_input)
        assert output.shape == (16, 10)

    def test_from_layer_sizes_with_kwargs(self) -> None:
        """Test creating KAN with additional kwargs."""
        model = KAN.from_layer_sizes(
            layer_sizes=[100, 50, 10],
            grid_size=6,
            spline_order=4,
        )

        # Check that kwargs were passed through
        config = model.layer_configs[0]
        assert config["spline_order"] == 4

    def test_from_layer_sizes_invalid(self) -> None:
        """Test invalid layer sizes."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            KAN.from_layer_sizes(layer_sizes=[10])

        with pytest.raises(ValueError, match="at least 2 elements"):
            KAN.from_layer_sizes(layer_sizes=[])


class TestCreateKanModelHelper:
    """Test the create_kan_model convenience function."""

    def test_create_standard_model(self) -> None:
        """Test creating a standard KAN model (uncompiled)."""
        model = create_kan_model(
            variant="small",
            input_features=784,
            output_features=10
        )

        # Check architecture
        assert isinstance(model, KAN)
        assert model.input_features == 784
        assert model.layer_configs[-1]["features"] == 10

    def test_create_model_with_explicit_activation(self) -> None:
        """Test creating model with specific output activation."""
        model = create_kan_model(
            variant="medium",
            input_features=256,
            output_features=1,
            output_activation="sigmoid" # Binary classification
        )

        assert model.input_features == 256
        assert model.layer_configs[-1]["features"] == 1
        assert model.layer_configs[-1]["activation"] == "sigmoid"

        # Test training loop integration
        model.compile(optimizer='adam', loss='binary_crossentropy')

        x_test = keras.random.normal([16, 256])
        y_test = keras.random.uniform([16, 1])

        loss = model.evaluate(x_test, y_test, verbose=0)
        assert isinstance(loss, (float, list))


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_single_sample(self) -> None:
        """Test with single sample inputs."""
        model = KAN(
            layer_configs=[{"features": 4}],
            input_features=6
        )

        single_input = keras.random.normal([1, 6])
        output = model(single_input)

        assert output.shape == (1, 4)
        assert not ops.any(ops.isnan(output))

    def test_large_batch(self) -> None:
        """Test with large batch sizes."""
        model = KAN(
            layer_configs=[{"features": 5}],
            input_features=8
        )

        large_batch_input = keras.random.normal([1000, 8])
        output = model(large_batch_input)

        assert output.shape == (1000, 5)
        assert not ops.any(ops.isnan(output))

    def test_extreme_values(self) -> None:
        """Test with extreme input values."""
        model = KAN(
            layer_configs=[{"features": 2}],
            input_features=4
        )

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
            model = KAN(
                layer_configs=[{"features": 2, "grid_range": grid_range}],
                input_features=3
            )

            test_input = keras.random.normal([4, 3])
            output = model(test_input)

            assert not ops.any(ops.isnan(output))
            assert not ops.any(ops.isinf(output))


class TestModelSaveLoad:
    """Test model saving and loading."""

    def test_model_save_load_with_kan(self) -> None:
        """Test saving and loading a hybrid model with KAN layers."""
        inputs = keras.Input(shape=(8,))

        # Add KAN layers
        x = KANLinear(features=12, grid_size=6, activation="swish", name="kan_1")(inputs)
        x = KANLinear(features=6, grid_size=5, activation="gelu", name="kan_2")(x)

        # Add final dense layer
        outputs = keras.layers.Dense(1, activation='sigmoid', name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy'
        )

        # Generate predictions before saving
        x_test = keras.random.normal([20, 8])
        original_predictions = model.predict(x_test, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "kan_hybrid.keras")

            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"KANLinear": KANLinear}
            )

            loaded_predictions = loaded_model.predict(x_test, verbose=0)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_predictions),
                ops.convert_to_numpy(loaded_predictions),
                rtol=1e-5, atol=1e-6
            )

    def test_kan_model_save_load_direct(self) -> None:
        """Test saving and loading KAN model directly."""
        original_model = KAN(
            layer_configs=[
                {"features": 8, "grid_size": 6, "activation": "swish"},
                {"features": 3, "grid_size": 4, "activation": "linear"}
            ],
            input_features=5,
            name="test_kan_model"
        )

        x_test = keras.random.normal([10, 5])
        original_output = original_model(x_test)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "kan_direct.keras")
            original_model.save(model_path)

            # Load model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"KAN": KAN, "KANLinear": KANLinear}
            )

            loaded_output = loaded_model(x_test)

            # Check outputs and architecture
            assert loaded_output.shape == original_output.shape
            assert loaded_model.input_features == original_model.input_features
            assert loaded_model.layer_configs == original_model.layer_configs


    def test_variant_model_save_load(self) -> None:
        """Test saving and loading model created with from_variant."""
        original_model = KAN.from_variant(
            variant="small",
            input_features=100,
            output_features=10
        )

        x_test = keras.random.normal([8, 100])
        original_output = original_model(x_test)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "variant_kan.keras")
            original_model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"KAN": KAN, "KANLinear": KANLinear}
            )

            loaded_output = loaded_model(x_test)
            assert loaded_output.shape == original_output.shape
            assert loaded_model.input_features == original_model.input_features


class TestPerformance:
    """Test performance characteristics."""

    def test_memory_usage_different_grid_sizes(self) -> None:
        """Test memory usage with different grid sizes."""
        grid_sizes = [3, 5, 10, 20]

        for grid_size in grid_sizes:
            model = KAN(
                layer_configs=[{"features": 5, "grid_size": grid_size}],
                input_features=10
            )
            test_input = keras.random.normal([32, 10])
            output = model(test_input)
            assert output.shape == (32, 5)

    def test_computational_complexity_scaling(self) -> None:
        """Test computational behavior with increasing complexity."""
        complexities = [
            {
                "layer_configs": [{"features": 5, "grid_size": 5}],
                "input_features": 5
            },
            {
                "layer_configs": [
                    {"features": 15, "grid_size": 8},
                    {"features": 10, "grid_size": 6}
                ],
                "input_features": 20
            }
        ]

        for config in complexities:
            model = KAN(**config)

            batch_size = 64
            input_features = config["input_features"]
            output_features = config["layer_configs"][-1]["features"]

            test_input = keras.random.normal([batch_size, input_features])
            output = model(test_input)

            assert output.shape == (batch_size, output_features)
            assert not ops.any(ops.isnan(output))

    def test_parameter_estimation_accuracy(self) -> None:
        """Test that parameter estimation is reasonable."""
        model = KAN(
            layer_configs=[
                {"features": 64, "grid_size": 5, "spline_order": 3},
                {"features": 32, "grid_size": 5, "spline_order": 3},
                {"features": 10, "grid_size": 5, "spline_order": 3}
            ],
            input_features=128
        )

        estimated_params = model._estimate_parameters()
        actual_params = model.count_params()

        # Estimation should be within reasonable bounds (0.5x to 2.0x)
        ratio = estimated_params / actual_params
        assert 0.5 <= ratio <= 2.0, f"Ratio {ratio} outside bounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])