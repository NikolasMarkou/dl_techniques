"""
Tests for ValueRangeConstraint

This module contains comprehensive tests for the ValueRangeConstraint class:
- Initialization with default and custom parameters
- Constraint application with various input scenarios
- Edge cases and numerical stability
- Serialization and deserialization
- Model integration scenarios

Tests cover proper clipping behavior, validation, and integration with Keras layers.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tempfile
import os
from typing import Tuple

from dl_techniques.utils.logger import logger

# Import the constraint to test (adjust import path as needed)
from dl_techniques.constraints.value_range_constraint import ValueRangeConstraint


class TestValueRangeConstraint:
    """Test suite for ValueRangeConstraint implementation."""

    @pytest.fixture
    def sample_weights(self) -> keras.KerasTensor:
        """Create sample weight tensor for testing.

        Returns:
            keras.KerasTensor: Sample weight tensor with values in range [-2, 2].
        """
        weights = np.random.uniform(-2.0, 2.0, size=(10, 5))
        return ops.cast(weights, dtype="float32")

    @pytest.fixture
    def extreme_weights(self) -> keras.KerasTensor:
        """Create weight tensor with extreme values for testing.

        Returns:
            keras.KerasTensor: Weight tensor with very large and small values.
        """
        weights = np.array([
            [-1000.0, -10.0, -1.0, 0.0, 1.0],
            [2.0, 10.0, 100.0, 1000.0, 1e6],
            [-1e6, -0.001, 0.001, 999.0, -999.0]
        ])
        return ops.cast(weights, dtype="float32")

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        constraint = ValueRangeConstraint(min_value=0.0)

        assert constraint.min_value == 0.0
        assert constraint.max_value is None
        assert constraint.clip_gradients is True

    def test_initialization_custom(self) -> None:
        """Test initialization with custom parameters."""
        constraint = ValueRangeConstraint(
            min_value=-1.0,
            max_value=1.0,
            clip_gradients=False
        )

        assert constraint.min_value == -1.0
        assert constraint.max_value == 1.0
        assert constraint.clip_gradients is False

    def test_invalid_parameters(self) -> None:
        """Test that invalid parameters raise appropriate errors."""
        # Test min_value > max_value
        with pytest.raises(ValueError, match="min_value .* cannot be greater than max_value"):
            ValueRangeConstraint(min_value=1.0, max_value=0.5)

        # Test min_value == max_value (should be valid)
        constraint = ValueRangeConstraint(min_value=1.0, max_value=1.0)
        assert constraint.min_value == 1.0
        assert constraint.max_value == 1.0

    def test_clip_gradients_parameter(self) -> None:
        """Test clip_gradients parameter is properly stored and accessible."""
        # Test default value
        constraint1 = ValueRangeConstraint(min_value=0.0)
        assert constraint1.clip_gradients is True

        # Test custom value
        constraint2 = ValueRangeConstraint(min_value=0.0, clip_gradients=False)
        assert constraint2.clip_gradients is False

        # The constraint should work the same regardless of clip_gradients value
        # since clipping is inherent to the constraint operation
        weights = ops.cast([[-1.0, 2.0]], dtype="float32")

        result1 = constraint1(weights)
        result2 = constraint2(weights)

        # Both should clip to [0.0, 2.0] (min_value=0.0, no max_value)
        expected = ops.cast([[0.0, 2.0]], dtype="float32")
        assert ops.all(ops.isclose(result1, expected))
        assert ops.all(ops.isclose(result2, expected))

    def test_constraint_min_only(self, sample_weights: keras.KerasTensor) -> None:
        """Test constraint with only minimum value specified."""
        constraint = ValueRangeConstraint(min_value=0.0)
        constrained_weights = constraint(sample_weights)

        # All values should be >= 0.0
        assert ops.all(constrained_weights >= 0.0)
        # Check no NaN or Inf values
        assert not ops.any(ops.isnan(constrained_weights))
        assert not ops.any(ops.isinf(constrained_weights))

    def test_constraint_min_max(self, sample_weights: keras.KerasTensor) -> None:
        """Test constraint with both minimum and maximum values."""
        constraint = ValueRangeConstraint(min_value=-1.0, max_value=1.0)
        constrained_weights = constraint(sample_weights)

        # All values should be in range [-1.0, 1.0]
        assert ops.all(constrained_weights >= -1.0)
        assert ops.all(constrained_weights <= 1.0)
        # Check no NaN or Inf values
        assert not ops.any(ops.isnan(constrained_weights))
        assert not ops.any(ops.isinf(constrained_weights))

    def test_constraint_extreme_values(self, extreme_weights: keras.KerasTensor) -> None:
        """Test constraint with extreme input values."""
        constraint = ValueRangeConstraint(min_value=-10.0, max_value=10.0)
        constrained_weights = constraint(extreme_weights)

        # All values should be clipped to [-10.0, 10.0]
        assert ops.all(constrained_weights >= -10.0)
        assert ops.all(constrained_weights <= 10.0)
        # Check no NaN or Inf values
        assert not ops.any(ops.isnan(constrained_weights))
        assert not ops.any(ops.isinf(constrained_weights))

    def test_no_clipping_needed(self) -> None:
        """Test constraint when no clipping is needed."""
        # Create weights already within range
        weights = ops.cast([[0.5, -0.3, 0.8], [0.1, -0.9, 0.0]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=-1.0, max_value=1.0)

        constrained_weights = constraint(weights)

        # Values should remain unchanged
        assert ops.all(ops.isclose(weights, constrained_weights, atol=1e-7))

    def test_min_clipping_only(self) -> None:
        """Test constraint that only clips minimum values."""
        weights = ops.cast([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=-0.5, max_value=3.0)

        constrained_weights = constraint(weights)
        expected = ops.cast([[-0.5, -0.5, 0.0, 1.0, 2.0]], dtype="float32")

        assert ops.all(ops.isclose(constrained_weights, expected, atol=1e-6))

    def test_max_clipping_only(self) -> None:
        """Test constraint that only clips maximum values."""
        weights = ops.cast([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=-3.0, max_value=0.5)

        constrained_weights = constraint(weights)
        expected = ops.cast([[-2.0, -1.0, 0.0, 0.5, 0.5]], dtype="float32")

        assert ops.all(ops.isclose(constrained_weights, expected, atol=1e-6))

    def test_both_clipping(self) -> None:
        """Test constraint that clips both minimum and maximum values."""
        weights = ops.cast([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=-0.5, max_value=0.5)

        constrained_weights = constraint(weights)
        expected = ops.cast([[-0.5, -0.5, 0.0, 0.5, 0.5]], dtype="float32")

        assert ops.all(ops.isclose(constrained_weights, expected, atol=1e-6))

    def test_zero_range_constraint(self) -> None:
        """Test constraint with min_value == max_value."""
        weights = ops.cast([[-1.0, 0.0, 1.0, 2.0]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=0.5, max_value=0.5)

        constrained_weights = constraint(weights)
        expected = ops.cast([[0.5, 0.5, 0.5, 0.5]], dtype="float32")

        assert ops.all(ops.isclose(constrained_weights, expected, atol=1e-6))

    def test_serialization(self) -> None:
        """Test serialization and deserialization of the constraint."""
        original_constraint = ValueRangeConstraint(
            min_value=-2.5,
            max_value=3.7,
            clip_gradients=False
        )

        # Get config and recreate
        config = original_constraint.get_config()
        recreated_constraint = ValueRangeConstraint.from_config(config)

        # Check configuration matches
        assert recreated_constraint.min_value == original_constraint.min_value
        assert recreated_constraint.max_value == original_constraint.max_value
        assert recreated_constraint.clip_gradients == original_constraint.clip_gradients

    def test_serialization_min_only(self) -> None:
        """Test serialization with only minimum value specified."""
        original_constraint = ValueRangeConstraint(min_value=0.1, clip_gradients=False)

        # Get config and recreate
        config = original_constraint.get_config()
        recreated_constraint = ValueRangeConstraint.from_config(config)

        # Check configuration matches
        assert recreated_constraint.min_value == original_constraint.min_value
        assert recreated_constraint.max_value is None
        assert recreated_constraint.clip_gradients == original_constraint.clip_gradients

    def test_layer_integration(self, sample_weights: keras.KerasTensor) -> None:
        """Test the constraint in a layer context."""
        # Create a layer with the constraint
        constraint = ValueRangeConstraint(min_value=0.0, max_value=1.0)

        layer = keras.layers.Dense(
            units=5,
            kernel_constraint=constraint,
            input_shape=(sample_weights.shape[-1],)
        )

        # Build the layer
        layer.build(sample_weights.shape)

        # Check that constraint is applied
        assert layer.kernel_constraint == constraint

        # Test forward pass
        dummy_input = ops.ones((1, sample_weights.shape[-1]))
        output = layer(dummy_input)

        # Check output is valid
        assert not ops.any(ops.isnan(output))
        assert not ops.any(ops.isinf(output))

    def test_model_integration(self) -> None:
        """Test the constraint in a model context."""
        # Create a model with constrained layers
        model = keras.Sequential([
            keras.layers.Dense(
                units=32,
                activation="relu",
                kernel_constraint=ValueRangeConstraint(min_value=0.0, max_value=2.0),
                input_shape=(10,)
            ),
            keras.layers.Dense(
                units=16,
                activation="relu",
                kernel_constraint=ValueRangeConstraint(min_value=-1.0, max_value=1.0)
            ),
            keras.layers.Dense(
                units=1,
                activation="sigmoid"
            )
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss="binary_crossentropy"
        )

        # Generate dummy data
        x_train = np.random.random((100, 10))
        y_train = np.random.choice([0, 1], size=(100, 1))

        # Train for a few steps to ensure constraints are applied
        history = model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=0)

        # Check that training completed without errors
        assert len(history.history['loss']) == 2
        assert not any(np.isnan(loss) for loss in history.history['loss'])

        # Check that weights are within constraints after training
        first_layer_weights = model.layers[0].get_weights()[0]  # kernel weights
        second_layer_weights = model.layers[1].get_weights()[0]  # kernel weights

        # First layer should have weights in [0.0, 2.0]
        assert np.all(first_layer_weights >= 0.0)
        assert np.all(first_layer_weights <= 2.0)

        # Second layer should have weights in [-1.0, 1.0]
        assert np.all(second_layer_weights >= -1.0)
        assert np.all(second_layer_weights <= 1.0)

    def test_model_save_load_with_constraint(self) -> None:
        """Test saving and loading a model with the constraint."""
        # Create a simple model with the constraint
        model = keras.Sequential([
            keras.layers.Dense(
                units=16,
                activation="relu",
                kernel_constraint=ValueRangeConstraint(min_value=-0.5, max_value=0.5),
                bias_constraint=ValueRangeConstraint(min_value=0.0),
                input_shape=(8,)
            ),
            keras.layers.Dense(units=1, activation="sigmoid")
        ])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="binary_crossentropy"
        )

        # Generate test data
        x_test = np.random.random((10, 8))
        original_predictions = model.predict(x_test, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "constrained_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"ValueRangeConstraint": ValueRangeConstraint}
            )

            # Generate predictions with loaded model
            loaded_predictions = loaded_model.predict(x_test, verbose=0)

            # Predictions should match
            np.testing.assert_allclose(
                original_predictions,
                loaded_predictions,
                rtol=1e-5
            )

            # Check that constraints are preserved
            loaded_constraint = loaded_model.layers[0].kernel_constraint
            assert isinstance(loaded_constraint, ValueRangeConstraint)
            assert loaded_constraint.min_value == -0.5
            assert loaded_constraint.max_value == 0.5

            logger.info("Model save/load test with ValueRangeConstraint passed successfully")

    def test_string_representation(self) -> None:
        """Test string representation of the constraint."""
        # Test with both min and max
        constraint1 = ValueRangeConstraint(min_value=-1.0, max_value=1.0, clip_gradients=False)
        repr1 = repr(constraint1)
        assert "ValueRangeConstraint" in repr1
        assert "min_value=-1.0" in repr1
        assert "max_value=1.0" in repr1
        assert "clip_gradients=False" in repr1

        # Test with min only
        constraint2 = ValueRangeConstraint(min_value=0.0, clip_gradients=True)
        repr2 = repr(constraint2)
        assert "ValueRangeConstraint" in repr2
        assert "min_value=0.0" in repr2
        assert "clip_gradients=True" in repr2
        assert "max_value" not in repr2


class TestEdgeCases:
    """Test edge cases and numerical stability for ValueRangeConstraint."""

    def test_empty_tensors(self) -> None:
        """Test constraint with empty tensors."""
        empty_weights = ops.cast(np.array([]).reshape(0, 0), dtype="float32")
        constraint = ValueRangeConstraint(min_value=0.0, max_value=1.0)

        result = constraint(empty_weights)

        # Should handle empty tensors gracefully
        assert result.shape == empty_weights.shape
        assert not ops.any(ops.isnan(result))

    def test_single_value_tensor(self) -> None:
        """Test constraint with single value tensor."""
        single_weight = ops.cast([[5.0]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=0.0, max_value=1.0)

        result = constraint(single_weight)
        expected = ops.cast([[1.0]], dtype="float32")

        assert ops.all(ops.isclose(result, expected))

    def test_very_small_differences(self) -> None:
        """Test constraint with very small numerical differences."""
        # Values very close to boundaries
        weights = ops.cast([[1e-10, 1.0 - 1e-10, 1.0 + 1e-10]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=0.0, max_value=1.0)

        result = constraint(weights)

        # Should handle small differences properly
        assert ops.all(result >= 0.0)
        assert ops.all(result <= 1.0)
        assert not ops.any(ops.isnan(result))

    def test_inf_and_nan_inputs(self) -> None:
        """Test constraint behavior with inf and nan inputs."""
        # Test with inf values
        inf_weights = ops.cast([[-np.inf, np.inf, 0.0]], dtype="float32")
        constraint = ValueRangeConstraint(min_value=-1.0, max_value=1.0)

        result = constraint(inf_weights)

        # Should clip inf values to bounds
        assert ops.all(ops.isfinite(result) | (result == -1.0) | (result == 1.0))

        # Note: NaN behavior is typically undefined for constraints,
        # so we don't test it explicitly as it's backend-dependent

    def test_large_tensor_performance(self) -> None:
        """Test constraint performance with large tensors."""
        # Create a large tensor
        large_weights = ops.cast(np.random.uniform(-10, 10, size=(1000, 1000)), dtype="float32")
        constraint = ValueRangeConstraint(min_value=-1.0, max_value=1.0)

        # This should complete without memory errors
        result = constraint(large_weights)

        # Verify constraint is applied correctly
        assert ops.all(result >= -1.0)
        assert ops.all(result <= 1.0)
        assert result.shape == large_weights.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])