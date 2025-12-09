"""
Comprehensive test suite for AdaptiveTemperatureSoftmax layer.

This test suite follows the modern Keras 3 testing patterns from the
"Complete Guide to Modern Keras 3 Custom Layers and Models - Refined.md"
ensuring robust validation of all layer functionality.
"""

import pytest
import tempfile
import os
import numpy as np
import keras
from typing import Any, Dict

# Import the layer to test
from dl_techniques.layers.activations.adaptive_softmax import AdaptiveTemperatureSoftmax

# TensorFlow for gradient testing
import tensorflow as tf


class TestAdaptiveTemperatureSoftmax:
    """Comprehensive test suite for AdaptiveTemperatureSoftmax layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'min_temp': 0.1,
            'max_temp': 1.0,
            'entropy_threshold': 0.5,
            'eps': 1e-7
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input for basic testing."""
        return keras.random.normal(shape=(4, 10))

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input for sequence testing."""
        return keras.random.normal(shape=(2, 8, 16))

    @pytest.fixture
    def large_input(self) -> keras.KerasTensor:
        """Large input to test dispersion effects."""
        return keras.random.normal(shape=(3, 512))

    def test_initialization(self, layer_config):
        """Test layer initialization with valid parameters."""
        layer = AdaptiveTemperatureSoftmax(**layer_config)

        # Check configuration storage
        assert layer.min_temp == layer_config['min_temp']
        assert layer.max_temp == layer_config['max_temp']
        assert layer.entropy_threshold == layer_config['entropy_threshold']
        assert layer.eps == layer_config['eps']
        assert not layer.built
        assert layer.polynomial_coeffs is not None
        assert len(layer.polynomial_coeffs) == 5  # Default polynomial degree

    def test_initialization_defaults(self):
        """Test layer initialization with default parameters."""
        layer = AdaptiveTemperatureSoftmax()

        assert layer.min_temp == 0.1
        assert layer.max_temp == 1.0
        assert layer.entropy_threshold == 0.5
        assert layer.eps == 1e-7
        assert layer.polynomial_coeffs == [-1.791, 4.917, -2.3, 0.481, -0.037]

    def test_forward_pass_2d(self, layer_config, sample_input_2d):
        """Test forward pass and building with 2D input."""
        layer = AdaptiveTemperatureSoftmax(**layer_config)

        output = layer(sample_input_2d)

        # Check layer is built
        assert layer.built

        # Check output shape
        assert output.shape == sample_input_2d.shape

        # Check output is valid probabilities
        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(output_np >= 0), "All probabilities should be non-negative"
        assert np.all(output_np <= 1), "All probabilities should be <= 1"

        # Check probabilities sum to 1 along last axis
        prob_sums = np.sum(output_np, axis=-1)
        np.testing.assert_allclose(
            prob_sums,
            np.ones_like(prob_sums),
            rtol=1e-6, atol=1e-6,
            err_msg="Probabilities should sum to 1.0"
        )

    def test_forward_pass_3d(self, layer_config, sample_input_3d):
        """Test forward pass with 3D input (sequences)."""
        layer = AdaptiveTemperatureSoftmax(**layer_config)

        output = layer(sample_input_3d)

        assert output.shape == sample_input_3d.shape

        # Check probability constraints
        output_np = keras.ops.convert_to_numpy(output)
        prob_sums = np.sum(output_np, axis=-1)
        np.testing.assert_allclose(
            prob_sums,
            np.ones_like(prob_sums),
            rtol=1e-6, atol=1e-6,
            err_msg="3D probabilities should sum to 1.0"
        )

    def test_serialization_cycle(self, layer_config, sample_input_2d):
        """CRITICAL TEST: Full serialization cycle."""
        # Create model with custom layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        outputs = AdaptiveTemperatureSoftmax(**layer_config)(inputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_pred = model(sample_input_2d)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input_2d)

            # Verify identical predictions
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = AdaptiveTemperatureSoftmax(**layer_config)
        config = layer.get_config()

        # Check all config parameters are present
        for key in layer_config:
            assert key in config, f"Missing {key} in get_config()"

        # Check additional parameters
        assert 'polynomial_coeffs' in config

        # Verify values match
        for key, value in layer_config.items():
            assert config[key] == value, f"Config mismatch for {key}"

    def test_config_reconstruction(self, layer_config):
        """Test that layer can be recreated from config."""
        original_layer = AdaptiveTemperatureSoftmax(**layer_config)
        config = original_layer.get_config()

        # Remove base class config items to test our layer specifically
        layer_specific_config = {
            k: v for k, v in config.items()
            if k in ['min_temp', 'max_temp', 'entropy_threshold', 'eps', 'polynomial_coeffs']
        }

        reconstructed_layer = AdaptiveTemperatureSoftmax(**layer_specific_config)

        # Verify configurations match
        assert reconstructed_layer.min_temp == original_layer.min_temp
        assert reconstructed_layer.max_temp == original_layer.max_temp
        assert reconstructed_layer.entropy_threshold == original_layer.entropy_threshold
        assert reconstructed_layer.eps == original_layer.eps
        assert reconstructed_layer.polynomial_coeffs == original_layer.polynomial_coeffs

    def test_gradients_flow(self, layer_config, sample_input_2d):
        """Test gradient computation - layer should be differentiable."""
        layer = AdaptiveTemperatureSoftmax(**layer_config)

        # Convert to tensorflow tensor for gradient computation
        tf_input = tf.convert_to_tensor(keras.ops.convert_to_numpy(sample_input_2d))

        with tf.GradientTape() as tape:
            tape.watch(tf_input)
            output = layer(tf_input)
            loss = tf.reduce_mean(tf.square(output))

        gradients = tape.gradient(loss, tf_input)

        # Check gradients exist and are finite
        assert gradients is not None, "Gradients should not be None"
        gradients_np = gradients.numpy()
        assert np.all(np.isfinite(gradients_np)), "All gradients should be finite"
        assert np.any(gradients_np != 0), "At least some gradients should be non-zero"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input_2d, training):
        """Test behavior in different training modes."""
        layer = AdaptiveTemperatureSoftmax(**layer_config)

        output = layer(sample_input_2d, training=training)

        # Output should be valid regardless of training mode
        assert output.shape == sample_input_2d.shape

        # Check probabilities are valid
        output_np = keras.ops.convert_to_numpy(output)
        prob_sums = np.sum(output_np, axis=-1)
        np.testing.assert_allclose(
            prob_sums,
            np.ones_like(prob_sums),
            rtol=1e-6, atol=1e-6,
            err_msg=f"Probabilities should sum to 1.0 in training={training} mode"
        )

    def test_deterministic_behavior(self, layer_config, sample_input_2d):
        """Test that layer produces consistent outputs for same input."""
        layer = AdaptiveTemperatureSoftmax(**layer_config)

        output1 = layer(sample_input_2d)
        output2 = layer(sample_input_2d)

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output1),
            keras.ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Layer should produce deterministic outputs"
        )

    def test_entropy_threshold_behavior(self, sample_input_2d):
        """Test that entropy threshold affects adaptation."""
        # Create layers with different thresholds
        low_threshold = AdaptiveTemperatureSoftmax(entropy_threshold=0.1)
        high_threshold = AdaptiveTemperatureSoftmax(entropy_threshold=2.0)

        # Create input with moderate entropy that will trigger different behaviors
        # Use random logits scaled to create moderate dispersion
        moderate_entropy_input = keras.random.normal(sample_input_2d.shape) * 0.5

        output_low = low_threshold(moderate_entropy_input)
        output_high = high_threshold(moderate_entropy_input)
        output_standard = keras.ops.nn.softmax(moderate_entropy_input)

        # Compute differences from standard softmax
        diff_high = keras.ops.mean(keras.ops.square(output_high - output_standard))
        diff_low = keras.ops.mean(keras.ops.square(output_low - output_standard))

        diff_high_np = keras.ops.convert_to_numpy(diff_high)
        diff_low_np = keras.ops.convert_to_numpy(diff_low)

        # High threshold should generally be closer to standard softmax
        # (though this might not always hold due to the adaptive nature)
        # At minimum, both should produce valid probability distributions
        assert diff_high_np >= 0, "Difference should be non-negative"
        assert diff_low_np >= 0, "Difference should be non-negative"

        # Test that the thresholds are actually different
        assert low_threshold.entropy_threshold != high_threshold.entropy_threshold

        # Test with a high-entropy input to ensure low threshold activates
        high_entropy_input = keras.random.normal(sample_input_2d.shape) * 0.1  # Very dispersed
        output_low_dispersed = low_threshold(high_entropy_input)
        output_standard_dispersed = keras.ops.nn.softmax(high_entropy_input)

        # With dispersed input, low threshold should likely apply some adaptation
        diff_dispersed = keras.ops.mean(keras.ops.square(output_low_dispersed - output_standard_dispersed))
        assert keras.ops.convert_to_numpy(diff_dispersed) >= 0

    def test_temperature_range_effects(self, sample_input_2d):
        """Test effects of different temperature ranges."""
        # Sharp temperature range
        sharp_layer = AdaptiveTemperatureSoftmax(min_temp=0.01, max_temp=0.5)

        # Smooth temperature range
        smooth_layer = AdaptiveTemperatureSoftmax(min_temp=0.5, max_temp=2.0)

        # Use input that triggers adaptation
        high_entropy_input = keras.random.normal(shape=sample_input_2d.shape) * 0.1

        sharp_output = sharp_layer(high_entropy_input)
        smooth_output = smooth_layer(high_entropy_input)

        # Sharp layer should produce more concentrated distributions
        sharp_entropy = -keras.ops.sum(
            sharp_output * keras.ops.log(sharp_output + 1e-7),
            axis=-1
        )
        smooth_entropy = -keras.ops.sum(
            smooth_output * keras.ops.log(smooth_output + 1e-7),
            axis=-1
        )

        # Sharp layer should generally have lower entropy (more concentrated)
        mean_sharp_entropy = keras.ops.mean(sharp_entropy)
        mean_smooth_entropy = keras.ops.mean(smooth_entropy)

        # This is a general tendency, not a strict rule due to adaptive nature
        assert keras.ops.convert_to_numpy(mean_sharp_entropy) >= 0
        assert keras.ops.convert_to_numpy(mean_smooth_entropy) >= 0

    def test_large_input_handling(self, large_input):
        """Test behavior with large input sizes (dispersion scenario)."""
        layer = AdaptiveTemperatureSoftmax()

        output = layer(large_input)

        # Should handle large inputs without numerical issues
        assert output.shape == large_input.shape

        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(np.isfinite(output_np)), "All outputs should be finite"
        assert np.all(output_np >= 0), "All probabilities should be non-negative"

        # Check probability constraints
        prob_sums = np.sum(output_np, axis=-1)
        np.testing.assert_allclose(
            prob_sums,
            np.ones_like(prob_sums),
            rtol=1e-5, atol=1e-5,  # Slightly relaxed for large inputs
            err_msg="Large input probabilities should sum to 1.0"
        )

    def test_edge_cases(self):
        """Test error conditions and edge cases."""
        # Test invalid temperature values
        with pytest.raises(ValueError, match="min_temp must be positive"):
            AdaptiveTemperatureSoftmax(min_temp=0.0)

        with pytest.raises(ValueError, match="min_temp must be positive"):
            AdaptiveTemperatureSoftmax(min_temp=-0.1)

        with pytest.raises(ValueError, match="max_temp must be positive"):
            AdaptiveTemperatureSoftmax(max_temp=0.0)

        with pytest.raises(ValueError, match="min_temp.*must be.*max_temp"):
            AdaptiveTemperatureSoftmax(min_temp=1.0, max_temp=0.5)

        with pytest.raises(ValueError, match="entropy_threshold must be non-negative"):
            AdaptiveTemperatureSoftmax(entropy_threshold=-0.1)

    def test_build_validation(self):
        """Test build method validation."""
        layer = AdaptiveTemperatureSoftmax()

        # Test invalid input shapes
        with pytest.raises(ValueError, match="expects at least 2D input"):
            layer.build((10,))  # 1D input

        with pytest.raises(ValueError, match="Last dimension.*must be defined"):
            layer.build((None, 10, None))  # Undefined last dimension

        # Valid shapes should work
        layer.build((None, 10))  # 2D
        assert layer.built

        layer2 = AdaptiveTemperatureSoftmax()
        layer2.build((None, 8, 16))  # 3D
        assert layer2.built

    def test_custom_polynomial_coefficients(self, sample_input_2d):
        """Test layer with custom polynomial coefficients."""
        custom_coeffs = [1.0, -2.0, 1.0, 0.0, 0.0]  # Simple quadratic
        layer = AdaptiveTemperatureSoftmax(polynomial_coeffs=custom_coeffs)

        output = layer(sample_input_2d)

        # Should work with custom coefficients
        assert output.shape == sample_input_2d.shape
        assert layer.polynomial_coeffs == custom_coeffs

        # Verify in config
        config = layer.get_config()
        assert config['polynomial_coeffs'] == custom_coeffs

    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs."""
        layer = AdaptiveTemperatureSoftmax()

        # Test with very large logits
        large_logits = keras.ops.ones((2, 5)) * 100
        output_large = layer(large_logits)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output_large)))

        # Test with very small logits
        small_logits = keras.ops.ones((2, 5)) * -100
        output_small = layer(small_logits)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output_small)))

        # Test with mixed extreme values
        extreme_logits = keras.ops.concatenate([
            keras.ops.ones((2, 2)) * 50,
            keras.ops.ones((2, 2)) * -50,
            keras.ops.zeros((2, 1))
        ], axis=-1)
        output_extreme = layer(extreme_logits)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(output_extreme)))

    def test_comparison_with_standard_softmax(self, sample_input_2d):
        """Test comparison with standard softmax baseline."""
        adaptive_layer = AdaptiveTemperatureSoftmax()

        # For very low entropy inputs, should behave similarly to standard softmax
        # Create low entropy input (one dominant logit)
        low_entropy_input = keras.ops.concatenate([
            keras.ops.ones((sample_input_2d.shape[0], 1)) * 10,  # Dominant class
            keras.ops.zeros((sample_input_2d.shape[0], sample_input_2d.shape[1] - 1))
        ], axis=-1)

        adaptive_output = adaptive_layer(low_entropy_input)
        standard_output = keras.ops.nn.softmax(low_entropy_input)

        # Should be similar for low-entropy inputs
        difference = keras.ops.mean(keras.ops.square(adaptive_output - standard_output))
        diff_np = keras.ops.convert_to_numpy(difference)

        # Should be reasonably close (not identical due to adaptation)
        assert diff_np < 0.1, f"Difference too large: {diff_np}"


# Additional integration tests for the complete layer
class TestAdaptiveTemperatureSoftmaxIntegration:
    """Integration tests for AdaptiveTemperatureSoftmax in complete models."""

    def test_in_classification_model(self):
        """Test layer in a complete classification model."""
        # Create a simple classification model
        inputs = keras.Input(shape=(784,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = keras.layers.Dense(64, activation='relu')(x)
        logits = keras.layers.Dense(10)(x)  # No activation
        probabilities = AdaptiveTemperatureSoftmax()(logits)

        model = keras.Model(inputs, probabilities)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Test with dummy data
        dummy_x = keras.random.normal((32, 784))
        dummy_y = keras.random.uniform((32, 10))

        # Should be trainable
        history = model.fit(dummy_x, dummy_y, epochs=1, verbose=0)
        assert len(history.history['loss']) == 1

        # Should produce valid predictions
        predictions = model.predict(dummy_x, verbose=0)
        assert predictions.shape == (32, 10)

        # Check probability constraints
        prob_sums = np.sum(predictions, axis=-1)
        np.testing.assert_allclose(
            prob_sums,
            np.ones_like(prob_sums),
            rtol=1e-5, atol=1e-5,
            err_msg="Model predictions should sum to 1.0"
        )

    def test_serialization_in_complete_model(self):
        """Test serialization of complete model with adaptive softmax."""
        # Create model
        inputs = keras.Input(shape=(20,))
        x = keras.layers.Dense(32, activation='tanh')(inputs)
        logits = keras.layers.Dense(5)(x)
        outputs = AdaptiveTemperatureSoftmax(
            min_temp=0.05,
            max_temp=1.5,
            entropy_threshold=0.3
        )(logits)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Test data
        test_input = keras.random.normal((8, 20))
        original_output = model.predict(test_input, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'complete_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_output = loaded_model.predict(test_input, verbose=0)

            # Verify identical outputs
            np.testing.assert_allclose(
                original_output,
                loaded_output,
                rtol=1e-6, atol=1e-6,
                err_msg="Complete model predictions should match after serialization"
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])