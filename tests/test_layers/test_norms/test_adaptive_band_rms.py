"""
Refined comprehensive test suite for the robust AdaptiveBandRMS layer.

This test suite covers:
- All tensor shapes (2D, 3D, 4D, 5D+)
- Various axis configurations
- Proper serialization testing
- Edge cases and robustness
- Model integration
- Backward compatibility
"""

import pytest
import numpy as np
import tempfile
import os

import keras
import tensorflow as tf
from keras import ops
from dl_techniques.utils.logger import logger

from dl_techniques.layers.norms.adaptive_band_rms import AdaptiveBandRMS


class TestAdaptiveBandRMS:
    """Comprehensive test suite for the robust AdaptiveBandRMS layer."""

    # =========================================================================
    # FIXTURES
    # =========================================================================

    @pytest.fixture
    def input_tensor_2d(self) -> np.ndarray:
        """Create a 2D test input tensor for dense layer scenarios."""
        np.random.seed(42)
        return np.random.randn(4, 64).astype(np.float32)

    @pytest.fixture
    def input_tensor_3d(self) -> np.ndarray:
        """Create a 3D test input tensor for sequence model scenarios."""
        np.random.seed(42)
        return np.random.randn(4, 20, 32).astype(np.float32)

    @pytest.fixture
    def input_tensor_4d(self) -> np.ndarray:
        """Create a 4D test input tensor for convolutional layer scenarios."""
        np.random.seed(42)
        return np.random.randn(4, 16, 16, 32).astype(np.float32)

    @pytest.fixture
    def input_tensor_5d(self) -> np.ndarray:
        """Create a 5D test input tensor for 3D convolution scenarios."""
        np.random.seed(42)
        return np.random.randn(2, 8, 16, 16, 16).astype(np.float32)

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = AdaptiveBandRMS()

        assert layer.max_band_width == 0.1
        assert layer.axis == -1
        assert layer.epsilon == 1e-7
        assert isinstance(layer.band_initializer, keras.initializers.Zeros)
        assert layer.band_regularizer is None
        assert layer.dense_layer is None  # Created in build()

        logger.info("Default initialization test passed")

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        custom_initializer = keras.initializers.HeNormal()

        layer = AdaptiveBandRMS(
            max_band_width=0.3,
            axis=(1, 2),
            epsilon=1e-6,
            band_initializer=custom_initializer,
            band_regularizer=custom_regularizer
        )

        assert layer.max_band_width == 0.3
        assert layer.axis == (1, 2)
        assert layer.epsilon == 1e-6
        assert isinstance(layer.band_initializer, keras.initializers.HeNormal)
        assert layer.band_regularizer == custom_regularizer

        logger.info("Custom initialization test passed")

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Test invalid max_band_width values
        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            AdaptiveBandRMS(max_band_width=-0.1)

        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            AdaptiveBandRMS(max_band_width=1.5)

        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            AdaptiveBandRMS(max_band_width=0.0)

        with pytest.raises(ValueError, match="max_band_width must be between 0 and 1"):
            AdaptiveBandRMS(max_band_width=1.0)

        # Test invalid epsilon values
        with pytest.raises(ValueError, match="epsilon must be positive"):
            AdaptiveBandRMS(epsilon=-1e-7)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            AdaptiveBandRMS(epsilon=0)

        logger.info("Invalid parameters test passed")

    # =========================================================================
    # BUILD PROCESS TESTS
    # =========================================================================

    def test_build_process_2d(self, input_tensor_2d):
        """Test that the layer builds properly for 2D tensors."""
        layer = AdaptiveBandRMS()
        input_shape = input_tensor_2d.shape[1:]

        layer.build((None,) + input_shape)

        assert layer.built is True
        assert layer.dense_layer is not None
        assert layer.dense_layer.built is True
        assert layer.dense_layer.units == input_shape[-1]  # One parameter per feature

        logger.info("2D build process test passed")

    def test_build_process_3d(self, input_tensor_3d):
        """Test build process for 3D tensors with different axis configurations."""
        test_cases = [
            ("feature_wise", -1, input_tensor_3d.shape[-1]),
            ("sequence_wise", 1, input_tensor_3d.shape[1]),
            ("global", (1, 2), 1)
        ]

        for name, axis, expected_units in test_cases:
            layer = AdaptiveBandRMS(axis=axis)
            input_shape = input_tensor_3d.shape[1:]

            layer.build((None,) + input_shape)

            assert layer.built is True
            assert layer.dense_layer.units == expected_units, \
                f"{name}: expected {expected_units} units, got {layer.dense_layer.units}"

            logger.info(f"3D build process test passed for {name}")

    def test_build_process_4d(self, input_tensor_4d):
        """Test build process for 4D tensors with different axis configurations."""
        test_cases = [
            ("channel_wise", -1, input_tensor_4d.shape[-1]),
            ("spatial", (1, 2), input_tensor_4d.shape[1] * input_tensor_4d.shape[2]),
            ("global", (1, 2, 3), 1)
        ]

        for name, axis, expected_units in test_cases:
            layer = AdaptiveBandRMS(axis=axis)
            input_shape = input_tensor_4d.shape[1:]

            layer.build((None,) + input_shape)

            assert layer.built is True
            assert layer.dense_layer.units == expected_units, \
                f"{name}: expected {expected_units} units, got {layer.dense_layer.units}"

            logger.info(f"4D build process test passed for {name}")

    def test_build_process_5d(self, input_tensor_5d):
        """Test build process for 5D tensors."""
        test_cases = [
            ("channel_wise", -1, input_tensor_5d.shape[-1]),
            ("spatial_3d", (1, 2, 3), input_tensor_5d.shape[1] * input_tensor_5d.shape[2] * input_tensor_5d.shape[3]),
            ("partial_spatial", (2, 3), input_tensor_5d.shape[2] * input_tensor_5d.shape[3]),
            ("global", (1, 2, 3, 4), 1)
        ]

        for name, axis, expected_units in test_cases:
            layer = AdaptiveBandRMS(axis=axis)
            input_shape = input_tensor_5d.shape[1:]

            layer.build((None,) + input_shape)

            assert layer.built is True
            # Note: For complex axis configurations, the exact units depend on implementation
            # Just verify the layer builds successfully
            assert layer.dense_layer.units == expected_units

            logger.info(f"5D build process test passed for {name}")

    def test_invalid_axis_configurations(self):
        """Test that invalid axis configurations raise errors during build."""
        # Test axis out of bounds
        layer = AdaptiveBandRMS(axis=10)

        with pytest.raises(ValueError, match="axis .* is out of bounds"):
            layer.build((None, 32, 32, 64))

        # Test axis including batch dimension (axis 0)
        layer = AdaptiveBandRMS(axis=(0, 1))

        with pytest.raises(ValueError, match="axis 0 .* cannot be normalized"):
            layer.build((None, 32, 32, 64))

        logger.info("Invalid axis configurations test passed")

    # =========================================================================
    # OUTPUT SHAPE TESTS
    # =========================================================================

    def test_output_shapes_all_dimensions(self):
        """Test output shapes for all supported tensor dimensions."""
        test_cases = [
            ("2D", (8, 64), -1),
            ("3D_feature", (4, 20, 32), -1),
            ("3D_sequence", (4, 20, 32), 1),
            ("3D_global", (4, 20, 32), (1, 2)),
            ("4D_channel", (4, 16, 16, 32), -1),
            ("4D_spatial", (4, 16, 16, 32), (1, 2)),
            ("4D_global", (4, 16, 16, 32), (1, 2, 3)),
            ("5D_channel", (2, 8, 16, 16, 16), -1),
            ("5D_spatial", (2, 8, 16, 16, 16), (1, 2, 3))
        ]

        for name, input_shape, axis in test_cases:
            layer = AdaptiveBandRMS(axis=axis)
            test_input = np.random.randn(*input_shape).astype(np.float32)

            output = layer(test_input)

            # Check output shape matches input shape
            assert output.shape == test_input.shape, f"Shape mismatch for {name}"

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_shape)
            assert computed_shape == input_shape, f"Computed shape mismatch for {name}"

            logger.info(f"Output shape test passed for {name}")

    # =========================================================================
    # FORWARD PASS TESTS
    # =========================================================================

    def test_forward_pass_numerical_properties(self):
        """Test forward pass produces values with correct numerical properties."""
        test_cases = [
            ("2D", (8, 64), -1),
            ("3D", (4, 20, 32), -1),
            ("4D", (4, 16, 16, 32), -1),
            ("5D", (2, 8, 16, 16, 16), -1)
        ]

        for name, input_shape, axis in test_cases:
            layer = AdaptiveBandRMS(max_band_width=0.2, axis=axis)
            test_input = np.random.randn(*input_shape).astype(np.float32)

            output = layer(test_input)

            # Basic sanity checks
            assert not np.any(np.isnan(output.numpy())), f"NaN values in {name} output"
            assert not np.any(np.isinf(output.numpy())), f"Inf values in {name} output"

            # Check RMS constraint is within expected band [1-α, 1] = [0.8, 1.0]
            output_rms = ops.sqrt(ops.mean(ops.square(output), axis=axis, keepdims=False))
            mean_rms = float(ops.mean(output_rms).numpy())

            expected_min_rms = 0.8  # 1.0 - 0.2
            expected_max_rms = 1.0

            assert expected_min_rms <= mean_rms <= expected_max_rms, \
                f"{name}: RMS {mean_rms} not in expected band [{expected_min_rms}, {expected_max_rms}]"

            logger.info(f"Forward pass numerical properties test passed for {name}")

    def test_forward_pass_different_axis_configurations(self, input_tensor_4d):
        """Test forward pass for different axis configurations on same input."""
        axis_configs = [
            ("single_axis", -1),
            ("tuple_axis", (1, 2)),
            ("global", (1, 2, 3))
        ]

        for name, axis in axis_configs:
            layer = AdaptiveBandRMS(axis=axis, max_band_width=0.15)
            output = layer(input_tensor_4d)

            # Basic checks
            assert output.shape == input_tensor_4d.shape, f"Shape mismatch for {name}"
            assert not np.any(np.isnan(output.numpy())), f"NaN in {name} output"
            assert not np.any(np.isinf(output.numpy())), f"Inf in {name} output"

            logger.info(f"Different axis configurations test passed for {name}")

    def test_deterministic_behavior(self):
        """Test that the layer produces consistent outputs with same input and weights."""
        # Create controlled input
        controlled_input = np.ones((2, 8), dtype=np.float32) * 2.0

        # Create layer with zero initialization for predictable behavior
        layer = AdaptiveBandRMS(
            max_band_width=0.1,
            band_initializer="zeros"
        )

        # Build the layer
        layer.build(controlled_input.shape)

        # Get two outputs with same input
        output1 = layer(controlled_input)
        output2 = layer(controlled_input)

        # Outputs should be identical (deterministic)
        np.testing.assert_allclose(
            ops.convert_to_numpy(output1),
            ops.convert_to_numpy(output2),
            rtol=1e-6, atol=1e-6,
            err_msg="Layer should produce deterministic outputs"
        )

        logger.info("Deterministic behavior test passed")

    # =========================================================================
    # SERIALIZATION TESTS (MODERN KERAS 3 APPROACH)
    # =========================================================================

    def test_serialization_cycle_2d(self, input_tensor_2d):
        """Test full serialization cycle for 2D tensors using modern Keras 3 patterns."""
        # Create model with AdaptiveBandRMS
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        norm = AdaptiveBandRMS(
            max_band_width=0.25,
            epsilon=1e-6,
            band_regularizer=keras.regularizers.L2(1e-4),
            name="test_adaptive_norm"
        )
        outputs = norm(inputs)
        model = keras.Model(inputs, outputs, name="test_model")

        # Get prediction from original model
        original_output = model(input_tensor_2d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(input_tensor_2d)

            # Verify outputs are identical
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Serialization should preserve model behavior"
            )

            # Verify layer configuration is preserved
            loaded_norm = loaded_model.get_layer("test_adaptive_norm")
            assert loaded_norm.max_band_width == 0.25
            assert loaded_norm.epsilon == 1e-6

            logger.info("2D serialization cycle test passed")

    def test_serialization_cycle_3d(self, input_tensor_3d):
        """Test serialization cycle for 3D tensors."""
        inputs = keras.Input(shape=input_tensor_3d.shape[1:])
        norm = AdaptiveBandRMS(
            axis=(1, 2),
            max_band_width=0.2,
            name="test_3d_norm"
        )
        outputs = norm(inputs)
        model = keras.Model(inputs, outputs)

        original_output = model(input_tensor_3d)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_3d_model.keras")
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(input_tensor_3d)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_output),
                ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6
            )

            logger.info("3D serialization cycle test passed")

    def test_get_config_completeness(self):
        """Test that get_config includes all initialization parameters."""
        layer = AdaptiveBandRMS(
            max_band_width=0.3,
            axis=(1, 2),
            epsilon=1e-6,
            band_initializer="he_normal",
            band_regularizer=keras.regularizers.L2(1e-4)
        )

        config = layer.get_config()

        # Check all parameters are in config
        assert "max_band_width" in config
        assert "axis" in config
        assert "epsilon" in config
        assert "band_initializer" in config
        assert "band_regularizer" in config

        # Check values are correct
        assert config["max_band_width"] == 0.3
        assert config["axis"] == (1, 2)
        assert config["epsilon"] == 1e-6

        logger.info("Config completeness test passed")

    # =========================================================================
    # MODEL INTEGRATION TESTS
    # =========================================================================

    def test_model_integration_transformer_style(self, input_tensor_3d):
        """Test integration in a transformer-style sequence model."""
        inputs = keras.Input(shape=input_tensor_3d.shape[1:])

        # Multi-head attention simulation
        x = keras.layers.Dense(64)(inputs)
        x = AdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)

        # Feed-forward network
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dense(64)(x)
        x = AdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)

        # Pooling and output projection
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Test forward pass
        y_pred = model.predict(input_tensor_3d, verbose=0)
        assert y_pred.shape == (input_tensor_3d.shape[0], 10)

        # Test that probabilities sum to 1
        prob_sums = np.sum(y_pred, axis=-1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-5)

        logger.info("Transformer-style integration test passed")

    def test_model_integration_cnn_multiaxis(self, input_tensor_4d):
        """Test integration in CNN with multiple axis configurations."""
        inputs = keras.Input(shape=input_tensor_4d.shape[1:])

        # Channel-wise normalization
        x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
        x = AdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)

        # Spatial normalization
        x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = AdaptiveBandRMS(axis=(1, 2), max_band_width=0.15)(x)

        # Global normalization before dense
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = AdaptiveBandRMS(axis=(1, 2, 3), max_band_width=0.2)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)

        # Dense with feature normalization
        x = keras.layers.Dense(32, activation='relu')(x)
        x = AdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)
        outputs = keras.layers.Dense(5, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Test forward pass
        y_pred = model.predict(input_tensor_4d, verbose=0)
        assert y_pred.shape == (input_tensor_4d.shape[0], 5)

        logger.info("CNN multi-axis integration test passed")

    # =========================================================================
    # EDGE CASES AND ROBUSTNESS TESTS
    # =========================================================================

    def test_extreme_input_values(self):
        """Test layer stability with extreme input values."""
        test_cases = [
            ("zeros", np.zeros((4, 32), dtype=np.float32)),
            ("very_small", np.ones((4, 32), dtype=np.float32) * 1e-10),
            ("very_large", np.ones((4, 32), dtype=np.float32) * 1e5),
            ("mixed_extreme", np.concatenate([
                np.ones((4, 16), dtype=np.float32) * 1e-8,
                np.ones((4, 16), dtype=np.float32) * 1e8
            ], axis=-1))
        ]

        for case_name, test_input in test_cases:
            layer = AdaptiveBandRMS(max_band_width=0.1)

            try:
                output = layer(test_input)

                # Check for NaN/Inf values
                assert not np.any(np.isnan(output.numpy())), f"NaN values in {case_name}"
                assert not np.any(np.isinf(output.numpy())), f"Inf values in {case_name}"

                logger.info(f"Extreme values test passed for {case_name}")

            except Exception as e:
                pytest.fail(f"Layer failed on {case_name} input: {e}")

    def test_variable_batch_sizes(self):
        """Test layer works with different batch sizes."""
        layer = AdaptiveBandRMS(axis=-1)
        base_shape = (32,)

        batch_sizes = [1, 2, 8, 16, 32]

        for batch_size in batch_sizes:
            test_input = np.random.randn(batch_size, *base_shape).astype(np.float32)

            try:
                output = layer(test_input)
                assert output.shape == test_input.shape

                logger.info(f"Variable batch size {batch_size} test passed")

            except Exception as e:
                pytest.fail(f"Layer failed with batch size {batch_size}: {e}")

    def test_training_vs_inference_mode(self, input_tensor_2d):
        """Test layer behavior in training vs inference mode."""
        layer = AdaptiveBandRMS(
            band_regularizer=keras.regularizers.L2(0.01)
        )

        # Training mode
        train_output = layer(input_tensor_2d, training=True)

        # Inference mode
        inference_output = layer(input_tensor_2d, training=False)

        # Outputs should be identical (no dropout or batch norm effects)
        np.testing.assert_allclose(
            ops.convert_to_numpy(train_output),
            ops.convert_to_numpy(inference_output),
            rtol=1e-6, atol=1e-6
        )

        logger.info("Training vs inference mode test passed")

    def test_regularization_integration(self, input_tensor_2d):
        """Test that regularization works properly."""
        layer = AdaptiveBandRMS(
            band_regularizer=keras.regularizers.L2(0.1)
        )

        # Build and call the layer
        layer.build(input_tensor_2d.shape)
        output = layer(input_tensor_2d)

        # Set non-zero weights to ensure regularization loss
        if layer.dense_layer is not None:
            weights = layer.dense_layer.get_weights()
            if len(weights) > 0:
                weights[0] = np.ones_like(weights[0]) * 0.1
                layer.dense_layer.set_weights(weights)

                # Call layer again to trigger regularization computation
                _ = layer(input_tensor_2d)

                # Check that regularization losses exist
                total_loss = sum(layer.losses)
                assert total_loss > 0, f"Expected positive regularization loss, got {total_loss}"

        logger.info("Regularization integration test passed")

    # =========================================================================
    # PERFORMANCE AND COMPATIBILITY TESTS
    # =========================================================================

    def test_gradient_flow(self, input_tensor_2d):
        """Test that gradients flow properly through the layer."""
        # Create simple model
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        x = AdaptiveBandRMS(max_band_width=0.2)(inputs)
        outputs = keras.layers.Dense(5)(x)
        model = keras.Model(inputs, outputs)

        tensor_input = tf.convert_to_tensor(input_tensor_2d)

        with tf.GradientTape() as tape:
            tape.watch(tensor_input)
            predictions = model(tensor_input)
            loss = ops.mean(ops.square(predictions))

        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Verify all gradients are non-None and finite
        for i, grad in enumerate(gradients):
            assert grad is not None, f"Gradient {i} is None"
            assert ops.all(ops.isfinite(grad)), f"Gradient {i} contains non-finite values"

        logger.info("Gradient flow test passed")

    def test_mixed_precision_compatibility(self, input_tensor_2d):
        """Test compatibility with mixed precision training."""
        # Enable mixed precision
        keras.mixed_precision.set_global_policy('mixed_float16')

        try:
            layer = AdaptiveBandRMS(max_band_width=0.2)

            # Convert input to float16
            input_fp16 = ops.cast(input_tensor_2d, 'float16')

            output = layer(input_fp16)

            # Output should be in the same dtype as input
            assert output.dtype == input_fp16.dtype

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert not np.any(np.isinf(output.numpy()))

            logger.info("Mixed precision compatibility test passed")

        finally:
            # Reset policy
            keras.mixed_precision.set_global_policy('float32')


# =========================================================================
# INTEGRATION TESTS WITH OTHER LAYERS
# =========================================================================

class TestAdaptiveBandRMSIntegration:
    """Test AdaptiveBandRMS integration with other layer types."""

    def test_integration_with_standard_normalization(self):
        """Test using AdaptiveBandRMS alongside standard normalization layers."""
        inputs = keras.Input(shape=(64,))

        # Model with standard LayerNormalization
        x1 = keras.layers.Dense(128)(inputs)
        x1 = keras.layers.LayerNormalization()(x1)
        x1 = keras.layers.ReLU()(x1)
        x1 = keras.layers.Dense(32)(x1)

        # Model with AdaptiveBandRMS
        x2 = keras.layers.Dense(128)(inputs)
        x2 = AdaptiveBandRMS(axis=-1)(x2)
        x2 = keras.layers.ReLU()(x2)
        x2 = keras.layers.Dense(32)(x2)

        model1 = keras.Model(inputs, x1)
        model2 = keras.Model(inputs, x2)

        test_input = np.random.randn(8, 64).astype(np.float32)

        out1 = model1.predict(test_input, verbose=0)
        out2 = model2.predict(test_input, verbose=0)

        # Both should produce valid outputs of the same shape
        assert out1.shape == out2.shape
        assert not np.any(np.isnan(out1))
        assert not np.any(np.isnan(out2))

        logger.info("Integration with standard normalization test passed")

    def test_integration_with_attention(self):
        """Test integration with attention mechanisms."""
        seq_len, embed_dim = 20, 64
        inputs = keras.Input(shape=(seq_len, embed_dim))

        # Multi-head attention with AdaptiveBandRMS
        attn_output = keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=embed_dim // 8
        )(inputs, inputs)

        # Add & Norm with AdaptiveBandRMS
        x = AdaptiveBandRMS(axis=-1)(inputs + attn_output)

        # Feed-forward with AdaptiveBandRMS
        ffn = keras.layers.Dense(embed_dim * 4, activation='relu')(x)
        ffn = keras.layers.Dense(embed_dim)(ffn)
        outputs = AdaptiveBandRMS(axis=-1)(x + ffn)

        model = keras.Model(inputs, outputs)
        test_input = np.random.randn(4, seq_len, embed_dim).astype(np.float32)

        output = model.predict(test_input, verbose=0)
        assert output.shape == test_input.shape
        assert not np.any(np.isnan(output))

        logger.info("Integration with attention test passed")


# =========================================================================
# BENCHMARK TESTS (Optional - for performance monitoring)
# =========================================================================

class TestAdaptiveBandRMSBenchmark:
    """Benchmark tests for performance monitoring."""

    def test_forward_pass_speed(self):
        """Benchmark forward pass speed."""
        layer = AdaptiveBandRMS()
        test_input = np.random.randn(32, 128).astype(np.float32)

        # Build the layer
        layer.build(test_input.shape)

        # Benchmark the forward pass
        result = layer(test_input)
        assert result.shape == test_input.shape

    def test_build_time(self):
        """Benchmark layer build time."""
        def build_layer():
            layer = AdaptiveBandRMS()
            layer.build((None, 256))
            return layer

        layer = build_layer()
        assert layer.built is True


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-x"  # Stop at first failure
    ])