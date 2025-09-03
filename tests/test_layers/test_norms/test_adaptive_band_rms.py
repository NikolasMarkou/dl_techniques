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
    """Comprehensive test suite for AdaptiveBandRMS layer."""

    @pytest.fixture
    def input_tensor_2d(self) -> keras.KerasTensor:
        """Create a 2D test input tensor for dense layer scenarios."""
        return np.random.randn(4, 64).astype(np.float32)

    @pytest.fixture
    def input_tensor_4d(self) -> keras.KerasTensor:
        """Create a 4D test input tensor for convolutional layer scenarios."""
        return np.random.randn(4, 16, 16, 32).astype(np.float32)

    @pytest.fixture
    def large_input_tensor_4d(self) -> keras.KerasTensor:
        """Create a larger 4D test input tensor for robustness testing."""
        return np.random.randn(8, 32, 32, 128).astype(np.float32)

    @pytest.fixture
    def layer_instance_default(self) -> 'AdaptiveBandRMS':
        """Create a default AdaptiveBandRMS layer instance."""
        return AdaptiveBandRMS()

    @pytest.fixture
    def layer_instance_custom(self) -> 'AdaptiveBandRMS':
        """Create a custom AdaptiveBandRMS layer instance with non-default parameters."""
        return AdaptiveBandRMS(
            max_band_width=0.2,
            epsilon=1e-6,
            band_regularizer=keras.regularizers.L2(1e-4),
            band_initializer="he_normal"
        )

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
        assert layer._is_conv_layer is False
        assert layer._normalization_strategy is None

        logger.info("✅ Default initialization test passed")

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

        logger.info("✅ Custom initialization test passed")

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

        logger.info("✅ Invalid parameters test passed")

    # =========================================================================
    # BUILD PROCESS TESTS
    # =========================================================================

    def test_build_process_2d(self, input_tensor_2d):
        """Test that the layer builds properly for 2D tensors."""
        layer = AdaptiveBandRMS()
        input_shape = input_tensor_2d.shape[1:]  # Remove batch dimension

        layer.build((None,) + input_shape)

        # Check build state
        assert layer.built is True
        assert layer._is_conv_layer is False

        # Check dense layer configuration
        assert layer.dense_layer is not None
        assert layer.dense_layer.built is True
        assert layer.dense_layer.units == input_shape[-1]  # Output per feature

        logger.info("✅ 2D build process test passed")

    def test_build_process_4d_channel_wise(self, input_tensor_4d):
        """Test build process for 4D tensors with channel-wise normalization."""
        layer = AdaptiveBandRMS(axis=-1)
        input_shape = input_tensor_4d.shape[1:]  # Remove batch dimension

        layer.build((None,) + input_shape)

        # Check build state
        assert layer.built is True
        assert layer._is_conv_layer is True
        assert layer._normalization_strategy == "channel_wise"

        # Check dense layer configuration for channel-wise normalization
        assert layer.dense_layer.units == input_shape[-1]  # One output per channel

        logger.info("✅ 4D channel-wise build process test passed")

    def test_build_process_4d_spatial(self, input_tensor_4d):
        """Test build process for 4D tensors with spatial normalization."""
        layer = AdaptiveBandRMS(axis=(1, 2))
        input_shape = input_tensor_4d.shape[1:]  # Remove batch dimension

        layer.build((None,) + input_shape)

        # Check build state
        assert layer.built is True
        assert layer._is_conv_layer is True
        assert layer._normalization_strategy == "spatial"

        # Check dense layer configuration for spatial normalization
        assert layer.dense_layer.units == input_shape[-1]  # One output per channel

        logger.info("✅ 4D spatial build process test passed")

    def test_build_process_4d_global(self, input_tensor_4d):
        """Test build process for 4D tensors with global normalization."""
        layer = AdaptiveBandRMS(axis=(1, 2, 3))
        input_shape = input_tensor_4d.shape[1:]  # Remove batch dimension

        layer.build((None,) + input_shape)

        # Check build state
        assert layer.built is True
        assert layer._is_conv_layer is True
        assert layer._normalization_strategy == "global"

        # Check dense layer configuration for global normalization
        assert layer.dense_layer.units == 1  # Single global scaling factor

        logger.info("✅ 4D global build process test passed")

    def test_invalid_axis_configurations_4d(self, input_tensor_4d):
        """Test that invalid axis configurations raise errors for 4D tensors."""
        input_shape = input_tensor_4d.shape[1:]

        # Test invalid single axis
        layer_invalid_axis = AdaptiveBandRMS(axis=1)  # Height axis
        with pytest.raises(ValueError, match="single axis must be -1 or 3"):
            layer_invalid_axis.build((None,) + input_shape)

        # Test invalid axis tuple
        layer_invalid_tuple = AdaptiveBandRMS(axis=(0, 1))  # Includes batch axis
        with pytest.raises(ValueError, match="axis tuple must be \\(1,2\\) or \\(1,2,3\\)"):
            layer_invalid_tuple.build((None,) + input_shape)

        logger.info("✅ Invalid axis configurations test passed")

    # =========================================================================
    # OUTPUT SHAPE TESTS
    # =========================================================================

    def test_output_shapes_2d(self, input_tensor_2d):
        """Test that output shapes are computed correctly for 2D tensors."""
        layer = AdaptiveBandRMS()
        output = layer(input_tensor_2d)

        # Check output shape matches input shape
        assert output.shape == input_tensor_2d.shape

        # Test compute_output_shape separately
        computed_shape = layer.compute_output_shape(input_tensor_2d.shape)
        assert computed_shape == input_tensor_2d.shape

        logger.info("✅ 2D output shapes test passed")

    def test_output_shapes_4d(self, input_tensor_4d):
        """Test that output shapes are computed correctly for 4D tensors."""
        normalization_configs = [
            ("channel_wise", -1),
            ("spatial", (1, 2)),
            ("global", (1, 2, 3))
        ]

        for strategy_name, axis in normalization_configs:
            layer = AdaptiveBandRMS(axis=axis)
            output = layer(input_tensor_4d)

            # Check output shape matches input shape
            assert output.shape == input_tensor_4d.shape, f"Failed for {strategy_name}"

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor_4d.shape)
            assert computed_shape == input_tensor_4d.shape, f"Failed for {strategy_name}"

        logger.info("✅ 4D output shapes test passed")

    # =========================================================================
    # FORWARD PASS TESTS
    # =========================================================================

    def test_forward_pass_2d(self, input_tensor_2d):
        """Test forward pass produces expected values for 2D tensors."""
        layer = AdaptiveBandRMS(max_band_width=0.2)
        output = layer(input_tensor_2d)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy())), "Output contains NaN values"
        assert not np.any(np.isinf(output.numpy())), "Output contains Inf values"

        # Check RMS constraint is within expected band [1-α, 1]
        output_rms = float(ops.sqrt(ops.mean(ops.square(output), axis=-1, keepdims=False)).numpy().mean())
        expected_min_rms = 1.0 - 0.2  # 0.8
        expected_max_rms = 1.0

        assert expected_min_rms <= output_rms <= expected_max_rms, \
            f"RMS {output_rms} not in expected band [{expected_min_rms}, {expected_max_rms}]"

        logger.info("✅ 2D forward pass test passed")

    def test_forward_pass_4d_strategies(self, input_tensor_4d):
        """Test forward pass for different 4D normalization strategies."""
        strategies = [
            ("channel_wise", -1),
            ("spatial", (1, 2)),
            ("global", (1, 2, 3))
        ]

        for strategy_name, axis in strategies:
            layer = AdaptiveBandRMS(axis=axis, max_band_width=0.15)
            output = layer(input_tensor_4d)

            # Basic sanity checks
            assert not np.any(np.isnan(output.numpy())), f"NaN in {strategy_name} output"
            assert not np.any(np.isinf(output.numpy())), f"Inf in {strategy_name} output"

            # Check that output shape is preserved
            assert output.shape == input_tensor_4d.shape, f"Shape mismatch in {strategy_name}"

            logger.info(f"✅ 4D {strategy_name} forward pass test passed")

    def test_deterministic_output_controlled_input(self):
        """Test with controlled inputs for deterministic output verification."""
        # Create controlled input with known statistics
        controlled_input = ops.ones((2, 8)) * 2.0  # RMS = 2.0

        # Create layer with specific parameters for predictable behavior
        layer = AdaptiveBandRMS(
            max_band_width=0.1,
            band_initializer="zeros"  # This will make dense layer output near zero initially
        )

        # Forward pass
        output = layer(controlled_input)

        # With zero-initialized dense layer, scaling should be close to (1 - max_band_width)
        # After RMS normalization, each element should be 1.0, then scaled by ~0.9
        expected_magnitude_range = (0.85, 0.95)  # Account for sigmoid and initialization
        actual_magnitude = float(ops.mean(ops.abs(output)).numpy())

        assert expected_magnitude_range[0] <= actual_magnitude <= expected_magnitude_range[1], \
            f"Magnitude {actual_magnitude} not in expected range {expected_magnitude_range}"

        logger.info("✅ Deterministic controlled input test passed")

    # =========================================================================
    # SERIALIZATION TESTS
    # =========================================================================

    def test_serialization_2d(self, input_tensor_2d):
        """Test serialization and deserialization of the layer for 2D tensors."""
        original_layer = AdaptiveBandRMS(
            max_band_width=0.25,
            epsilon=1e-6,
            band_regularizer=keras.regularizers.L2(1e-4)
        )

        # Build the layer
        original_layer.build(input_tensor_2d.shape)

        # Get prediction before serialization
        original_output = original_layer(input_tensor_2d)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = AdaptiveBandRMS.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Copy weights using proper Keras methods
        recreated_layer.set_weights(original_layer.get_weights())

        # Get prediction after serialization
        recreated_output = recreated_layer(input_tensor_2d)

        # Check configuration matches
        assert recreated_layer.max_band_width == original_layer.max_band_width
        assert recreated_layer.epsilon == original_layer.epsilon
        assert recreated_layer.axis == original_layer.axis

        # Check outputs match (approximately due to floating point precision)
        assert np.allclose(original_output.numpy(), recreated_output.numpy(), rtol=1e-6)

        logger.info("✅ 2D serialization test passed")

    def test_serialization_4d(self, input_tensor_4d):
        """Test serialization and deserialization of the layer for 4D tensors."""
        original_layer = AdaptiveBandRMS(
            axis=(1, 2),
            max_band_width=0.2,
            band_initializer="he_normal"
        )

        # Build the layer
        original_layer.build(input_tensor_4d.shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = AdaptiveBandRMS.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.max_band_width == original_layer.max_band_width
        assert recreated_layer.axis == original_layer.axis
        assert recreated_layer._normalization_strategy == original_layer._normalization_strategy

        logger.info("✅ 4D serialization test passed")

    # =========================================================================
    # MODEL INTEGRATION TESTS
    # =========================================================================

    def test_model_integration_2d(self, input_tensor_2d):
        """Test the layer in a model context with 2D tensors."""
        # Create a simple model with the custom layer
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = AdaptiveBandRMS(max_band_width=0.15)(x)
        x = keras.layers.Dense(16, activation='relu')(x)
        x = AdaptiveBandRMS(max_band_width=0.1)(x)
        outputs = keras.layers.Dense(8, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy']
        )

        # Test forward pass
        y_pred = model.predict(input_tensor_2d, verbose=0)
        assert y_pred.shape == (input_tensor_2d.shape[0], 8)

        # Test that probabilities sum to 1 (softmax check)
        prob_sums = np.sum(y_pred, axis=1)
        assert np.allclose(prob_sums, 1.0, rtol=1e-5)

        logger.info("✅ 2D model integration test passed")

    def test_model_integration_4d(self, input_tensor_4d):
        """Test the layer in a model context with 4D tensors."""
        # Create a convolutional model with the custom layer
        inputs = keras.Input(shape=input_tensor_4d.shape[1:])

        # First conv block with channel-wise normalization
        x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
        x = AdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)
        x = keras.layers.MaxPooling2D(2)(x)

        # Second conv block with spatial normalization
        x = keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = AdaptiveBandRMS(axis=(1, 2), max_band_width=0.15)(x)
        x = keras.layers.MaxPooling2D(2)(x)

        # Third conv block with global normalization
        x = keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = AdaptiveBandRMS(axis=(1, 2, 3), max_band_width=0.2)(x)
        x = keras.layers.GlobalAveragePooling2D()(x)

        # Dense layers
        x = keras.layers.Dense(32, activation='relu')(x)
        x = AdaptiveBandRMS(axis=-1, max_band_width=0.1)(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Test forward pass
        y_pred = model.predict(input_tensor_4d, verbose=0)
        assert y_pred.shape == (input_tensor_4d.shape[0], 10)

        logger.info("✅ 4D model integration test passed")

    # =========================================================================
    # MODEL SAVE/LOAD TESTS
    # =========================================================================

    def test_model_save_load(self, input_tensor_2d):
        """Test saving and loading a model with the custom layer."""
        # Create a model with the custom layer
        inputs = keras.Input(shape=input_tensor_2d.shape[1:])
        x = keras.layers.Dense(32, activation='relu')(inputs)
        x = AdaptiveBandRMS(max_band_width=0.2, name="adaptive_norm_1")(x)
        x = keras.layers.Dense(16, activation='relu')(x)
        x = AdaptiveBandRMS(max_band_width=0.15, name="adaptive_norm_2")(x)
        outputs = keras.layers.Dense(8, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model.predict(input_tensor_2d, verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "test_model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            with keras.utils.custom_object_scope({'AdaptiveBandRMS': AdaptiveBandRMS}):
                loaded_model = keras.models.load_model(model_path)

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor_2d, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer("adaptive_norm_1"), AdaptiveBandRMS)
            assert isinstance(loaded_model.get_layer("adaptive_norm_2"), AdaptiveBandRMS)

            # Check layer parameters are preserved
            loaded_layer = loaded_model.get_layer("adaptive_norm_1")
            assert loaded_layer.max_band_width == 0.2

        logger.info("✅ Model save/load test passed")

    # =========================================================================
    # EDGE CASE AND ROBUSTNESS TESTS
    # =========================================================================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = AdaptiveBandRMS(max_band_width=0.1)

        batch_size = 4
        feature_dim = 32

        test_cases = [
            ("zeros", np.zeros((batch_size, feature_dim))),
            ("very_small", np.ones((batch_size, feature_dim)) * 1e-10),
            ("very_large", np.ones((batch_size, feature_dim)) * 1e5),
            ("mixed_extreme", np.concatenate([
                np.ones((batch_size, feature_dim//2)) * 1e-8,
                np.ones((batch_size, feature_dim//2)) * 1e8
            ], axis=-1))
        ]

        for case_name, test_input in test_cases:
            test_input = test_input.astype(np.float32)

            try:
                output = layer(test_input)

                # Check for NaN/Inf values
                assert not np.any(np.isnan(output.numpy())), f"NaN values in {case_name} case"
                assert not np.any(np.isinf(output.numpy())), f"Inf values in {case_name} case"

                logger.info(f"✅ {case_name} numerical stability test passed")

            except Exception as e:
                pytest.fail(f"Layer failed on {case_name} input: {e}")

    def test_different_batch_sizes(self, input_tensor_4d):
        """Test layer with different batch sizes."""
        layer = AdaptiveBandRMS(axis=-1)
        original_shape = input_tensor_4d.shape

        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            # Create input with different batch size
            test_input = np.random.randn(batch_size, *original_shape[1:]).astype(np.float32)

            try:
                output = layer(test_input)
                assert output.shape == test_input.shape

                logger.info(f"✅ Batch size {batch_size} test passed")

            except Exception as e:
                pytest.fail(f"Layer failed with batch size {batch_size}: {e}")

    def test_regularization_losses(self, input_tensor_2d):
        """Test that regularization losses are properly applied."""
        # Create layer with regularization
        layer = AdaptiveBandRMS(
            band_regularizer=keras.regularizers.L2(0.1)
        )

        # Build the layer to initialize weights
        layer.build(input_tensor_2d.shape)

        # No regularization losses before calling the layer
        initial_losses = len(layer.losses)

        # Apply the layer (this should trigger regularization loss computation)
        output = layer(input_tensor_2d)

        # Force regularization loss computation by accessing layer losses
        # In Keras 3.x, losses are computed when weights are accessed
        total_reg_loss = sum(layer.losses)

        # Should have regularization losses now
        assert len(layer.losses) >= initial_losses, "No regularization losses were added"

        # If weights are non-zero (after layer call), regularization loss should be positive
        # Let's modify weights to ensure they're non-zero
        if hasattr(layer, 'dense_layer') and layer.dense_layer is not None:
            # Set some non-zero weights to ensure regularization loss
            weights = layer.dense_layer.get_weights()
            if len(weights) > 0:
                weights[0] = np.ones_like(weights[0]) * 0.1  # Set kernel to small non-zero values
                layer.dense_layer.set_weights(weights)

                # Recompute losses after setting weights
                _ = layer(input_tensor_2d)  # Forward pass to compute regularization
                total_reg_loss = sum(layer.losses)

                assert total_reg_loss > 0, f"Expected positive regularization loss, got {total_reg_loss}"

        logger.info("✅ Regularization losses test passed")

    # =========================================================================
    # UTILITY METHODS FOR TESTING
    # =========================================================================

    def test_rms_statistics_computation_internal(self, input_tensor_4d):
        """Test internal RMS statistics computation for different strategies."""
        strategies = [
            ("channel_wise", -1),
            ("spatial", (1, 2)),
            ("global", (1, 2, 3))
        ]

        for strategy_name, axis in strategies:
            layer = AdaptiveBandRMS(axis=axis)
            layer.build(input_tensor_4d.shape)

            # Access internal method for testing
            inputs_fp32 = ops.cast(input_tensor_4d, "float32")
            rms_stats, rms_full = layer._compute_rms_statistics(inputs_fp32)

            # Check that RMS statistics have correct shape [batch, 1]
            assert rms_stats.shape == (input_tensor_4d.shape[0], 1), \
                f"Wrong RMS stats shape for {strategy_name}: {rms_stats.shape}"

            # Check that all RMS statistics are positive
            assert ops.min(rms_stats) > 0, f"Non-positive RMS stats in {strategy_name}"

            logger.info(f"✅ Internal RMS computation test passed for {strategy_name}")


# =========================================================================
# INTEGRATION TESTS WITH OTHER LAYERS
# =========================================================================

class TestAdaptiveBandRMSIntegration:
    """Test AdaptiveBandRMS integration with other layer types."""

    def test_integration_with_batch_normalization(self):
        """Test using AdaptiveBandRMS alongside BatchNormalization."""
        inputs = keras.Input(shape=(16, 16, 32))

        # Compare model with BatchNorm vs AdaptiveBandRMS
        x1 = keras.layers.Conv2D(64, 3, padding='same')(inputs)
        x1 = keras.layers.BatchNormalization()(x1)
        x1 = keras.layers.ReLU()(x1)

        x2 = keras.layers.Conv2D(64, 3, padding='same')(inputs)
        x2 = AdaptiveBandRMS(axis=-1)(x2)
        x2 = keras.layers.ReLU()(x2)

        # Both should produce valid outputs
        model1 = keras.Model(inputs, x1)
        model2 = keras.Model(inputs, x2)

        test_input = np.random.randn(4, 16, 16, 32).astype(np.float32)

        out1 = model1.predict(test_input, verbose=0)
        out2 = model2.predict(test_input, verbose=0)

        assert out1.shape == out2.shape
        assert not np.any(np.isnan(out1))
        assert not np.any(np.isnan(out2))

        logger.info("✅ Integration with BatchNormalization test passed")

    def test_integration_with_dropout(self):
        """Test using AdaptiveBandRMS with Dropout layers."""
        inputs = keras.Input(shape=(64,))
        x = keras.layers.Dense(128)(inputs)
        x = AdaptiveBandRMS()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(32)(x)
        x = AdaptiveBandRMS()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs, outputs)
        test_input = np.random.randn(8, 64).astype(np.float32)

        # Test in training and inference modes
        train_output = model(test_input, training=True)
        inference_output = model(test_input, training=False)

        assert train_output.shape == inference_output.shape
        assert not np.any(np.isnan(train_output.numpy()))
        assert not np.any(np.isnan(inference_output.numpy()))

        logger.info("✅ Integration with Dropout test passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])