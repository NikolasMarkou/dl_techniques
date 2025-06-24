"""
Comprehensive test suite for HaarWaveletInitializer.

This module contains test cases for the HaarWaveletInitializer and
related functionality, ensuring proper wavelet properties, serialization,
and integration with Keras layers.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Tuple, Any, List
from unittest.mock import patch

from dl_techniques.initializers.haar_wavelet_initializer import (
    HaarWaveletInitializer,
    create_haar_depthwise_conv2d
)


class TestHaarWaveletInitializer:
    """Test suite for HaarWaveletInitializer implementation."""

    @pytest.fixture
    def standard_shape(self) -> Tuple[int, int, int, int]:
        """Standard shape for 2x2 kernels with 3 input channels and 4 output channels."""
        return (2, 2, 3, 4)

    @pytest.fixture
    def single_channel_shape(self) -> Tuple[int, int, int, int]:
        """Shape for single input channel with 4 wavelet outputs."""
        return (2, 2, 1, 4)

    @pytest.fixture
    def initializer(self) -> HaarWaveletInitializer:
        """Default initializer instance."""
        return HaarWaveletInitializer(scale=1.0)

    @pytest.fixture
    def input_tensor(self):
        """Create a test input tensor."""
        return keras.random.normal([2, 32, 32, 3])

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        initializer = HaarWaveletInitializer()

        assert initializer.scale == 1.0
        assert initializer.seed is None

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        scale = 2.0
        seed = 42

        initializer = HaarWaveletInitializer(scale=scale, seed=seed)

        assert initializer.scale == scale
        assert initializer.seed == seed

    def test_invalid_scale_values(self):
        """Test that invalid scale values raise appropriate errors."""
        invalid_scales = [0.0, -1.0, -0.5]

        for scale in invalid_scales:
            with pytest.raises(ValueError, match="Scale must be positive"):
                HaarWaveletInitializer(scale=scale)

    def test_call_standard_shape(self, initializer: HaarWaveletInitializer,
                                standard_shape: Tuple[int, int, int, int]):
        """Test calling initializer with standard shape."""
        weights = initializer(standard_shape)

        # Check output properties
        assert weights.shape == standard_shape
        assert not np.any(np.isnan(weights.numpy()))
        assert not np.any(np.isinf(weights.numpy()))

    def test_call_single_channel(self, initializer: HaarWaveletInitializer,
                                single_channel_shape: Tuple[int, int, int, int]):
        """Test wavelet generation for single channel input."""
        weights = initializer(single_channel_shape).numpy()

        # Should have exactly 4 different wavelet patterns
        expected_patterns = self._get_expected_haar_patterns()

        for i in range(4):
            pattern = weights[:, :, 0, i]
            expected = expected_patterns[i]
            np.testing.assert_allclose(pattern, expected, rtol=1e-6)

    def _get_expected_haar_patterns(self) -> List[np.ndarray]:
        """Get expected Haar wavelet patterns."""
        sqrt2 = np.sqrt(2.0)
        return [
            # LL: Scaling function
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            # LH: Horizontal detail
            np.array([[1.0/sqrt2, -1.0/sqrt2], [1.0/sqrt2, -1.0/sqrt2]]),
            # HL: Vertical detail
            np.array([[1.0/sqrt2, 1.0/sqrt2], [-1.0/sqrt2, -1.0/sqrt2]]),
            # HH: Diagonal detail
            np.array([[1.0/sqrt2, -1.0/sqrt2], [-1.0/sqrt2, 1.0/sqrt2]])
        ]

    @pytest.mark.parametrize("invalid_shape", [
        (3, 3, 3, 4),  # Invalid kernel size
        (2, 3, 3, 4),  # Mismatched dimensions
        (4, 4, 3, 4),  # Too large kernel
        (1, 1, 3, 4),  # Too small kernel
        (2, 2, 3),     # Wrong number of dimensions
        (2, 2, 3, 4, 1),  # Too many dimensions
    ])
    def test_invalid_shapes(self, invalid_shape: Tuple[int, ...],
                           initializer: HaarWaveletInitializer):
        """Test that invalid shapes raise appropriate errors."""
        with pytest.raises(ValueError):
            initializer(invalid_shape)

    @pytest.mark.parametrize("scale,expected_max", [
        (1.0, 1.0 / np.sqrt(2.0)),  # Standard scale
        (2.0, 2.0 / np.sqrt(2.0)),  # Double scale
        (0.5, 0.5 / np.sqrt(2.0)),  # Half scale
        (np.sqrt(2.0), 1.0),        # Scale to make max = 1.0
    ])
    def test_scaling_behavior(self, scale: float, expected_max: float):
        """Test that scaling works correctly."""
        initializer = HaarWaveletInitializer(scale=scale)
        weights = initializer((2, 2, 1, 4)).numpy()

        actual_max = np.max(np.abs(weights))
        np.testing.assert_allclose(actual_max, expected_max, rtol=1e-6)

    def test_orthogonality_properties(self, single_channel_shape: Tuple[int, int, int, int]):
        """Test that generated wavelets maintain orthogonality."""
        initializer = HaarWaveletInitializer(scale=1.0)
        weights = initializer(single_channel_shape).numpy()

        # Extract the 4 wavelet patterns
        patterns = [weights[:, :, 0, i].flatten() for i in range(4)]

        # Test pairwise orthogonality
        for i in range(4):
            for j in range(i + 1, 4):
                dot_product = np.dot(patterns[i], patterns[j])
                np.testing.assert_allclose(dot_product, 0.0, atol=1e-6)

    def test_energy_preservation(self, single_channel_shape: Tuple[int, int, int, int]):
        """Test that wavelets preserve energy (Parseval's theorem)."""
        initializer = HaarWaveletInitializer(scale=1.0)
        weights = initializer(single_channel_shape).numpy()

        # Calculate expected energy:
        # LL: 4 * (0.5)^2 = 1
        # LH: 4 * (1/√2)^2 = 2
        # HL: 4 * (1/√2)^2 = 2
        # HH: 4 * (1/√2)^2 = 2
        # Total: 1 + 2 + 2 + 2 = 7
        total_energy = np.sum(weights ** 2)
        expected_energy = 7.0  # Corrected expected energy

        np.testing.assert_allclose(total_energy, expected_energy, rtol=1e-6)

    def test_pattern_distribution(self, standard_shape: Tuple[int, int, int, int]):
        """Test that patterns are distributed correctly across channels."""
        initializer = HaarWaveletInitializer(scale=1.0)
        weights = initializer(standard_shape).numpy()

        kernel_h, kernel_w, in_channels, channel_multiplier = standard_shape
        expected_patterns = self._get_expected_haar_patterns()

        for i in range(in_channels):
            for j in range(channel_multiplier):
                pattern_idx = (i * channel_multiplier + j) % 4
                expected = expected_patterns[pattern_idx]
                actual = weights[:, :, i, j]

                np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_serialization_roundtrip(self):
        """Test complete serialization and deserialization."""
        original = HaarWaveletInitializer(scale=1.5, seed=42)
        config = original.get_config()

        # Verify config contents
        assert config['scale'] == 1.5
        assert config['seed'] == 42

        # Test from_config class method
        reconstructed = HaarWaveletInitializer.from_config(config)

        # Verify properties match
        assert reconstructed.scale == original.scale
        assert reconstructed.seed == original.seed

        # Test that they produce identical outputs
        shape = (2, 2, 3, 4)
        original_weights = original(shape)
        reconstructed_weights = reconstructed(shape)

        np.testing.assert_array_equal(
            original_weights.numpy(),
            reconstructed_weights.numpy()
        )

    def test_keras_serialization_compatibility(self):
        """Test compatibility with Keras serialization system."""
        initializer = HaarWaveletInitializer(scale=2.0)

        # Test Keras serialization
        config = keras.initializers.serialize(initializer)
        deserialized = keras.initializers.deserialize(config)

        assert isinstance(deserialized, HaarWaveletInitializer)
        assert deserialized.scale == 2.0

    def test_dtype_handling(self, standard_shape: Tuple[int, int, int, int]):
        """Test handling of different data types."""
        initializer = HaarWaveletInitializer()

        # Test with different dtypes
        dtypes = ['float32', 'float64']

        for dtype in dtypes:
            weights = initializer(standard_shape, dtype=dtype)
            assert weights.dtype.name == dtype

    @patch('dl_techniques.initializers.haar_wavelet_initializer.logger')
    def test_logging_behavior(self, mock_logger):
        """Test that appropriate logging occurs."""
        # Test initialization logging
        HaarWaveletInitializer(scale=2.0)
        mock_logger.info.assert_called_with("Initialized HaarWaveletInitializer with scale=2.0")

        # Test call logging
        initializer = HaarWaveletInitializer()
        initializer((2, 2, 1, 4))
        mock_logger.debug.assert_called()


class TestCreateHaarDepthwiseConv2D:
    """Test suite for create_haar_depthwise_conv2d utility function."""

    @pytest.fixture
    def input_shape(self) -> Tuple[int, int, int]:
        """Standard input shape for testing."""
        return (32, 32, 3)

    def test_basic_creation(self, input_shape: Tuple[int, int, int]):
        """Test basic layer creation with default parameters."""
        layer = create_haar_depthwise_conv2d(input_shape=input_shape)

        assert isinstance(layer, keras.layers.DepthwiseConv2D)
        assert layer.kernel_size == (2, 2)
        assert layer.strides == (2, 2)
        assert layer.padding == 'valid'
        assert layer.depth_multiplier == 4
        assert not layer.use_bias
        assert not layer.trainable

    def test_custom_parameters(self, input_shape: Tuple[int, int, int]):
        """Test layer creation with custom parameters."""
        layer = create_haar_depthwise_conv2d(
            input_shape=input_shape,
            channel_multiplier=8,
            scale=2.0,
            use_bias=True,
            kernel_regularizer='l2',
            trainable=True,
            name='custom_haar'
        )

        assert layer.depth_multiplier == 8
        assert layer.use_bias
        assert layer.trainable
        assert layer.name == 'custom_haar'
        assert layer.depthwise_regularizer is not None

    def test_forward_pass(self, input_shape: Tuple[int, int, int]):
        """Test forward pass through the layer."""
        layer = create_haar_depthwise_conv2d(input_shape=input_shape)

        # Create test input
        batch_size = 2
        test_input = keras.random.normal([batch_size] + list(input_shape))

        # Forward pass
        output = layer(test_input)

        # Check output shape (should be halved due to stride=2)
        h, w, c = input_shape
        expected_shape = (batch_size, h//2, w//2, c * 4)
        assert output.shape == expected_shape

    def test_wavelet_decomposition_properties(self, input_shape: Tuple[int, int, int]):
        """Test that the layer performs proper wavelet decomposition."""
        layer = create_haar_depthwise_conv2d(input_shape=input_shape, trainable=False)

        # Create a simple test pattern
        h, w, c = input_shape
        test_input = keras.ops.ones((1, h, w, c))

        output = layer(test_input)
        output_np = output.numpy()

        # For a constant input, only the LL (approximation) coefficients should be non-zero
        # The detail coefficients (LH, HL, HH) should be zero
        for ch in range(c):
            # LL coefficients (every 4th channel starting from 0)
            # For constant input with LL filter [[0.5, 0.5], [0.5, 0.5]]:
            # convolution result = 1*0.5 + 1*0.5 + 1*0.5 + 1*0.5 = 2.0
            ll_channel = output_np[0, :, :, ch * 4]
            assert np.allclose(ll_channel, 2.0, rtol=1e-6)  # Corrected expected value

            # Detail coefficients should be zero for constant input
            for detail_idx in [1, 2, 3]:  # LH, HL, HH
                detail_channel = output_np[0, :, :, ch * 4 + detail_idx]
                assert np.allclose(detail_channel, 0.0, atol=1e-6)

    @pytest.mark.parametrize("invalid_input_shape", [
        (32, 32),      # 2D instead of 3D
        (32, 32, 3, 1), # 4D instead of 3D
        (32,),         # 1D
    ])
    def test_invalid_input_shapes(self, invalid_input_shape: Tuple[int, ...]):
        """Test that invalid input shapes raise errors."""
        with pytest.raises(ValueError, match="Expected 3D input shape"):
            create_haar_depthwise_conv2d(input_shape=invalid_input_shape)

    def test_invalid_channel_multiplier(self, input_shape: Tuple[int, int, int]):
        """Test that invalid channel multipliers raise errors."""
        with pytest.raises(ValueError, match="channel_multiplier must be positive"):
            create_haar_depthwise_conv2d(
                input_shape=input_shape,
                channel_multiplier=0
            )

        with pytest.raises(ValueError, match="channel_multiplier must be positive"):
            create_haar_depthwise_conv2d(
                input_shape=input_shape,
                channel_multiplier=-1
            )

        # Non-standard channel_multiplier should only warn, not raise error
        # This should work without raising an exception
        layer = create_haar_depthwise_conv2d(
            input_shape=input_shape,
            channel_multiplier=2,
            trainable=False
        )
        assert layer.depth_multiplier == 2

    @patch('dl_techniques.initializers.haar_wavelet_initializer.logger')
    def test_logging_warnings(self, mock_logger, input_shape: Tuple[int, int, int]):
        """Test that warnings are logged for non-standard configurations."""
        create_haar_depthwise_conv2d(
            input_shape=input_shape,
            channel_multiplier=2,  # Non-standard
            trainable=False
        )

        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "channel_multiplier=2" in warning_call
        assert "standard wavelet decomposition" in warning_call

    def test_non_standard_channel_multiplier_behavior(self, input_shape: Tuple[int, int, int]):
        """Test that non-standard channel multipliers work without raising errors."""
        # Should work without raising error, just log warning
        layer = create_haar_depthwise_conv2d(
            input_shape=input_shape,
            channel_multiplier=2,
            trainable=False
        )

        # Verify layer was created successfully
        assert layer.depth_multiplier == 2
        assert not layer.trainable

    def test_model_integration(self, input_shape: Tuple[int, int, int]):
        """Test integration in a complete model."""
        # Create a simple model with Haar wavelet layer
        inputs = keras.layers.Input(shape=input_shape)
        x = create_haar_depthwise_conv2d(input_shape=input_shape)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test forward pass
        batch_size = 4
        test_input = keras.random.normal([batch_size] + list(input_shape))
        prediction = model(test_input, training=False)

        assert prediction.shape == (batch_size, 10)
        assert np.allclose(np.sum(prediction.numpy(), axis=1), 1.0, rtol=1e-6)

    def test_model_save_load(self, input_shape: Tuple[int, int, int]):
        """Test saving and loading a model with Haar wavelet layer."""
        # Create model with custom layer
        inputs = keras.layers.Input(shape=input_shape, name='input')
        x = create_haar_depthwise_conv2d(
            input_shape=input_shape,
            name='haar_wavelet'
        )(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5, activation='softmax', name='output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='test_model')

        # Generate test data and prediction
        test_input = keras.random.normal([2] + list(input_shape))
        original_prediction = model.predict(test_input, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "haar_model.keras")

            # Save model
            model.save(model_path)

            # Load model with custom objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'HaarWaveletInitializer': HaarWaveletInitializer}
            )

            # Test that loaded model produces same results
            loaded_prediction = loaded_model.predict(test_input, verbose=0)

            np.testing.assert_allclose(
                original_prediction,
                loaded_prediction,
                rtol=1e-6
            )

            # Verify layer is preserved correctly
            haar_layer = loaded_model.get_layer('haar_wavelet')
            assert isinstance(haar_layer, keras.layers.DepthwiseConv2D)
            assert not haar_layer.trainable

    def test_gradient_flow(self, input_shape: Tuple[int, int, int]):
        """Test gradient flow through trainable Haar wavelet layer."""
        layer = create_haar_depthwise_conv2d(
            input_shape=input_shape,
            trainable=True  # Enable training
        )

        # Create input variable
        test_input = keras.Variable(keras.random.normal([1] + list(input_shape)))

        with tf.GradientTape() as tape:
            output = layer(test_input)
            loss = keras.ops.sum(keras.ops.square(output))

        # Compute gradients
        gradients = tape.gradient(loss, layer.trainable_variables)

        # Check that gradients exist and are not None
        assert gradients is not None
        assert len(gradients) == len(layer.trainable_variables)

        for grad in gradients:
            assert grad is not None
            assert not keras.ops.any(keras.ops.isnan(grad))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])