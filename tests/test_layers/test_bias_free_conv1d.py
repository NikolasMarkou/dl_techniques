"""
Comprehensive test suite for BiasFreeConv1D and BiasFreeResidualBlock1D layers.

Tests cover initialization, build process, forward pass, serialization,
model integration, and edge cases following the dl-techniques testing guide.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

# Import the layers to test
from dl_techniques.layers.bias_free_conv1d import BiasFreeConv1D, BiasFreeResidualBlock1D


class TestBiasFreeConv1D:
    """Test suite for BiasFreeConv1D layer implementation."""

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create a test input tensor for time series."""
        return tf.random.normal([4, 100, 3])  # [batch, time, features]

    @pytest.fixture
    def small_input_tensor(self) -> tf.Tensor:
        """Create a smaller test input tensor for controlled tests."""
        return tf.random.normal([2, 20, 1])  # [batch, time, features]

    @pytest.fixture
    def layer_instance(self) -> BiasFreeConv1D:
        """Create a default layer instance for testing."""
        return BiasFreeConv1D(filters=32)

    # -------------------------------------------------------------------------
    # 1. Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = BiasFreeConv1D(filters=64)

        # Check default values
        assert layer.filters == 64
        assert layer.kernel_size == 3
        assert layer.activation == 'relu'
        assert isinstance(layer.kernel_initializer, str)
        assert layer.kernel_initializer == 'glorot_uniform'
        assert layer.kernel_regularizer is None
        assert layer.use_batch_norm is True

        # Check sublayers are initially None
        assert layer.conv is None
        assert layer.batch_norm is None
        assert layer.activation_layer is None

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_regularizer = keras.regularizers.L2(1e-4)

        layer = BiasFreeConv1D(
            filters=128,
            kernel_size=5,
            activation='gelu',
            kernel_initializer='he_normal',
            kernel_regularizer=custom_regularizer,
            use_batch_norm=False
        )

        # Check custom values
        assert layer.filters == 128
        assert layer.kernel_size == 5
        assert layer.activation == 'gelu'
        assert layer.kernel_initializer == 'he_normal'
        assert layer.kernel_regularizer == custom_regularizer
        assert layer.use_batch_norm is False

    def test_initialization_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Keras validates parameters during build/call, not initialization
        layer = BiasFreeConv1D(filters=-10)  # This won't raise error yet
        test_input = tf.random.normal([1, 20, 3])

        # Error should occur during layer call/build
        with pytest.raises((ValueError, TypeError)):
            layer(test_input)

    def test_initialization_different_kernel_sizes(self):
        """Test initialization with different kernel size formats."""
        # Different integer kernel sizes
        for kernel_size in [1, 3, 5, 7, 9]:
            layer = BiasFreeConv1D(filters=32, kernel_size=kernel_size)
            assert layer.kernel_size == kernel_size

    # -------------------------------------------------------------------------
    # 2. Build Process Tests
    # -------------------------------------------------------------------------

    def test_build_process(self, input_tensor):
        """Test that the layer builds properly."""
        layer = BiasFreeConv1D(filters=64)
        layer(input_tensor)  # Forward pass triggers build

        # Check that layer is built
        assert layer.built is True
        assert layer._build_input_shape is not None

        # Check sublayers were created
        assert layer.conv is not None
        assert layer.batch_norm is not None  # default use_batch_norm=True
        assert layer.activation_layer is not None  # default activation='relu'

        # Check conv layer properties
        assert layer.conv.filters == 64
        assert layer.conv.use_bias is False  # Key requirement
        assert layer.conv.padding == 'same'

        # Check batch norm properties
        assert layer.batch_norm.center is False  # Key requirement
        assert layer.batch_norm.scale is True

    def test_build_without_batch_norm(self, input_tensor):
        """Test building with batch normalization disabled."""
        layer = BiasFreeConv1D(filters=32, use_batch_norm=False)
        layer(input_tensor)

        assert layer.conv is not None
        assert layer.batch_norm is None  # Should be None when disabled
        assert layer.activation_layer is not None

    def test_build_without_activation(self, input_tensor):
        """Test building with activation disabled."""
        layer = BiasFreeConv1D(filters=32, activation=None)
        layer(input_tensor)

        assert layer.conv is not None
        assert layer.batch_norm is not None
        assert layer.activation_layer is None  # Should be None when disabled

    def test_build_minimal_configuration(self, input_tensor):
        """Test building with minimal configuration."""
        layer = BiasFreeConv1D(
            filters=16,
            use_batch_norm=False,
            activation=None
        )
        layer(input_tensor)

        # Only conv layer should be created
        assert layer.conv is not None
        assert layer.batch_norm is None
        assert layer.activation_layer is None

    # -------------------------------------------------------------------------
    # 3. Output Shape Tests
    # -------------------------------------------------------------------------

    def test_output_shapes(self, input_tensor):
        """Test that output shapes are computed correctly."""
        filters_to_test = [16, 32, 64, 128]

        for filters in filters_to_test:
            layer = BiasFreeConv1D(filters=filters)
            output = layer(input_tensor)

            # Check output shape (same padding preserves time dimension)
            expected_shape = list(input_tensor.shape)
            expected_shape[-1] = filters  # Only feature dim should change
            expected_shape = tuple(expected_shape)

            assert output.shape == expected_shape

            # Test compute_output_shape separately
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == expected_shape

    def test_output_shapes_different_kernel_sizes(self, input_tensor):
        """Test output shapes with different kernel sizes."""
        kernel_sizes = [1, 3, 5, 7, 9]

        for kernel_size in kernel_sizes:
            layer = BiasFreeConv1D(filters=32, kernel_size=kernel_size)
            output = layer(input_tensor)

            # With 'same' padding, time dimension should be preserved
            expected_shape = list(input_tensor.shape)
            expected_shape[-1] = 32
            assert output.shape == tuple(expected_shape)

    def test_compute_output_shape_before_build(self):
        """Test compute_output_shape before layer is built."""
        layer = BiasFreeConv1D(filters=64)
        input_shape = (None, 100, 3)

        output_shape = layer.compute_output_shape(input_shape)
        expected_shape = (None, 100, 64)
        assert output_shape == expected_shape

    def test_output_shapes_different_sequence_lengths(self):
        """Test output shapes with different sequence lengths."""
        sequence_lengths = [10, 50, 100, 200, 500]

        for seq_len in sequence_lengths:
            layer = BiasFreeConv1D(filters=32)
            test_input = tf.random.normal([2, seq_len, 3])
            output = layer(test_input)

            # Time dimension should be preserved with same padding
            expected_shape = (2, seq_len, 32)
            assert output.shape == expected_shape

    # -------------------------------------------------------------------------
    # 4. Forward Pass Tests
    # -------------------------------------------------------------------------

    def test_forward_pass_basic(self, input_tensor):
        """Test that forward pass produces valid outputs."""
        layer = BiasFreeConv1D(filters=32)
        output = layer(input_tensor)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))
        assert output.shape[0] == input_tensor.shape[0]  # Batch dim preserved
        assert output.shape[1] == input_tensor.shape[1]  # Time dim preserved

    def test_forward_pass_different_training_modes(self, input_tensor):
        """Test forward pass in training and inference modes."""
        layer = BiasFreeConv1D(filters=32)

        # Training mode
        output_train = layer(input_tensor, training=True)

        # Inference mode
        output_inference = layer(input_tensor, training=False)

        # Outputs should be different due to batch norm behavior
        # but both should be valid
        assert not np.any(np.isnan(output_train.numpy()))
        assert not np.any(np.isnan(output_inference.numpy()))
        assert output_train.shape == output_inference.shape

    def test_forward_pass_no_bias_property(self):
        """Test that the layer maintains bias-free property."""
        # Create controlled input
        controlled_input = tf.ones([1, 10, 2])

        # Layer with linear activation to test bias-free property
        layer = BiasFreeConv1D(
            filters=1,
            kernel_size=1,
            activation='linear',
            kernel_initializer='ones',
            use_batch_norm=False
        )

        output = layer(controlled_input)

        # With ones initializer and no bias, output should be predictable
        # Each output should be sum of input features (2 in this case)
        expected_value = 2.0  # Sum of input features

        # Check that output has expected pattern (approximately)
        output_values = output.numpy().flatten()
        assert np.allclose(output_values, expected_value, rtol=0.1)

    def test_forward_pass_different_activations(self, small_input_tensor):
        """Test forward pass with different activation functions."""
        activations = ['relu', 'gelu', 'swish', 'linear', None]

        for activation in activations:
            layer = BiasFreeConv1D(filters=16, activation=activation)
            output = layer(small_input_tensor)

            # Check output is valid
            assert not np.any(np.isnan(output.numpy()))
            assert output.shape[-1] == 16

    def test_forward_pass_temporal_consistency(self):
        """Test that the convolution respects temporal structure."""
        # Create input with known temporal pattern
        seq_len = 50
        test_input = tf.zeros([1, seq_len, 1])

        # Set a single time step to 1.0
        test_input = tf.tensor_scatter_nd_update(
            test_input,
            [[0, 25, 0]],
            [1.0]
        )

        layer = BiasFreeConv1D(
            filters=1,
            kernel_size=3,
            activation='linear',
            kernel_initializer='ones',
            use_batch_norm=False
        )

        output = layer(test_input)

        # The peak in output should be around the same time step
        peak_time = tf.argmax(output[0, :, 0]).numpy()
        assert abs(peak_time - 25) <= 1  # Allow for small shifts due to convolution

    # -------------------------------------------------------------------------
    # 5. Serialization Tests
    # -------------------------------------------------------------------------

    def test_serialization_basic(self):
        """Test basic serialization and deserialization."""
        original_layer = BiasFreeConv1D(
            filters=64,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_normal',
            use_batch_norm=True
        )

        # Build the layer
        input_shape = (None, 100, 3)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = BiasFreeConv1D.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.filters == original_layer.filters
        assert recreated_layer.kernel_size == original_layer.kernel_size
        assert recreated_layer.activation == original_layer.activation
        assert recreated_layer.use_batch_norm == original_layer.use_batch_norm

    def test_serialization_with_regularizers(self):
        """Test serialization with various regularizers."""
        original_layer = BiasFreeConv1D(
            filters=32,
            kernel_regularizer=keras.regularizers.L2(0.01),
            kernel_initializer='glorot_normal'
        )

        # Build and serialize
        original_layer.build((None, 50, 3))
        config = original_layer.get_config()

        # Recreate and check
        recreated_layer = BiasFreeConv1D.from_config(config)
        assert recreated_layer.filters == 32

    def test_get_build_config(self):
        """Test get_build_config returns correct information."""
        layer = BiasFreeConv1D(filters=32)
        input_shape = (None, 75, 3)
        layer.build(input_shape)

        build_config = layer.get_build_config()

        assert 'input_shape' in build_config
        assert build_config['input_shape'] == input_shape

    # -------------------------------------------------------------------------
    # 6. Model Integration Tests
    # -------------------------------------------------------------------------

    def test_model_integration_sequential(self, input_tensor):
        """Test the layer in a Sequential model."""
        model = keras.Sequential([
            keras.layers.Input(shape=input_tensor.shape[1:]),
            BiasFreeConv1D(filters=32, activation='relu'),
            BiasFreeConv1D(filters=64, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Test forward pass
        y_pred = model(input_tensor, training=False)
        assert y_pred.shape == (input_tensor.shape[0], 10)

    def test_model_integration_functional(self, input_tensor):
        """Test the layer in a Functional API model."""
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = BiasFreeConv1D(filters=32, name='bf_conv1')(inputs)
        x = BiasFreeConv1D(filters=64, name='bf_conv2')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        y_pred = model(input_tensor)
        assert y_pred.shape == (input_tensor.shape[0], 5)

    def test_model_integration_with_other_layers(self, input_tensor):
        """Test integration with standard Keras layers."""
        model = keras.Sequential([
            keras.layers.Input(shape=input_tensor.shape[1:]),
            keras.layers.Conv1D(16, 3, activation='relu'),
            BiasFreeConv1D(filters=32, activation='relu'),
            keras.layers.MaxPooling1D(2),
            BiasFreeConv1D(filters=64, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10)
        ])

        output = model(input_tensor)
        assert output.shape == (input_tensor.shape[0], 10)

    def test_time_series_forecasting_architecture(self, input_tensor):
        """Test in a time series forecasting architecture."""
        # Encoder
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = BiasFreeConv1D(filters=32, activation='relu')(inputs)
        x = BiasFreeConv1D(filters=64, activation='relu')(x)
        x = BiasFreeConv1D(filters=32, activation='relu')(x)

        # Output for forecasting
        outputs = keras.layers.Conv1D(1, 1, activation='linear')(x)

        model = keras.Model(inputs, outputs)

        output = model(input_tensor)
        # Output should have same time dimension, single feature
        assert output.shape == (input_tensor.shape[0], input_tensor.shape[1], 1)

    # -------------------------------------------------------------------------
    # 7. Model Save/Load Tests
    # -------------------------------------------------------------------------

    def test_model_save_load(self, input_tensor):
        """Test saving and loading a model with BiasFreeConv1D layers."""
        # Create model with custom layer
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = BiasFreeConv1D(filters=32, name='bf_conv1')(inputs)
        x = BiasFreeConv1D(filters=64, name='bf_conv2')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate prediction before saving
        original_prediction = model.predict(input_tensor, verbose=0)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'model.keras')

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'BiasFreeConv1D': BiasFreeConv1D
                }
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer types are preserved
            assert isinstance(loaded_model.get_layer('bf_conv1'), BiasFreeConv1D)
            assert isinstance(loaded_model.get_layer('bf_conv2'), BiasFreeConv1D)

    def test_model_save_load_with_training(self, input_tensor):
        """Test save/load after some training."""
        # Create simple model
        model = keras.Sequential([
            keras.layers.Input(shape=input_tensor.shape[1:]),
            BiasFreeConv1D(filters=16, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Create dummy training data
        y_train = np.random.randint(0, 2, size=(input_tensor.shape[0],))

        # Train for a few steps
        model.fit(input_tensor, y_train, epochs=1, verbose=0)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'trained_model.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'BiasFreeConv1D': BiasFreeConv1D}
            )

            # Compare predictions
            original_pred = model.predict(input_tensor, verbose=0)
            loaded_pred = loaded_model.predict(input_tensor, verbose=0)

            assert np.allclose(original_pred, loaded_pred, rtol=1e-5)

    # -------------------------------------------------------------------------
    # 8. Edge Cases and Robustness Tests
    # -------------------------------------------------------------------------

    def test_different_input_shapes(self):
        """Test layer with various input shapes."""
        # Test different temporal and feature dimensions
        test_shapes = [
            (1, 10, 1),    # Short sequence, single feature
            (4, 50, 3),    # Medium sequence, few features
            (2, 200, 16),  # Long sequence, many features
            (1, 1000, 1),  # Very long sequence
            (8, 25, 128)   # Many features
        ]

        for shape in test_shapes:
            # Create new layer instance for each shape
            layer = BiasFreeConv1D(filters=32)
            test_input = tf.random.normal(shape)
            output = layer(test_input)

            expected_shape = list(shape)
            expected_shape[-1] = 32
            assert output.shape == tuple(expected_shape)

    def test_extreme_filter_numbers(self):
        """Test with extreme numbers of filters."""
        test_input = tf.random.normal([1, 20, 3])

        # Very few filters
        layer_small = BiasFreeConv1D(filters=1)
        output_small = layer_small(test_input)
        assert output_small.shape[-1] == 1

        # Many filters
        layer_large = BiasFreeConv1D(filters=512)
        output_large = layer_large(test_input)
        assert output_large.shape[-1] == 512

    def test_extreme_kernel_sizes(self):
        """Test with extreme kernel sizes."""
        test_input = tf.random.normal([2, 50, 3])

        # Very small kernel
        layer_small = BiasFreeConv1D(filters=16, kernel_size=1)
        output_small = layer_small(test_input)
        assert output_small.shape[1] == test_input.shape[1]

        # Large kernel
        layer_large = BiasFreeConv1D(filters=16, kernel_size=15)
        output_large = layer_large(test_input)
        assert output_large.shape[1] == test_input.shape[1]  # Same padding

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        layer = BiasFreeConv1D(filters=16)

        # Test different input magnitudes
        test_cases = [
            tf.zeros((2, 20, 3)),  # Zeros
            tf.ones((2, 20, 3)) * 1e-10,  # Very small values
            tf.ones((2, 20, 3)) * 1e10,  # Very large values
            tf.random.normal((2, 20, 3)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            output = layer(test_input)

            # Check for NaN/Inf values
            assert not np.any(np.isnan(output.numpy())), "NaN values detected in output"
            assert not np.any(np.isinf(output.numpy())), "Inf values detected in output"

    def test_gradient_flow(self, input_tensor):
        """Test gradient flow through the layer."""
        layer = BiasFreeConv1D(filters=32)

        # Watch the variables
        with tf.GradientTape() as tape:
            inputs = tf.Variable(input_tensor)
            outputs = layer(inputs)
            loss = tf.reduce_mean(tf.square(outputs))

        # Get gradients
        grads = tape.gradient(loss, layer.trainable_variables)

        # Check gradients exist and are not None
        assert all(g is not None for g in grads)

        # Check gradients have values (not all zeros)
        assert all(np.any(g.numpy() != 0) for g in grads)


class TestBiasFreeResidualBlock1D:
    """Test suite for BiasFreeResidualBlock1D layer implementation."""

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create a test input tensor."""
        return tf.random.normal([4, 100, 3])

    @pytest.fixture
    def matching_channels_tensor(self) -> tf.Tensor:
        """Create input tensor with matching channel count."""
        return tf.random.normal([2, 50, 32])

    # -------------------------------------------------------------------------
    # 1. Initialization Tests
    # -------------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = BiasFreeResidualBlock1D(filters=64)

        assert layer.filters == 64
        assert layer.kernel_size == 3
        assert layer.activation == 'relu'

        # Check sublayers are initially None
        assert layer.conv1 is None
        assert layer.conv2 is None
        assert layer.shortcut_conv is None

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        layer = BiasFreeResidualBlock1D(
            filters=128,
            kernel_size=5,
            activation='gelu',
            kernel_initializer='he_normal'
        )

        assert layer.filters == 128
        assert layer.kernel_size == 5
        assert layer.activation == 'gelu'
        assert layer.kernel_initializer == 'he_normal'

    # -------------------------------------------------------------------------
    # 2. Build Process Tests
    # -------------------------------------------------------------------------

    def test_build_with_channel_mismatch(self, input_tensor):
        """Test building when input and output channels differ."""
        # Input has 3 features, output will have 64
        layer = BiasFreeResidualBlock1D(filters=64)
        layer(input_tensor)

        # Should create shortcut convolution
        assert layer.conv1 is not None
        assert layer.conv2 is not None
        assert layer.shortcut_conv is not None  # Should be created
        assert layer.add_layer is not None
        assert layer.final_activation is not None

    def test_build_with_matching_channels(self, matching_channels_tensor):
        """Test building when input and output channels match."""
        # Input has 32 features, output will have 32
        layer = BiasFreeResidualBlock1D(filters=32)
        layer(matching_channels_tensor)

        # Should not create shortcut convolution
        assert layer.conv1 is not None
        assert layer.conv2 is not None
        assert layer.shortcut_conv is None  # Should not be created
        assert layer.add_layer is not None

    def test_sublayer_properties(self, input_tensor):
        """Test that sublayers have correct properties."""
        layer = BiasFreeResidualBlock1D(filters=64)
        layer(input_tensor)

        # Check conv1 has activation
        assert layer.conv1.activation == 'relu'

        # Check conv2 has no activation (applied after addition)
        assert layer.conv2.activation is None

        # Check shortcut conv properties (if created)
        if layer.shortcut_conv is not None:
            assert layer.shortcut_conv.use_bias is False
            assert layer.shortcut_conv.kernel_size == (1,)  # Conv1D returns tuple

    # -------------------------------------------------------------------------
    # 3. Forward Pass Tests
    # -------------------------------------------------------------------------

    def test_forward_pass_with_shortcut(self, input_tensor):
        """Test forward pass when shortcut convolution is needed."""
        layer = BiasFreeResidualBlock1D(filters=64)
        output = layer(input_tensor)

        # Check output shape
        expected_shape = list(input_tensor.shape)
        expected_shape[-1] = 64
        assert output.shape == tuple(expected_shape)

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_forward_pass_without_shortcut(self, matching_channels_tensor):
        """Test forward pass when shortcut convolution is not needed."""
        layer = BiasFreeResidualBlock1D(filters=32)
        output = layer(matching_channels_tensor)

        # Output shape should match input
        assert output.shape == matching_channels_tensor.shape

        # Basic sanity checks
        assert not np.any(np.isnan(output.numpy()))
        assert not np.any(np.isinf(output.numpy()))

    def test_residual_connection_effect(self):
        """Test that residual connection has the expected effect."""
        # Create controlled input
        controlled_input = tf.ones([1, 20, 32])

        layer = BiasFreeResidualBlock1D(
            filters=32,
            activation='linear',  # Linear to see residual effect clearly
            kernel_initializer='zeros'  # Zero weights make conv output = 0
        )

        output = layer(controlled_input)

        # With zero weights in conv layers, output should approximately equal input
        # (residual connection preserves input)
        input_mean = tf.reduce_mean(controlled_input).numpy()
        output_mean = tf.reduce_mean(output).numpy()

        # They should be similar (allowing for batch norm effects)
        assert abs(output_mean - input_mean) < 2.0

    def test_training_vs_inference_modes(self, input_tensor):
        """Test different behavior in training vs inference."""
        layer = BiasFreeResidualBlock1D(filters=64)

        # Training mode
        output_train = layer(input_tensor, training=True)

        # Inference mode
        output_inference = layer(input_tensor, training=False)

        # Outputs should be different due to batch norm
        assert output_train.shape == output_inference.shape
        assert not np.allclose(output_train.numpy(), output_inference.numpy())

    def test_temporal_structure_preservation(self):
        """Test that residual block preserves temporal structure."""
        # Create input with temporal pattern
        seq_len = 100
        test_input = tf.zeros([1, seq_len, 32])

        # Add a pattern: peak at time step 50
        test_input = tf.tensor_scatter_nd_update(
            test_input,
            [[0, 50]],
            [tf.ones([32])]
        )

        layer = BiasFreeResidualBlock1D(filters=32)
        output = layer(test_input)

        # Check that the peak is preserved in the output
        output_max_time = tf.argmax(tf.reduce_mean(output[0], axis=-1)).numpy()
        assert abs(output_max_time - 50) <= 2  # Allow small shifts

    # -------------------------------------------------------------------------
    # 4. Output Shape Tests
    # -------------------------------------------------------------------------

    def test_output_shapes_various_filters(self, input_tensor):
        """Test output shapes with various filter numbers."""
        filter_counts = [16, 32, 64, 128, 256]

        for filters in filter_counts:
            layer = BiasFreeResidualBlock1D(filters=filters)
            output = layer(input_tensor)

            expected_shape = list(input_tensor.shape)
            expected_shape[-1] = filters
            assert output.shape == tuple(expected_shape)

            # Test compute_output_shape
            computed_shape = layer.compute_output_shape(input_tensor.shape)
            assert computed_shape == tuple(expected_shape)

    def test_output_shapes_different_kernel_sizes(self, input_tensor):
        """Test output shapes with different kernel sizes."""
        kernel_sizes = [1, 3, 5, 7, 9]

        for kernel_size in kernel_sizes:
            layer = BiasFreeResidualBlock1D(filters=32, kernel_size=kernel_size)
            output = layer(input_tensor)

            # Time dimension should be preserved
            expected_shape = list(input_tensor.shape)
            expected_shape[-1] = 32
            assert output.shape == tuple(expected_shape)

    # -------------------------------------------------------------------------
    # 5. Serialization Tests
    # -------------------------------------------------------------------------

    def test_serialization(self):
        """Test serialization and deserialization."""
        original_layer = BiasFreeResidualBlock1D(
            filters=64,
            kernel_size=3,
            activation='relu'
        )

        # Build the layer
        input_shape = (None, 100, 16)
        original_layer.build(input_shape)

        # Get configs
        config = original_layer.get_config()
        build_config = original_layer.get_build_config()

        # Recreate the layer
        recreated_layer = BiasFreeResidualBlock1D.from_config(config)
        recreated_layer.build_from_config(build_config)

        # Check configuration matches
        assert recreated_layer.filters == original_layer.filters
        assert recreated_layer.kernel_size == original_layer.kernel_size
        assert recreated_layer.activation == original_layer.activation

    # -------------------------------------------------------------------------
    # 6. Model Integration Tests
    # -------------------------------------------------------------------------

    def test_model_integration(self, input_tensor):
        """Test the residual block in a model."""
        model = keras.Sequential([
            keras.layers.Input(shape=input_tensor.shape[1:]),
            BiasFreeResidualBlock1D(filters=32),
            BiasFreeResidualBlock1D(filters=64),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(10)
        ])

        output = model(input_tensor)
        assert output.shape == (input_tensor.shape[0], 10)

    def test_resnet_style_architecture(self, input_tensor):
        """Test building a ResNet-style architecture for time series."""
        inputs = keras.Input(shape=input_tensor.shape[1:])

        # Initial conv
        x = keras.layers.Conv1D(32, 7, padding='same', use_bias=False)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        # Residual blocks
        x = BiasFreeResidualBlock1D(filters=32)(x)
        x = BiasFreeResidualBlock1D(filters=32)(x)
        x = BiasFreeResidualBlock1D(filters=64)(x)  # Channel increase
        x = BiasFreeResidualBlock1D(filters=64)(x)

        # Output
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs, outputs)

        prediction = model(input_tensor)
        assert prediction.shape == (input_tensor.shape[0], 10)

    def test_time_series_autoencoder_architecture(self, input_tensor):
        """Test in a time series autoencoder architecture."""
        # Encoder
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = BiasFreeResidualBlock1D(filters=64)(inputs)
        x = keras.layers.MaxPooling1D(2)(x)
        x = BiasFreeResidualBlock1D(filters=128)(x)
        encoded = keras.layers.MaxPooling1D(2)(x)

        # Decoder
        x = keras.layers.UpSampling1D(2)(encoded)
        x = BiasFreeResidualBlock1D(filters=64)(x)
        x = keras.layers.UpSampling1D(2)(x)
        x = BiasFreeResidualBlock1D(filters=32)(x)

        # Output layer to match input shape
        outputs = keras.layers.Conv1D(
            input_tensor.shape[-1],
            1,
            activation='linear'
        )(x)

        model = keras.Model(inputs, outputs)

        output = model(input_tensor)
        assert output.shape == input_tensor.shape

    # -------------------------------------------------------------------------
    # 7. Model Save/Load Tests
    # -------------------------------------------------------------------------

    def test_model_save_load_residual_block(self, input_tensor):
        """Test saving and loading model with residual blocks."""
        # Create model
        model = keras.Sequential([
            keras.layers.Input(shape=input_tensor.shape[1:]),
            BiasFreeResidualBlock1D(filters=32, name='residual_1'),
            BiasFreeResidualBlock1D(filters=64, name='residual_2'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(5)
        ])

        # Get original prediction
        original_pred = model.predict(input_tensor, verbose=0)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'residual_model.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={'BiasFreeResidualBlock1D': BiasFreeResidualBlock1D}
            )

            loaded_pred = loaded_model.predict(input_tensor, verbose=0)

            # Check predictions match
            assert np.allclose(original_pred, loaded_pred, rtol=1e-5)

            # Check layer types
            assert isinstance(loaded_model.get_layer('residual_1'), BiasFreeResidualBlock1D)
            assert isinstance(loaded_model.get_layer('residual_2'), BiasFreeResidualBlock1D)


class TestBiasFreeLayersIntegration1D:
    """Integration tests for both BiasFree 1D layers working together."""

    @pytest.fixture
    def input_tensor(self) -> tf.Tensor:
        """Create test input tensor."""
        return tf.random.normal([2, 100, 3])

    def test_mixed_architecture(self, input_tensor):
        """Test using both layer types in the same model."""
        inputs = keras.Input(shape=input_tensor.shape[1:])

        # Mix both layer types
        x = BiasFreeConv1D(filters=32, activation='relu')(inputs)
        x = BiasFreeConv1D(filters=32, activation='relu')(x)
        x = BiasFreeResidualBlock1D(filters=64)(x)
        x = BiasFreeConv1D(filters=64, activation='relu')(x)
        x = BiasFreeResidualBlock1D(filters=128)(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(10)(x)

        model = keras.Model(inputs, outputs)

        # Test forward pass
        prediction = model(input_tensor)
        assert prediction.shape == (input_tensor.shape[0], 10)

        # Test training capability
        model.compile(optimizer='adam', loss='mse')
        dummy_targets = tf.random.normal([input_tensor.shape[0], 10])

        # Should be able to train without errors
        history = model.fit(
            input_tensor, dummy_targets,
            epochs=1, verbose=0, validation_split=0.2
        )

        assert len(history.history['loss']) == 1

    def test_bias_free_property_preservation(self, input_tensor):
        """Test that bias-free property is maintained through the architecture."""
        # Create model with both layer types
        model = keras.Sequential([
            keras.layers.Input(shape=input_tensor.shape[1:]),
            BiasFreeConv1D(filters=16, use_batch_norm=False, activation='linear'),
            BiasFreeResidualBlock1D(filters=16),
            BiasFreeConv1D(filters=16, use_batch_norm=False, activation='linear')
        ])

        # Check that conv layers in the model don't have bias
        conv_layers = []
        for layer in model.layers:
            if isinstance(layer, (BiasFreeConv1D, BiasFreeResidualBlock1D)):
                # Build layer if not built
                if not layer.built:
                    layer.build(input_tensor.shape)

                if hasattr(layer, 'conv') and layer.conv is not None:
                    conv_layers.append(layer.conv)

                if hasattr(layer, 'conv1') and layer.conv1 is not None:
                    conv_layers.append(layer.conv1.conv)
                    conv_layers.append(layer.conv2.conv)

                if hasattr(layer, 'shortcut_conv') and layer.shortcut_conv is not None:
                    conv_layers.append(layer.shortcut_conv)

        # All conv layers should have use_bias=False
        for conv_layer in conv_layers:
            assert conv_layer.use_bias is False, f"Layer {conv_layer.name} has bias enabled"

    def test_save_load_mixed_model(self, input_tensor):
        """Test saving and loading model with both layer types."""
        # Create mixed model
        inputs = keras.Input(shape=input_tensor.shape[1:])
        x = BiasFreeConv1D(filters=16, name='bf_conv')(inputs)
        x = BiasFreeResidualBlock1D(filters=32, name='bf_residual')(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(5)(x)

        model = keras.Model(inputs, outputs)
        original_pred = model.predict(input_tensor, verbose=0)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'mixed_model.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'BiasFreeConv1D': BiasFreeConv1D,
                    'BiasFreeResidualBlock1D': BiasFreeResidualBlock1D
                }
            )

            loaded_pred = loaded_model.predict(input_tensor, verbose=0)

            # Predictions should match
            assert np.allclose(original_pred, loaded_pred, rtol=1e-5)

            # Check layer types preserved
            assert isinstance(loaded_model.get_layer('bf_conv'), BiasFreeConv1D)
            assert isinstance(loaded_model.get_layer('bf_residual'), BiasFreeResidualBlock1D)

    def test_time_series_denoising_architecture(self, input_tensor):
        """Test full time series denoising architecture."""
        # Add noise to input for denoising test
        noisy_input = input_tensor + tf.random.normal(input_tensor.shape) * 0.1

        # Create denoising model
        inputs = keras.Input(shape=input_tensor.shape[1:])

        # Encoder
        x = BiasFreeConv1D(filters=32, activation='relu')(inputs)
        x = BiasFreeResidualBlock1D(filters=32)(x)
        x = BiasFreeConv1D(filters=64, activation='relu')(x)
        x = BiasFreeResidualBlock1D(filters=64)(x)

        # Decoder
        x = BiasFreeConv1D(filters=32, activation='relu')(x)
        x = BiasFreeResidualBlock1D(filters=32)(x)
        x = BiasFreeConv1D(filters=16, activation='relu')(x)

        # Output layer (same shape as input)
        outputs = keras.layers.Conv1D(
            input_tensor.shape[-1],
            1,
            activation='linear',
            use_bias=False  # Keep bias-free
        )(x)

        model = keras.Model(inputs, outputs)

        # Test forward pass
        denoised = model(noisy_input)
        assert denoised.shape == input_tensor.shape

        # Test training
        model.compile(optimizer='adam', loss='mse')

        # Train model to denoise (using clean as target)
        history = model.fit(
            noisy_input, input_tensor,
            epochs=1, verbose=0, batch_size=1
        )

        assert len(history.history['loss']) == 1


# -------------------------------------------------------------------------
# Helper Functions for Testing
# -------------------------------------------------------------------------

def test_bias_free_conv1d_instantiation():
    """Test that we can import and instantiate the layers."""
    # This test ensures the imports work correctly
    layer1 = BiasFreeConv1D(filters=32)
    layer2 = BiasFreeResidualBlock1D(filters=64)

    assert isinstance(layer1, keras.layers.Layer)
    assert isinstance(layer2, keras.layers.Layer)


def test_compare_1d_vs_2d_bias_free_properties():
    """Test that 1D and 2D versions have similar bias-free properties."""
    # This conceptual test shows the similarity
    conv1d = BiasFreeConv1D(filters=32, use_batch_norm=False, activation=None)

    # Build and check properties
    conv1d.build((None, 100, 3))

    # Should have no bias
    assert conv1d.conv.use_bias is False
    assert conv1d.batch_norm is None  # Disabled
    assert conv1d.activation_layer is None  # Disabled


if __name__ == '__main__':
    # Run the tests
    pytest.main([__file__])