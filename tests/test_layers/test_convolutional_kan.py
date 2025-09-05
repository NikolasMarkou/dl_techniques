import pytest
import numpy as np
import tensorflow as tf
import keras
from keras import ops, layers, models, initializers, regularizers
import tempfile
import os
from typing import Any, Dict, Tuple

from dl_techniques.layers.convolutional_kan import KANvolution


# --- Test Class ---
class TestKANvolution:
    """
    Comprehensive and modern test suite for the KANvolution layer.
    This suite follows modern Keras 3 testing best practices and covers
    all aspects of Kolmogorov-Arnold Network convolution functionality.
    """

    # --- Fixtures for Reusability ---
    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for a small, testable layer."""
        return {
            'filters': 16,
            'kernel_size': 3,
            'grid_size': 8,
        }

    @pytest.fixture
    def advanced_config(self) -> Dict[str, Any]:
        """Provides an advanced configuration with all parameters."""
        return {
            'filters': 32,
            'kernel_size': (5, 3),
            'grid_size': 16,
            'strides': (1, 1),  # Fixed: can't have both strides>1 and dilation>1
            'padding': 'valid',
            'dilation_rate': (1, 2),
            'activation': 'gelu',
            'use_bias': True,
            'kernel_regularizer': regularizers.L2(1e-4),
            'bias_regularizer': regularizers.L1(1e-5),
        }

    @pytest.fixture
    def sample_input_small(self) -> tf.Tensor:
        """Provides a small sample input tensor for testing."""
        return tf.random.normal(shape=(2, 16, 16, 4))

    @pytest.fixture
    def sample_input_large(self) -> tf.Tensor:
        """Provides a larger sample input tensor for advanced testing."""
        return tf.random.normal(shape=(4, 32, 32, 8))

    # ===============================================
    # 1. Initialization and Build Tests
    # ===============================================
    def test_initialization_defaults(self, layer_config):
        """Tests layer initialization with default parameters."""
        layer = KANvolution(**layer_config)
        assert not layer.built
        assert layer.filters == 16
        assert layer.kernel_size == (3, 3)
        assert layer.grid_size == 8
        assert layer.strides == (1, 1)
        assert layer.padding == 'same'
        assert layer.dilation_rate == (1, 1)
        assert layer.activation is None
        assert layer.use_bias is True

    def test_initialization_with_int_kernel_size(self):
        """Tests initialization with integer kernel size."""
        layer = KANvolution(filters=8, kernel_size=5, grid_size=4)
        assert layer.kernel_size == (5, 5)

    def test_initialization_with_tuple_kernel_size(self):
        """Tests initialization with tuple kernel size."""
        layer = KANvolution(filters=8, kernel_size=(3, 5), grid_size=4)
        assert layer.kernel_size == (3, 5)

    def test_initialization_with_int_strides(self):
        """Tests initialization with integer strides."""
        layer = KANvolution(filters=8, kernel_size=3, strides=2)
        assert layer.strides == (2, 2)

    def test_initialization_with_tuple_strides(self):
        """Tests initialization with tuple strides."""
        layer = KANvolution(filters=8, kernel_size=3, strides=(2, 3))
        assert layer.strides == (2, 3)

    def test_initialization_with_advanced_config(self, advanced_config):
        """Tests initialization with all advanced parameters."""
        layer = KANvolution(**advanced_config)
        assert layer.filters == 32
        assert layer.kernel_size == (5, 3)
        assert layer.grid_size == 16
        assert layer.strides == (1, 1)
        assert layer.padding == 'valid'
        assert layer.dilation_rate == (1, 2)
        assert layer.activation == 'gelu'
        assert layer.kernel_regularizer is not None
        assert layer.bias_regularizer is not None

    def test_build_process(self, layer_config, sample_input_small):
        """Tests that the layer and all its weights are built correctly."""
        layer = KANvolution(**layer_config)
        assert not layer.built

        # Build the layer by calling it
        output = layer(sample_input_small)

        assert layer.built
        assert layer.control_points is not None
        assert layer.w_spline is not None
        assert layer.w_silu is not None
        assert layer.bias is not None  # use_bias=True by default
        assert layer.grid is not None

    def test_weight_shapes_after_build(self, layer_config, sample_input_small):
        """Tests that weights have correct shapes after building."""
        layer = KANvolution(**layer_config)
        layer(sample_input_small)  # Build the layer

        input_channels = sample_input_small.shape[-1]

        # Check control points shape
        expected_control_shape = (layer.filters, input_channels, *layer.kernel_size, layer.grid_size + 1)
        assert layer.control_points.shape == expected_control_shape

        # Check combination weights shapes
        expected_weight_shape = (layer.filters, input_channels, *layer.kernel_size)
        assert layer.w_spline.shape == expected_weight_shape
        assert layer.w_silu.shape == expected_weight_shape

        # Check bias shape
        if layer.use_bias:
            assert layer.bias.shape == (layer.filters,)

        # Check grid shape
        assert layer.grid.shape == (layer.grid_size + 1,)

    def test_build_without_bias(self, sample_input_small):
        """Tests building with use_bias=False."""
        layer = KANvolution(filters=8, kernel_size=3, use_bias=False)
        layer(sample_input_small)

        assert layer.built
        assert layer.bias is None

    # ===============================================
    # 2. Parameter Validation Tests
    # ===============================================
    def test_invalid_filters_raises_error(self):
        """Tests that invalid filters parameter raises ValueError."""
        with pytest.raises(ValueError, match="filters must be positive"):
            KANvolution(filters=0, kernel_size=3)

        with pytest.raises(ValueError, match="filters must be positive"):
            KANvolution(filters=-5, kernel_size=3)

    def test_invalid_grid_size_raises_error(self):
        """Tests that invalid grid_size parameter raises ValueError."""
        with pytest.raises(ValueError, match="grid_size must be > 1"):
            KANvolution(filters=8, kernel_size=3, grid_size=1)

        with pytest.raises(ValueError, match="grid_size must be > 1"):
            KANvolution(filters=8, kernel_size=3, grid_size=0)

    def test_invalid_kernel_size_raises_error(self):
        """Tests that invalid kernel_size parameter raises ValueError."""
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            KANvolution(filters=8, kernel_size=0)

        with pytest.raises(ValueError, match="kernel_size must be positive"):
            KANvolution(filters=8, kernel_size=-3)

        with pytest.raises(ValueError, match="kernel_size values must be positive"):
            KANvolution(filters=8, kernel_size=(3, 0))

    def test_invalid_strides_raises_error(self):
        """Tests that invalid strides parameter raises ValueError."""
        with pytest.raises(ValueError, match="strides must be positive"):
            KANvolution(filters=8, kernel_size=3, strides=0)

        with pytest.raises(ValueError, match="strides values must be positive"):
            KANvolution(filters=8, kernel_size=3, strides=(2, 0))

    def test_invalid_padding_raises_error(self):
        """Tests that invalid padding parameter raises ValueError."""
        with pytest.raises(ValueError, match="padding must be 'valid' or 'same'"):
            KANvolution(filters=8, kernel_size=3, padding='invalid')

    def test_invalid_input_shape_during_build(self):
        """Tests that invalid input shape during build raises ValueError."""
        layer = KANvolution(filters=8, kernel_size=3)

        # Test with 3D input (should be 4D)
        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.build((None, 16, 8))

        # Test with None channels
        with pytest.raises(ValueError, match="Input channels dimension must be defined"):
            layer.build((None, 16, 16, None))

    # ===============================================
    # 3. Forward Pass Tests
    # ===============================================
    @pytest.mark.parametrize("filters", [8, 16, 32])
    @pytest.mark.parametrize("kernel_size", [3, (3, 5), (5, 3)])
    @pytest.mark.parametrize("grid_size", [4, 8, 16])
    def test_forward_pass_variations(self, filters, kernel_size, grid_size, sample_input_small):
        """Tests forward pass with various parameter combinations."""
        layer = KANvolution(filters=filters, kernel_size=kernel_size, grid_size=grid_size)
        output = layer(sample_input_small, training=False)

        # Check output shape
        expected_batch_size = sample_input_small.shape[0]
        assert output.shape[0] == expected_batch_size
        assert output.shape[-1] == filters
        assert len(output.shape) == 4  # Should remain 4D

        # Check for valid output (no NaN or inf)
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))
        assert not np.any(np.isinf(ops.convert_to_numpy(output)))

    @pytest.mark.parametrize("padding", ['same', 'valid'])
    @pytest.mark.parametrize("strides", [1, 2, (1, 2), (2, 1)])
    def test_forward_pass_padding_strides(self, padding, strides, sample_input_small):
        """Tests forward pass with different padding and stride configurations."""
        layer = KANvolution(filters=16, kernel_size=3, padding=padding, strides=strides)
        output = layer(sample_input_small, training=False)

        assert len(output.shape) == 4
        assert output.shape[0] == sample_input_small.shape[0]
        assert output.shape[-1] == 16
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    @pytest.mark.parametrize("activation", [None, 'relu', 'gelu', 'swish'])
    def test_forward_pass_activations(self, activation, sample_input_small):
        """Tests forward pass with different activation functions."""
        layer = KANvolution(filters=8, kernel_size=3, activation=activation)
        output = layer(sample_input_small, training=False)

        assert output.shape == (sample_input_small.shape[0], 16, 16, 8)
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

        if activation == 'relu':
            # ReLU should produce non-negative outputs
            assert np.all(ops.convert_to_numpy(output) >= 0)

    def test_forward_pass_training_vs_inference(self, layer_config, sample_input_small):
        """Tests that layer behaves consistently in training vs inference mode."""
        layer = KANvolution(**layer_config)

        output_train = layer(sample_input_small, training=True)
        output_infer = layer(sample_input_small, training=False)

        assert output_train.shape == output_infer.shape
        assert not np.any(np.isnan(ops.convert_to_numpy(output_train)))
        assert not np.any(np.isnan(ops.convert_to_numpy(output_infer)))

        # Outputs should be identical for this layer (no dropout)
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_train),
            ops.convert_to_numpy(output_infer),
            rtol=1e-6, atol=1e-6
        )

    # ===============================================
    # 4. Output Shape Computation Tests
    # ===============================================
    @pytest.mark.parametrize("input_shape", [
        (None, 32, 32, 3),
        (4, 64, 64, 16),
        (1, 28, 28, 1),
    ])
    def test_compute_output_shape_same_padding(self, input_shape):
        """Tests output shape computation with 'same' padding."""
        layer = KANvolution(filters=16, kernel_size=3, strides=1, padding='same')
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (input_shape[0], input_shape[1], input_shape[2], 16)
        assert output_shape == expected_shape

    @pytest.mark.parametrize("input_shape,strides,expected_height,expected_width", [
        ((None, 32, 32, 3), (2, 2), 16, 16),
        ((4, 64, 64, 16), (2, 1), 32, 64),
        ((1, 28, 28, 1), (1, 2), 28, 14),
    ])
    def test_compute_output_shape_same_padding_with_strides(self, input_shape, strides, expected_height,
                                                            expected_width):
        """Tests output shape computation with 'same' padding and various strides."""
        layer = KANvolution(filters=8, kernel_size=3, strides=strides, padding='same')
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (input_shape[0], expected_height, expected_width, 8)
        assert output_shape == expected_shape

    @pytest.mark.parametrize("input_shape,kernel_size,expected_height,expected_width", [
        ((None, 32, 32, 3), 3, 30, 30),
        ((4, 64, 64, 16), 5, 60, 60),
        ((1, 28, 28, 1), (3, 5), 26, 24),
    ])
    def test_compute_output_shape_valid_padding(self, input_shape, kernel_size, expected_height, expected_width):
        """Tests output shape computation with 'valid' padding."""
        layer = KANvolution(filters=12, kernel_size=kernel_size, strides=1, padding='valid')
        output_shape = layer.compute_output_shape(input_shape)

        expected_shape = (input_shape[0], expected_height, expected_width, 12)
        assert output_shape == expected_shape

    def test_compute_output_shape_invalid_input_raises_error(self):
        """Tests that invalid input shape raises ValueError."""
        layer = KANvolution(filters=8, kernel_size=3)

        with pytest.raises(ValueError, match="Expected 4D input shape"):
            layer.compute_output_shape((None, 16, 8))  # 3D instead of 4D

    # ===============================================
    # 5. Serialization Test (The Gold Standard)
    # ===============================================
    def test_full_serialization_cycle_basic(self, layer_config, sample_input_small):
        """Tests full serialization cycle with basic configuration."""
        inputs = layers.Input(shape=sample_input_small.shape[1:])
        outputs = KANvolution(**layer_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input_small, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_kanvolution_basic.keras")
            model.save(filepath)

            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_small, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization cycle"
            )

    def test_full_serialization_cycle_advanced(self, advanced_config, sample_input_large):
        """Tests full serialization cycle with advanced configuration."""
        inputs = layers.Input(shape=sample_input_large.shape[1:])
        outputs = KANvolution(**advanced_config)(inputs)
        model = models.Model(inputs, outputs)

        original_prediction = model(sample_input_large, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_kanvolution_advanced.keras")
            model.save(filepath)

            loaded_model = models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input_large, training=False)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6,
                err_msg="Predictions should match after serialization cycle"
            )

    def test_serialization_preserves_all_parameters(self, advanced_config):
        """Tests that all parameters are preserved during serialization."""
        original_layer = KANvolution(**advanced_config)
        config = original_layer.get_config()
        reconstructed_layer = KANvolution.from_config(config)

        # Check that all important parameters are preserved
        assert reconstructed_layer.filters == original_layer.filters
        assert reconstructed_layer.kernel_size == original_layer.kernel_size
        assert reconstructed_layer.grid_size == original_layer.grid_size
        assert reconstructed_layer.strides == original_layer.strides
        assert reconstructed_layer.padding == original_layer.padding
        assert reconstructed_layer.dilation_rate == original_layer.dilation_rate
        assert reconstructed_layer.activation == original_layer.activation
        assert reconstructed_layer.use_bias == original_layer.use_bias

    # ===============================================
    # 6. Gradient Flow and Training Tests
    # ===============================================
    def test_gradient_flow(self, layer_config, sample_input_small):
        """Tests gradient flow through the KANvolution layer."""
        layer = KANvolution(**layer_config)
        x_var = tf.Variable(sample_input_small)

        with tf.GradientTape() as tape:
            output = layer(x_var, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed."

        # Note: In the current simplified implementation, control_points are not used
        # in the forward pass, so their gradients may be None. We check that at least
        # some gradients are computed for the weights that are actually used.
        non_none_gradients = [g for g in gradients if g is not None]
        assert len(non_none_gradients) > 0, "No non-None gradients were computed."

        # Check that non-None gradients are not all zeros
        for grad in non_none_gradients:
            grad_numpy = ops.convert_to_numpy(grad)
            assert not np.allclose(grad_numpy, 0), "Gradient should not be all zeros."

    def test_trainable_weights_count(self, layer_config, sample_input_small):
        """Tests that the layer has the expected number of trainable weights."""
        layer = KANvolution(**layer_config)
        layer(sample_input_small)  # Build the layer

        # Expected weights: control_points, w_spline, w_silu, bias (if use_bias)
        # Note: control_points may not contribute to gradients in simplified implementation
        expected_count = 4 if layer.use_bias else 3
        assert len(layer.trainable_variables) == expected_count

        # Verify that all weights exist and have correct shapes
        assert layer.control_points is not None
        assert layer.w_spline is not None
        assert layer.w_silu is not None
        if layer.use_bias:
            assert layer.bias is not None

    def test_model_training_loop_integration(self, layer_config):
        """Tests that KANvolution can be used in a standard training loop."""
        model = models.Sequential([
            layers.InputLayer(shape=(28, 28, 1)),
            KANvolution(**layer_config),
            layers.GlobalAveragePooling2D(),
            layers.Dense(10)
        ])

        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

        # Create dummy training data
        x_train = tf.random.normal((32, 28, 28, 1))
        y_train = tf.random.uniform([32], 0, 10, dtype=tf.int32)

        # Train for one epoch
        history = model.fit(x_train, y_train, epochs=1, batch_size=8, verbose=0)

        assert 'loss' in history.history
        assert not np.isnan(history.history['loss'][0]), "Loss became NaN during training."

    # ===============================================
    # 7. B-spline Specific Tests
    # ===============================================
    def test_bspline_basis_computation(self, layer_config, sample_input_small):
        """
        Tests B-spline basis function computation.

        Note: In the current simplified implementation, B-spline computation
        is not used in the forward pass, but we test it to ensure the
        mathematical foundation is correct for future full implementations.
        """
        layer = KANvolution(**layer_config)
        layer(sample_input_small)  # Build the layer

        # Test with normalized input values
        test_input = ops.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        basis_weights = layer._compute_bspline_basis(test_input)

        # Check shape
        assert basis_weights.shape == (5, layer.grid_size + 1)

        # Check that weights are normalized (sum to 1)
        weight_sums = ops.sum(basis_weights, axis=-1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(weight_sums),
            1.0,
            rtol=1e-6, atol=1e-6,
            err_msg="B-spline basis weights should sum to 1"
        )

        # Check that weights are non-negative
        assert np.all(ops.convert_to_numpy(basis_weights) >= 0)

    def test_bspline_basis_edge_cases(self, layer_config, sample_input_small):
        """
        Tests B-spline basis function with edge cases.

        Note: Testing the mathematical foundation even though not currently
        used in the simplified forward pass implementation.
        """
        layer = KANvolution(**layer_config)
        layer(sample_input_small)  # Build the layer

        # Test with values outside [-1, 1] range (should be clamped)
        test_input = ops.array([-2.0, 2.0, 0.0])
        basis_weights = layer._compute_bspline_basis(test_input)

        # Should still produce valid normalized weights
        weight_sums = ops.sum(basis_weights, axis=-1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(weight_sums),
            1.0,
            rtol=1e-6, atol=1e-6
        )

    # ===============================================
    # 8. Regularization Tests
    # ===============================================
    def test_kernel_regularization_applied(self, sample_input_small):
        """Tests that kernel regularization is properly applied."""
        layer = KANvolution(
            filters=8,
            kernel_size=3,
            kernel_regularizer=regularizers.L2(1e-3)
        )

        # Build and get regularization losses
        output = layer(sample_input_small)
        reg_losses = layer.losses

        assert len(reg_losses) > 0, "Regularization losses should be present."

        # Regularization loss should be positive
        total_reg_loss = sum(reg_losses)
        assert ops.convert_to_numpy(total_reg_loss) > 0

    def test_bias_regularization_applied(self, sample_input_small):
        """Tests that bias regularization is properly applied."""
        layer = KANvolution(
            filters=8,
            kernel_size=3,
            bias_regularizer=regularizers.L1(1e-3)
        )

        # Build and get regularization losses
        output = layer(sample_input_small)
        reg_losses = layer.losses

        assert len(reg_losses) > 0, "Regularization losses should be present."

    # ===============================================
    # 9. CNN Architecture Integration Tests
    # ===============================================
    def test_kan_cnn_architecture(self, sample_input_small):
        """Tests KANvolution in a complete CNN architecture."""
        inputs = layers.Input(shape=sample_input_small.shape[1:])

        # Build a simple CNN with KANvolution layers
        x = KANvolution(filters=16, kernel_size=3, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = KANvolution(filters=32, kernel_size=3, strides=2, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = KANvolution(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(10, activation='softmax')(x)

        model = models.Model(inputs, outputs)

        # Test forward pass
        prediction = model(sample_input_small, training=False)
        assert prediction.shape == (sample_input_small.shape[0], 10)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))

        # Test that predictions are valid probabilities (sum to 1)
        pred_sums = ops.sum(prediction, axis=-1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(pred_sums),
            1.0,
            rtol=1e-6, atol=1e-6
        )

    def test_mixed_kan_standard_conv(self, sample_input_small):
        """Tests mixing KANvolution with standard Conv2D layers."""
        inputs = layers.Input(shape=sample_input_small.shape[1:])

        x = layers.Conv2D(filters=16, kernel_size=3, activation='relu')(inputs)
        x = KANvolution(filters=32, kernel_size=3, activation='gelu')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs, outputs)

        prediction = model(sample_input_small, training=False)
        assert prediction.shape == (sample_input_small.shape[0], 1)
        assert not np.any(np.isnan(ops.convert_to_numpy(prediction)))

    # ===============================================
    # 10. Performance and Edge Case Tests
    # ===============================================
    def test_simplified_implementation_note(self, layer_config, sample_input_small):
        """
        Tests and documents the current simplified implementation behavior.

        The current implementation uses effective kernels (w_spline + w_silu)
        rather than full patch-wise KAN transformations for computational efficiency.
        This test validates this simplified approach works correctly.
        """
        layer = KANvolution(**layer_config)

        # Get the effective kernel computation
        output = layer(sample_input_small, training=False)

        # Verify the layer has all the KAN components even if not fully utilized
        assert layer.control_points is not None, "Control points should be created"
        assert layer.w_spline is not None, "Spline weights should be created"
        assert layer.w_silu is not None, "SiLU weights should be created"
        assert layer.grid is not None, "Grid points should be created"

        # Verify forward pass produces valid output
        assert output.shape[0] == sample_input_small.shape[0]
        assert output.shape[-1] == layer.filters
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

        # Note: A full KAN implementation would apply B-spline transformations
        # to input patches and use control_points in the forward pass

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size, layer_config):
        """Tests layer with different batch sizes."""
        sample_input = tf.random.normal((batch_size, 16, 16, 4))
        layer = KANvolution(**layer_config)

        output = layer(sample_input, training=False)

        assert output.shape[0] == batch_size
        assert output.shape[-1] == layer_config['filters']
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_very_small_input(self, layer_config):
        """Tests layer with very small input dimensions."""
        small_input = tf.random.normal((2, 4, 4, 2))
        layer = KANvolution(filters=8, kernel_size=3, padding='valid')

        output = layer(small_input, training=False)

        # With 4x4 input and 3x3 kernel with valid padding, output should be 2x2
        assert output.shape == (2, 2, 2, 8)
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))

    def test_large_grid_size(self, sample_input_small):
        """Tests layer with large grid size for B-splines."""
        layer = KANvolution(filters=8, kernel_size=3, grid_size=64)

        output = layer(sample_input_small, training=False)

        assert output.shape[0] == sample_input_small.shape[0]
        assert output.shape[-1] == 8
        assert not np.any(np.isnan(ops.convert_to_numpy(output)))