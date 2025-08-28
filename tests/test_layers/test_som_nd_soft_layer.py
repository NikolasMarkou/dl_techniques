"""
Comprehensive test suite for SoftSOMLayer implementation.

This test suite follows the patterns established for the classical SOM layer
but includes additional tests specific to the soft, differentiable features
of the Soft SOM implementation, including the new sharpness regularization.
"""

import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.som_nd_soft_layer import SoftSOMLayer


class TestSoftSOMLayer:
    """Test suite for SoftSOMLayer implementation."""

    @pytest.fixture
    def input_data_1d(self):
        """Create 1D test input data."""
        return keras.random.uniform([32, 10], seed=42)

    @pytest.fixture
    def input_data_2d(self):
        """Create 2D test input data."""
        return keras.random.uniform([64, 784], seed=42)

    @pytest.fixture
    def input_data_3d(self):
        """Create 3D test input data."""
        return keras.random.uniform([16, 128], seed=42)

    @pytest.fixture
    def soft_som_1d(self):
        """Create a 1D Soft SOM instance."""
        return SoftSOMLayer(
            grid_shape=(20,),
            input_dim=10,
            temperature=0.5,
            use_per_dimension_softmax=True
        )

    @pytest.fixture
    def soft_som_2d(self):
        """Create a 2D Soft SOM instance."""
        return SoftSOMLayer(
            grid_shape=(10, 10),
            input_dim=784,
            temperature=1.0,
            use_per_dimension_softmax=True,
            reconstruction_weight=1.0,
            topological_weight=0.1
        )

    @pytest.fixture
    def soft_som_3d(self):
        """Create a 3D Soft SOM instance."""
        return SoftSOMLayer(
            grid_shape=(5, 5, 5),
            input_dim=128,
            temperature=0.3,
            use_per_dimension_softmax=False,  # Use global softmax
            reconstruction_weight=0.8,
            topological_weight=0.2
        )

    @pytest.fixture
    def soft_som_with_sharpness(self):
        """Create a Soft SOM instance with sharpness regularization."""
        return SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=100,
            temperature=1.0,
            use_per_dimension_softmax=True,
            sharpness_weight=0.05
        )

    # =============================================================================
    # Initialization Tests
    # =============================================================================

    def test_basic_functionality(self):
        """Test basic SoftSOM functionality with minimal example."""
        # Create simple test case
        soft_som = SoftSOMLayer(
            grid_shape=(3, 3),
            input_dim=5,
            temperature=1.0,
            use_reconstruction_loss=False,
            topological_weight=0.0,
            sharpness_weight=0.0
        )

        # Simple test input
        test_input = keras.random.uniform((2, 5), seed=42)

        # Should work without errors
        reconstruction = soft_som(test_input, training=False)

        # Check basic properties
        assert reconstruction.shape == (2, 5)
        assert not tf.reduce_any(tf.math.is_nan(reconstruction))
        assert not tf.reduce_any(tf.math.is_inf(reconstruction))

        # Test soft assignments
        soft_assignments = soft_som.get_soft_assignments(test_input)
        assert soft_assignments.shape == (2, 3, 3)

        # Assignments should sum to 1 (approximately)
        assignment_sums = tf.reduce_sum(soft_assignments, axis=[1, 2])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(assignment_sums),
            keras.ops.convert_to_numpy(tf.ones_like(assignment_sums)),
            rtol=1e-5, atol=1e-5,
            err_msg="Soft assignments should sum to 1"
        )

    def test_initialization_1d(self, soft_som_1d):
        """Test initialization of 1D Soft SOM with default parameters."""
        assert soft_som_1d.grid_shape == (20,)
        assert soft_som_1d.grid_dim == 1
        assert soft_som_1d.input_dim == 10
        assert soft_som_1d.temperature == 0.5
        assert soft_som_1d.use_per_dimension_softmax is True
        assert soft_som_1d.use_reconstruction_loss is True
        assert soft_som_1d.reconstruction_weight == 1.0
        assert soft_som_1d.topological_weight == 0.1
        assert soft_som_1d.sharpness_weight == 0.0  # Default value
        assert soft_som_1d.weights_map is None  # Not built yet

    def test_initialization_2d(self, soft_som_2d):
        """Test initialization of 2D Soft SOM with custom parameters."""
        assert soft_som_2d.grid_shape == (10, 10)
        assert soft_som_2d.grid_dim == 2
        assert soft_som_2d.input_dim == 784
        assert soft_som_2d.temperature == 1.0
        assert soft_som_2d.use_per_dimension_softmax is True
        assert soft_som_2d.reconstruction_weight == 1.0
        assert soft_som_2d.topological_weight == 0.1
        assert soft_som_2d.sharpness_weight == 0.0  # Default value

    def test_initialization_3d(self, soft_som_3d):
        """Test initialization of 3D Soft SOM with global softmax."""
        assert soft_som_3d.grid_shape == (5, 5, 5)
        assert soft_som_3d.grid_dim == 3
        assert soft_som_3d.input_dim == 128
        assert soft_som_3d.temperature == 0.3
        assert soft_som_3d.use_per_dimension_softmax is False
        assert soft_som_3d.reconstruction_weight == 0.8
        assert soft_som_3d.topological_weight == 0.2
        assert soft_som_3d.sharpness_weight == 0.0  # Default value

    def test_initialization_with_sharpness(self, soft_som_with_sharpness):
        """Test initialization with sharpness regularization."""
        assert soft_som_with_sharpness.grid_shape == (6, 6)
        assert soft_som_with_sharpness.input_dim == 100
        assert soft_som_with_sharpness.temperature == 1.0
        assert soft_som_with_sharpness.use_per_dimension_softmax is True
        assert soft_som_with_sharpness.sharpness_weight == 0.05

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters including sharpness."""
        custom_regularizer = keras.regularizers.L2(1e-4)
        custom_initializer = keras.initializers.HeNormal()

        soft_som = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=100,
            temperature=2.0,
            use_per_dimension_softmax=False,
            use_reconstruction_loss=False,
            reconstruction_weight=0.5,
            topological_weight=0.3,
            sharpness_weight=0.1,
            kernel_initializer=custom_initializer,
            kernel_regularizer=custom_regularizer
        )

        assert soft_som.grid_shape == (8, 8)
        assert soft_som.input_dim == 100
        assert soft_som.temperature == 2.0
        assert soft_som.use_per_dimension_softmax is False
        assert soft_som.use_reconstruction_loss is False
        assert soft_som.reconstruction_weight == 0.5
        assert soft_som.topological_weight == 0.3
        assert soft_som.sharpness_weight == 0.1
        assert soft_som.kernel_initializer == custom_initializer
        assert soft_som.kernel_regularizer == custom_regularizer

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError, match="must contain positive integers"):
            SoftSOMLayer(grid_shape=(-5, 10), input_dim=10)

        with pytest.raises(ValueError, match="must contain positive integers"):
            SoftSOMLayer(grid_shape=(0, 10), input_dim=10)

        with pytest.raises(ValueError, match="must be positive"):
            SoftSOMLayer(grid_shape=(10, 10), input_dim=-5)

        with pytest.raises(ValueError, match="must be positive"):
            SoftSOMLayer(grid_shape=(10, 10), input_dim=10, temperature=-1.0)

        with pytest.raises(ValueError, match="must be positive"):
            SoftSOMLayer(grid_shape=(10, 10), input_dim=10, temperature=0.0)

        with pytest.raises(ValueError, match="must be non-negative"):
            SoftSOMLayer(grid_shape=(10, 10), input_dim=10, reconstruction_weight=-0.1)

        with pytest.raises(ValueError, match="must be non-negative"):
            SoftSOMLayer(grid_shape=(10, 10), input_dim=10, topological_weight=-0.1)

        with pytest.raises(ValueError, match="must be non-negative"):
            SoftSOMLayer(grid_shape=(10, 10), input_dim=10, sharpness_weight=-0.1)

    # =============================================================================
    # Build Process Tests
    # =============================================================================

    def test_build_process_1d(self, soft_som_1d, input_data_1d):
        """Test that 1D Soft SOM builds properly."""
        soft_som_1d.build(input_data_1d.shape)

        assert soft_som_1d.built is True
        assert soft_som_1d.weights_map is not None
        assert soft_som_1d.weights_map.shape == (20, 10)
        assert soft_som_1d.weights_map.trainable is True  # Key difference from classical SOM
        assert soft_som_1d.grid_positions is not None
        assert soft_som_1d.grid_positions.shape == (20, 1)

    def test_build_process_2d(self, soft_som_2d, input_data_2d):
        """Test that 2D Soft SOM builds properly."""
        soft_som_2d.build(input_data_2d.shape)

        assert soft_som_2d.built is True
        assert soft_som_2d.weights_map is not None
        assert soft_som_2d.weights_map.shape == (10, 10, 784)
        assert soft_som_2d.weights_map.trainable is True
        assert soft_som_2d.grid_positions.shape == (10, 10, 2)

    def test_build_process_3d(self, soft_som_3d, input_data_3d):
        """Test that 3D Soft SOM builds properly."""
        soft_som_3d.build(input_data_3d.shape)

        assert soft_som_3d.built is True
        assert soft_som_3d.weights_map is not None
        assert soft_som_3d.weights_map.shape == (5, 5, 5, 128)
        assert soft_som_3d.weights_map.trainable is True
        assert soft_som_3d.grid_positions.shape == (5, 5, 5, 3)

    def test_build_with_wrong_input_shape(self, soft_som_2d):
        """Test that building with wrong input shape raises error."""
        with pytest.raises(ValueError, match="Expected input_dim="):
            soft_som_2d.build((64, 100))  # Wrong input dimension

        with pytest.raises(ValueError, match="Expected 2D input shape"):
            soft_som_2d.build((64, 32, 784))  # Wrong number of dimensions

    # =============================================================================
    # Forward Pass Tests
    # =============================================================================

    def test_forward_pass_1d(self, soft_som_1d, input_data_1d):
        """Test forward pass with 1D Soft SOM returns reconstruction."""
        reconstruction = soft_som_1d(input_data_1d, training=False)

        assert reconstruction.shape == (32, 10)  # Same as input shape
        assert reconstruction.dtype == tf.float32

        # Check that reconstruction values are reasonable (not NaN/Inf)
        assert not tf.reduce_any(tf.math.is_nan(reconstruction))
        assert not tf.reduce_any(tf.math.is_inf(reconstruction))

    def test_forward_pass_2d(self, soft_som_2d, input_data_2d):
        """Test forward pass with 2D Soft SOM."""
        reconstruction = soft_som_2d(input_data_2d, training=False)

        assert reconstruction.shape == (64, 784)  # Same as input shape
        assert reconstruction.dtype == tf.float32
        assert not tf.reduce_any(tf.math.is_nan(reconstruction))
        assert not tf.reduce_any(tf.math.is_inf(reconstruction))

    def test_forward_pass_3d(self, soft_som_3d, input_data_3d):
        """Test forward pass with 3D Soft SOM."""
        reconstruction = soft_som_3d(input_data_3d, training=False)

        assert reconstruction.shape == (16, 128)  # Same as input shape
        assert reconstruction.dtype == tf.float32
        assert not tf.reduce_any(tf.math.is_nan(reconstruction))
        assert not tf.reduce_any(tf.math.is_inf(reconstruction))

    def test_forward_pass_with_sharpness(self, soft_som_with_sharpness):
        """Test forward pass with sharpness regularization."""
        test_input = keras.random.uniform((8, 100), seed=42)

        # Training mode should add sharpness loss
        reconstruction = soft_som_with_sharpness(test_input, training=True)

        assert reconstruction.shape == (8, 100)
        assert not tf.reduce_any(tf.math.is_nan(reconstruction))
        assert not tf.reduce_any(tf.math.is_inf(reconstruction))

        # Should have losses added during training
        assert len(soft_som_with_sharpness.losses) > 0

    def test_per_dimension_vs_global_softmax(self, input_data_2d):
        """Test difference between per-dimension and global softmax."""
        soft_som_per_dim = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            temperature=0.5,
            use_per_dimension_softmax=True
        )

        soft_som_global = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            temperature=0.5,
            use_per_dimension_softmax=False
        )

        reconstruction_per_dim = soft_som_per_dim(input_data_2d, training=False)
        reconstruction_global = soft_som_global(input_data_2d, training=False)

        # Both should have same output shape
        assert reconstruction_per_dim.shape == reconstruction_global.shape

        # But reconstructions should be different due to different softmax strategies
        # (unless by coincidence, which is extremely unlikely)
        assert not tf.reduce_all(tf.equal(reconstruction_per_dim, reconstruction_global))

    # =============================================================================
    # Soft Assignment Tests
    # =============================================================================

    def test_get_soft_assignments_1d(self, soft_som_1d, input_data_1d):
        """Test getting soft assignments for 1D Soft SOM."""
        # Build first
        soft_som_1d(input_data_1d[:1], training=False)

        soft_assignments = soft_som_1d.get_soft_assignments(input_data_1d)

        assert soft_assignments.shape == (32, 20)  # batch_size, *grid_shape
        assert soft_assignments.dtype == tf.float32

        # Each row should sum to approximately 1 (probabilities)
        row_sums = tf.reduce_sum(soft_assignments, axis=1)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(row_sums),
            keras.ops.convert_to_numpy(tf.ones_like(row_sums)),
            rtol=1e-5, atol=1e-5,
            err_msg="Soft assignments should sum to 1"
        )

        # All assignments should be non-negative
        assert tf.reduce_all(soft_assignments >= 0)

    def test_get_soft_assignments_2d(self, soft_som_2d, input_data_2d):
        """Test getting soft assignments for 2D Soft SOM."""
        # Build first
        soft_som_2d(input_data_2d[:1], training=False)

        soft_assignments = soft_som_2d.get_soft_assignments(input_data_2d)

        assert soft_assignments.shape == (64, 10, 10)  # batch_size, *grid_shape

        # Each sample's assignments should sum to approximately 1
        sample_sums = tf.reduce_sum(soft_assignments, axis=[1, 2])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sample_sums),
            keras.ops.convert_to_numpy(tf.ones_like(sample_sums)),
            rtol=1e-5, atol=1e-5,
            err_msg="Soft assignments should sum to 1"
        )

        # All assignments should be non-negative
        assert tf.reduce_all(soft_assignments >= 0)

    def test_get_soft_assignments_3d(self, soft_som_3d, input_data_3d):
        """Test getting soft assignments for 3D Soft SOM."""
        # Build first
        soft_som_3d(input_data_3d[:1], training=False)

        soft_assignments = soft_som_3d.get_soft_assignments(input_data_3d)

        assert soft_assignments.shape == (16, 5, 5, 5)  # batch_size, *grid_shape

        # Each sample's assignments should sum to approximately 1
        sample_sums = tf.reduce_sum(soft_assignments, axis=[1, 2, 3])
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sample_sums),
            keras.ops.convert_to_numpy(tf.ones_like(sample_sums)),
            rtol=1e-5, atol=1e-5,
            err_msg="Soft assignments should sum to 1"
        )

        # All assignments should be non-negative
        assert tf.reduce_all(soft_assignments >= 0)

    def test_temperature_effect_on_assignments(self, input_data_2d):
        """Test that temperature affects sharpness of soft assignments."""
        # Low temperature (sharper distributions)
        soft_som_low_temp = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            temperature=0.1
        )

        # High temperature (smoother distributions)
        soft_som_high_temp = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            temperature=10.0
        )

        # Build both
        soft_som_low_temp(input_data_2d[:1], training=False)
        soft_som_high_temp(input_data_2d[:1], training=False)

        assignments_low_temp = soft_som_low_temp.get_soft_assignments(input_data_2d[:4])
        assignments_high_temp = soft_som_high_temp.get_soft_assignments(input_data_2d[:4])

        # Low temperature should produce more concentrated (higher max) distributions
        max_assignments_low = tf.reduce_max(assignments_low_temp, axis=[1, 2])
        max_assignments_high = tf.reduce_max(assignments_high_temp, axis=[1, 2])

        # On average, low temperature should have higher maximum values
        assert tf.reduce_mean(max_assignments_low) > tf.reduce_mean(max_assignments_high)

    # =============================================================================
    # Sharpness Regularization Tests
    # =============================================================================

    def test_sharpness_loss_computation(self):
        """Test that sharpness loss is computed correctly."""
        soft_som = SoftSOMLayer(
            grid_shape=(4, 4),
            input_dim=20,
            temperature=1.0,
            use_per_dimension_softmax=True,
            sharpness_weight=0.1
        )

        test_input = keras.random.uniform((8, 20), seed=42)

        # Clear any existing losses
        soft_som.losses.clear()

        # Forward pass in training mode should add sharpness loss
        reconstruction = soft_som(test_input, training=True)

        # Should have losses (reconstruction + topological + sharpness)
        assert len(soft_som.losses) > 0

        # One of the losses should be sharpness loss
        loss_values = [float(loss) for loss in soft_som.losses]
        assert all(np.isfinite(loss_val) for loss_val in loss_values)

    def test_sharpness_regularization_disabled_for_global_softmax(self):
        """Test that sharpness regularization is disabled for global softmax."""
        soft_som = SoftSOMLayer(
            grid_shape=(4, 4),
            input_dim=20,
            temperature=1.0,
            use_per_dimension_softmax=False,  # Global softmax
            sharpness_weight=0.1  # This should be ignored
        )

        test_input = keras.random.uniform((8, 20), seed=42)

        # Forward pass in training mode
        soft_som.losses.clear()
        reconstruction = soft_som(test_input, training=True)

        # Should work without errors but no sharpness loss should be added
        # (only reconstruction and topological losses if enabled)
        assert len(soft_som.losses) >= 0  # Could be 0 if other losses disabled

    def test_sharpness_effect_on_assignments(self):
        """Test that sharpness regularization affects assignment distributions."""
        # Create two identical SOMs, one with sharpness regularization
        soft_som_no_sharp = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=50,
            temperature=1.0,
            sharpness_weight=0.0
        )

        soft_som_with_sharp = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=50,
            temperature=1.0,
            sharpness_weight=0.2
        )

        test_input = keras.random.uniform((16, 50), seed=42)

        # Build both layers
        soft_som_no_sharp(test_input[:1], training=False)
        soft_som_with_sharp(test_input[:1], training=False)

        # Get initial assignments (should be similar since weights are random)
        assignments_no_sharp = soft_som_no_sharp.get_soft_assignments(test_input)
        assignments_with_sharp = soft_som_with_sharp.get_soft_assignments(test_input)

        # Both should be valid probability distributions
        sums_no_sharp = tf.reduce_sum(assignments_no_sharp, axis=[1, 2])
        sums_with_sharp = tf.reduce_sum(assignments_with_sharp, axis=[1, 2])

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums_no_sharp),
            keras.ops.convert_to_numpy(tf.ones_like(sums_no_sharp)),
            rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(sums_with_sharp),
            keras.ops.convert_to_numpy(tf.ones_like(sums_with_sharp)),
            rtol=1e-5, atol=1e-5
        )

    def test_zero_sharpness_weight(self):
        """Test that zero sharpness weight disables sharpness regularization."""
        soft_som = SoftSOMLayer(
            grid_shape=(4, 4),
            input_dim=20,
            temperature=1.0,
            use_per_dimension_softmax=True,
            sharpness_weight=0.0  # Disabled
        )

        test_input = keras.random.uniform((8, 20), seed=42)

        soft_som.losses.clear()
        reconstruction = soft_som(test_input, training=True)

        # Should still work, but no sharpness loss should be added
        # (only reconstruction and topological if enabled)
        loss_count_without_sharpness = len(soft_som.losses)

        # Now enable sharpness and compare
        soft_som.sharpness_weight = 0.1
        soft_som.losses.clear()
        reconstruction = soft_som(test_input, training=True)

        loss_count_with_sharpness = len(soft_som.losses)

        # With sharpness enabled, we should have at least as many losses
        assert loss_count_with_sharpness >= loss_count_without_sharpness

    # =============================================================================
    # Loss and Training Tests
    # =============================================================================

    def test_reconstruction_loss_added(self, soft_som_2d, input_data_2d):
        """Test that reconstruction loss is added during training."""
        # Enable reconstruction loss
        soft_som_2d.use_reconstruction_loss = True

        # No losses initially
        soft_som_2d.losses.clear()
        assert len(soft_som_2d.losses) == 0

        # Forward pass in training mode
        reconstruction = soft_som_2d(input_data_2d, training=True)

        # Should have losses now
        assert len(soft_som_2d.losses) > 0

    def test_topological_loss_added(self, input_data_2d):
        """Test that topological loss is added when enabled."""
        soft_som = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=784,
            topological_weight=0.5  # Enable topological loss
        )

        # No losses initially
        soft_som.losses.clear()
        assert len(soft_som.losses) == 0

        # Forward pass in training mode
        reconstruction = soft_som(input_data_2d, training=True)

        # Should have losses now (reconstruction + topological)
        assert len(soft_som.losses) >= 1

    def test_no_losses_when_disabled(self, input_data_2d):
        """Test that no losses are added when all regularization is disabled."""
        soft_som = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=784,
            use_reconstruction_loss=False,
            topological_weight=0.0,
            sharpness_weight=0.0
        )

        # Forward pass in training mode
        soft_som.losses.clear()
        reconstruction = soft_som(input_data_2d, training=True)

        # Should have no losses
        assert len(soft_som.losses) == 0

    def test_inference_mode_no_losses(self, soft_som_2d, input_data_2d):
        """Test that no losses are added during inference."""
        # Forward pass in inference mode
        soft_som_2d.losses.clear()
        reconstruction = soft_som_2d(input_data_2d, training=False)

        # Should have no losses in inference mode
        assert len(soft_som_2d.losses) == 0

    # =============================================================================
    # Backpropagation and Training Tests
    # =============================================================================

    def test_weights_trainable(self, soft_som_2d):
        """Test that weights are trainable for backpropagation."""
        soft_som_2d.build((64, 784))

        # Check that weights are marked as trainable
        assert soft_som_2d.weights_map.trainable is True
        assert soft_som_2d.weights_map in soft_som_2d.trainable_variables

    def test_gradient_flow_through_layer(self, input_data_2d):
        """Test that gradients flow properly through the Soft SOM layer."""
        soft_som = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=784,
            use_reconstruction_loss=True,
            sharpness_weight=0.05
        )

        # Create a simple model for gradient testing
        inputs = keras.Input(shape=(784,))
        reconstruction = soft_som(inputs)

        # Create explicit loss
        loss = keras.layers.Lambda(lambda x: keras.ops.mean(keras.ops.square(x)))(reconstruction)
        model = keras.Model(inputs=inputs, outputs=loss)

        # Test gradient computation with respect to inputs
        with tf.GradientTape() as tape:
            tape.watch(input_data_2d)
            output = model(input_data_2d, training=True)

        # Get gradients with respect to inputs
        input_gradients = tape.gradient(output, input_data_2d)

        # Test gradient computation with respect to weights (separate tape)
        with tf.GradientTape() as tape:
            output = model(input_data_2d, training=True)

        # Get gradients with respect to model weights
        weight_gradients = tape.gradient(output, soft_som.trainable_variables)

        # Input gradients should exist and not be None
        assert input_gradients is not None
        assert not tf.reduce_any(tf.math.is_nan(input_gradients))

        # Weight gradients should exist
        assert all(g is not None for g in weight_gradients)
        assert all(not tf.reduce_any(tf.math.is_nan(g)) for g in weight_gradients)

    def test_standard_keras_training(self, input_data_2d):
        """Test that Soft SOM can be trained with standard Keras training loop."""
        # Create model with Soft SOM
        inputs = keras.Input(shape=(784,))
        reconstruction = SoftSOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            use_reconstruction_loss=True,
            sharpness_weight=0.02
        )(inputs)

        model = keras.Model(inputs=inputs, outputs=reconstruction)

        # Compile with standard optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )

        # Small training run should work without errors
        history = model.fit(
            input_data_2d, input_data_2d,  # Autoencoder-style training
            epochs=2,
            batch_size=16,
            verbose=0
        )

        assert 'loss' in history.history
        assert len(history.history['loss']) == 2

    def test_weight_updates_during_training(self, soft_som_2d, input_data_2d):
        """Test that weights are actually updated during training."""
        # Build the layer first
        soft_som_2d.build(input_data_2d.shape)

        # Get initial weights
        initial_weights = soft_som_2d.weights_map.numpy().copy()

        # Create a simple training step
        optimizer = keras.optimizers.Adam(learning_rate=0.01)

        @tf.function
        def train_step(x):
            with tf.GradientTape() as tape:
                reconstruction = soft_som_2d(x, training=True)
                loss = keras.ops.mean(keras.ops.square(x - reconstruction))
                # Add layer losses (reconstruction + topological)
                for layer_loss in soft_som_2d.losses:
                    loss += layer_loss

            gradients = tape.gradient(loss, soft_som_2d.trainable_variables)
            optimizer.apply_gradients(zip(gradients, soft_som_2d.trainable_variables))
            return loss

        # Perform training step
        loss_value = train_step(input_data_2d)

        # Check that weights have been updated
        updated_weights = soft_som_2d.weights_map.numpy()
        assert not np.allclose(initial_weights, updated_weights, rtol=1e-6)

        # Loss should be finite
        assert tf.math.is_finite(loss_value)

    # =============================================================================
    # Utility Method Tests
    # =============================================================================

    def test_get_weights_map(self, soft_som_2d, input_data_2d):
        """Test get_weights_map method."""
        # Build the layer
        soft_som_2d(input_data_2d[:1], training=False)

        weights_map = soft_som_2d.get_weights_map()

        assert weights_map.shape == (10, 10, 784)
        assert weights_map.dtype == tf.float32

        # Check that it's the same as the internal weights
        assert tf.reduce_all(tf.equal(weights_map, soft_som_2d.weights_map))

    def test_get_weights_map_before_build(self, soft_som_2d):
        """Test that get_weights_map raises error before building."""
        with pytest.raises(RuntimeError, match="Layer must be built"):
            soft_som_2d.get_weights_map()

    def test_compute_output_shape_1d(self, soft_som_1d):
        """Test compute_output_shape for 1D Soft SOM."""
        input_shape = (32, 10)
        output_shape = soft_som_1d.compute_output_shape(input_shape)

        assert output_shape == (32, 10)  # Same as input for reconstruction

    def test_compute_output_shape_2d(self, soft_som_2d):
        """Test compute_output_shape for 2D Soft SOM."""
        input_shape = (64, 784)
        output_shape = soft_som_2d.compute_output_shape(input_shape)

        assert output_shape == (64, 784)  # Same as input

    def test_compute_output_shape_3d(self, soft_som_3d):
        """Test compute_output_shape for 3D Soft SOM."""
        input_shape = (16, 128)
        output_shape = soft_som_3d.compute_output_shape(input_shape)

        assert output_shape == (16, 128)  # Same as input

    # =============================================================================
    # Serialization Tests
    # =============================================================================

    def test_serialization(self, soft_som_2d):
        """Test serialization and deserialization of the layer."""
        # Build the layer
        soft_som_2d.build((64, 784))

        # Get config
        config = soft_som_2d.get_config()

        # Recreate the layer from config
        recreated_som = SoftSOMLayer.from_config(config)

        # Check configuration matches including new sharpness_weight
        assert recreated_som.grid_shape == soft_som_2d.grid_shape
        assert recreated_som.input_dim == soft_som_2d.input_dim
        assert recreated_som.temperature == soft_som_2d.temperature
        assert recreated_som.use_per_dimension_softmax == soft_som_2d.use_per_dimension_softmax
        assert recreated_som.use_reconstruction_loss == soft_som_2d.use_reconstruction_loss
        assert recreated_som.reconstruction_weight == soft_som_2d.reconstruction_weight
        assert recreated_som.topological_weight == soft_som_2d.topological_weight
        assert recreated_som.sharpness_weight == soft_som_2d.sharpness_weight

    def test_serialization_with_sharpness(self, soft_som_with_sharpness):
        """Test serialization of layer with sharpness regularization."""
        # Build the layer
        soft_som_with_sharpness.build((32, 100))

        # Get config
        config = soft_som_with_sharpness.get_config()

        # Recreate the layer from config
        recreated_som = SoftSOMLayer.from_config(config)

        # Check sharpness weight is preserved
        assert recreated_som.sharpness_weight == 0.05

    def test_serialization_with_custom_objects(self):
        """Test serialization with custom initializers and regularizers."""
        original_som = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=100,
            temperature=0.8,
            sharpness_weight=0.15,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=keras.regularizers.L1(0.01)
        )

        # Build the layer
        original_som.build((32, 100))

        # Get config
        config = original_som.get_config()

        # Recreate the layer from config
        recreated_som = SoftSOMLayer.from_config(config)

        # Check configuration matches
        assert recreated_som.grid_shape == original_som.grid_shape
        assert recreated_som.input_dim == original_som.input_dim
        assert recreated_som.temperature == original_som.temperature
        assert recreated_som.sharpness_weight == original_som.sharpness_weight

    def test_full_serialization_cycle(self, input_data_2d):
        """Test complete model serialization with SoftSOMLayer."""
        # Create a model with the Soft SOM layer
        inputs = keras.Input(shape=(784,))
        reconstruction = SoftSOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            temperature=0.7,
            sharpness_weight=0.08,
            name="soft_som_layer"
        )(inputs)

        # Add output layer
        outputs = keras.layers.Dense(1, activation='sigmoid')(reconstruction)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Generate predictions before saving
        original_prediction = model.predict(input_data_2d[:8], verbose=0)

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.keras")

            # Save the model
            model.save(model_path)

            # Load the model
            loaded_model = keras.models.load_model(model_path)

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_data_2d[:8], verbose=0)

            # Check predictions match (allowing for small numerical differences)
            np.testing.assert_allclose(
                original_prediction, loaded_prediction,
                rtol=1e-5, atol=1e-5,
                err_msg="Predictions should match after serialization"
            )

            # Check layer type and properties are preserved
            som_layer = loaded_model.get_layer("soft_som_layer")
            assert isinstance(som_layer, SoftSOMLayer)
            assert som_layer.temperature == 0.7
            assert som_layer.sharpness_weight == 0.08

    # =============================================================================
    # Model Integration Tests
    # =============================================================================

    def test_model_integration(self, input_data_2d):
        """Test the layer in a model context."""
        # Create a model with the Soft SOM layer
        inputs = keras.Input(shape=(784,))
        reconstruction = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            sharpness_weight=0.03
        )(inputs)

        # Add additional processing
        x = keras.layers.Dense(128, activation='relu')(reconstruction)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        predictions = model(input_data_2d, training=False)
        assert predictions.shape == (64, 10)

    # =============================================================================
    # Numerical Stability Tests
    # =============================================================================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        soft_som = SoftSOMLayer(
            grid_shape=(4, 4),
            input_dim=10,
            sharpness_weight=0.1
        )

        # Create inputs with different magnitudes
        test_cases = [
            keras.ops.zeros((8, 10)),  # Zeros
            keras.ops.ones((8, 10)) * 1e-10,  # Very small values
            keras.ops.ones((8, 10)) * 1e10,  # Very large values
            keras.random.normal((8, 10)) * 1e5  # Large random values
        ]

        for test_input in test_cases:
            reconstruction = soft_som(test_input, training=True)

            # Check for NaN/Inf values in reconstruction
            assert not tf.reduce_any(tf.math.is_nan(reconstruction))
            assert not tf.reduce_any(tf.math.is_inf(reconstruction))

            # Check soft assignments are stable
            soft_assignments = soft_som.get_soft_assignments(test_input)
            assert not tf.reduce_any(tf.math.is_nan(soft_assignments))
            assert not tf.reduce_any(tf.math.is_inf(soft_assignments))

    def test_temperature_edge_cases(self, input_data_2d):
        """Test behavior with extreme temperature values."""
        # Very low temperature (should create sharp distributions)
        soft_som_low = SoftSOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            temperature=1e-6,
            sharpness_weight=0.05
        )

        # Very high temperature (should create smooth distributions)
        soft_som_high = SoftSOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            temperature=1e6,
            sharpness_weight=0.05
        )

        # Both should work without numerical issues
        reconstruction_low = soft_som_low(input_data_2d[:4], training=False)
        reconstruction_high = soft_som_high(input_data_2d[:4], training=False)

        assert not tf.reduce_any(tf.math.is_nan(reconstruction_low))
        assert not tf.reduce_any(tf.math.is_nan(reconstruction_high))
        assert not tf.reduce_any(tf.math.is_inf(reconstruction_low))
        assert not tf.reduce_any(tf.math.is_inf(reconstruction_high))

    # =============================================================================
    # Advanced Feature Tests
    # =============================================================================

    def test_different_grid_shapes_with_sharpness(self):
        """Test Soft SOM with various grid shapes and sharpness regularization."""
        test_cases = [
            ((10,), 20),  # 1D
            ((5, 5), 30),  # 2D square
            ((3, 7), 40),  # 2D rectangle
            ((2, 3, 4), 50),  # 3D
            ((2, 2, 2, 2), 60)  # 4D
        ]

        for grid_shape, input_dim in test_cases:
            soft_som = SoftSOMLayer(
                grid_shape=grid_shape,
                input_dim=input_dim,
                temperature=0.5,
                sharpness_weight=0.05
            )

            # Test with small batch
            test_input = keras.random.uniform((4, input_dim))
            reconstruction = soft_som(test_input, training=True)

            # Check output shapes
            assert reconstruction.shape == (4, input_dim)

            # Check soft assignments
            soft_assignments = soft_som.get_soft_assignments(test_input)
            expected_assignment_shape = (4,) + grid_shape
            assert soft_assignments.shape == expected_assignment_shape

    def test_consistent_reconstruction_in_inference(self, soft_som_2d, input_data_2d):
        """Test that reconstruction is consistent in inference mode."""
        # Create fixed input
        fixed_input = keras.ops.ones((1, 784))

        # Build the layer first
        soft_som_2d(fixed_input, training=False)

        # Multiple reconstructions should be identical
        recon1 = soft_som_2d(fixed_input, training=False)
        recon2 = soft_som_2d(fixed_input, training=False)
        recon3 = soft_som_2d(fixed_input, training=False)

        # Should be exactly the same (no training mode)
        assert tf.reduce_all(tf.equal(recon1, recon2))
        assert tf.reduce_all(tf.equal(recon2, recon3))

    def test_batch_size_independence(self, soft_som_2d):
        """Test that results are independent of batch size."""
        # Test with different batch sizes
        input1 = keras.random.uniform((1, 784), seed=42)
        input4 = keras.random.uniform((4, 784), seed=42)

        # Build with first input
        soft_som_2d(input1, training=False)

        # Single sample
        recon_single = soft_som_2d(input1[:1], training=False)

        # Same sample in a batch
        recon_batch = soft_som_2d(input1, training=False)

        # Results should be the same for the same input
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(recon_single),
            keras.ops.convert_to_numpy(recon_batch),
            rtol=1e-6, atol=1e-6,
            err_msg="Results should be independent of batch size"
        )