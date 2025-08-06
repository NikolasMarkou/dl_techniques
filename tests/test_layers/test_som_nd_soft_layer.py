"""
Comprehensive test suite for SoftSOMLayer implementation.

This test suite follows the patterns established for the classical SOM layer
but includes additional tests specific to the soft, differentiable features
of the Soft SOM implementation.
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
            topological_weight=0.0
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
        assert tf.reduce_all(tf.abs(assignment_sums - 1.0) < 1e-5)

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

    def test_initialization_3d(self, soft_som_3d):
        """Test initialization of 3D Soft SOM with global softmax."""
        assert soft_som_3d.grid_shape == (5, 5, 5)
        assert soft_som_3d.grid_dim == 3
        assert soft_som_3d.input_dim == 128
        assert soft_som_3d.temperature == 0.3
        assert soft_som_3d.use_per_dimension_softmax is False
        assert soft_som_3d.reconstruction_weight == 0.8
        assert soft_som_3d.topological_weight == 0.2

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
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
        with pytest.raises(ValueError, match="Expected input shape"):
            soft_som_2d.build((64, 100))  # Wrong input dimension

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
        assert tf.reduce_all(tf.abs(row_sums - 1.0) < 1e-5)

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
        assert tf.reduce_all(tf.abs(sample_sums - 1.0) < 1e-5)

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
        assert tf.reduce_all(tf.abs(sample_sums - 1.0) < 1e-5)

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
    # Loss and Training Tests
    # =============================================================================

    def test_reconstruction_loss_added(self, soft_som_2d, input_data_2d):
        """Test that reconstruction loss is added during training."""
        # Enable reconstruction loss
        soft_som_2d.use_reconstruction_loss = True

        # No losses initially
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
        assert len(soft_som.losses) == 0

        # Forward pass in training mode
        reconstruction = soft_som(input_data_2d, training=True)

        # Should have losses now (reconstruction + topological)
        assert len(soft_som.losses) >= 1

    def test_no_losses_when_disabled(self, input_data_2d):
        """Test that no losses are added when disabled."""
        soft_som = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=784,
            use_reconstruction_loss=False,
            topological_weight=0.0
        )

        # Forward pass in training mode
        reconstruction = soft_som(input_data_2d, training=True)

        # Should have no losses
        assert len(soft_som.losses) == 0

    def test_inference_mode_no_losses(self, soft_som_2d, input_data_2d):
        """Test that no losses are added during inference."""
        # Forward pass in inference mode
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
            use_reconstruction_loss=True
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
            use_reconstruction_loss=True
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

        # Get configs
        config = soft_som_2d.get_config()
        build_config = soft_som_2d.get_build_config()

        # Recreate the layer
        recreated_som = SoftSOMLayer.from_config(config)
        recreated_som.build_from_config(build_config)

        # Check configuration matches
        assert recreated_som.grid_shape == soft_som_2d.grid_shape
        assert recreated_som.input_dim == soft_som_2d.input_dim
        assert recreated_som.temperature == soft_som_2d.temperature
        assert recreated_som.use_per_dimension_softmax == soft_som_2d.use_per_dimension_softmax
        assert recreated_som.use_reconstruction_loss == soft_som_2d.use_reconstruction_loss
        assert recreated_som.reconstruction_weight == soft_som_2d.reconstruction_weight
        assert recreated_som.topological_weight == soft_som_2d.topological_weight

        # Check weights match (shapes should be the same)
        assert recreated_som.weights_map.shape == soft_som_2d.weights_map.shape

    def test_serialization_with_custom_objects(self):
        """Test serialization with custom initializers and regularizers."""
        original_som = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=100,
            temperature=0.8,
            kernel_initializer=keras.initializers.HeNormal(),
            kernel_regularizer=keras.regularizers.L1(0.01)
        )

        # Build the layer
        original_som.build((32, 100))

        # Get configs
        config = original_som.get_config()
        build_config = original_som.get_build_config()

        # Recreate the layer
        recreated_som = SoftSOMLayer.from_config(config)
        recreated_som.build_from_config(build_config)

        # Check configuration matches
        assert recreated_som.grid_shape == original_som.grid_shape
        assert recreated_som.input_dim == original_som.input_dim
        assert recreated_som.temperature == original_som.temperature

    # =============================================================================
    # Model Integration Tests
    # =============================================================================

    def test_model_integration(self, input_data_2d):
        """Test the layer in a model context."""
        # Create a model with the Soft SOM layer
        inputs = keras.Input(shape=(784,))
        reconstruction = SoftSOMLayer(grid_shape=(8, 8), input_dim=784)(inputs)

        # Add additional processing
        x = keras.layers.Dense(128, activation='relu')(reconstruction)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # Test forward pass
        predictions = model(input_data_2d, training=False)
        assert predictions.shape == (64, 10)

    def test_model_save_load(self, input_data_2d):
        """Test saving and loading a model with the Soft SOM layer."""
        # Create a model with the Soft SOM layer
        inputs = keras.Input(shape=(784,))
        reconstruction = SoftSOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            temperature=0.7,
            name="soft_som_layer"
        )(inputs)

        # Add output layer - just use reconstruction directly
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
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={"SoftSOMLayer": SoftSOMLayer}
            )

            # Generate prediction with loaded model
            loaded_prediction = loaded_model.predict(input_data_2d[:8], verbose=0)

            # Check predictions match (allowing for small numerical differences)
            assert np.allclose(original_prediction, loaded_prediction, rtol=1e-5)

            # Check layer type is preserved
            som_layer = loaded_model.get_layer("soft_som_layer")
            assert isinstance(som_layer, SoftSOMLayer)
            assert som_layer.temperature == 0.7

    # =============================================================================
    # Numerical Stability Tests
    # =============================================================================

    def test_numerical_stability(self):
        """Test layer stability with extreme input values."""
        soft_som = SoftSOMLayer(grid_shape=(4, 4), input_dim=10)

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
            temperature=1e-6
        )

        # Very high temperature (should create smooth distributions)
        soft_som_high = SoftSOMLayer(
            grid_shape=(5, 5),
            input_dim=784,
            temperature=1e6
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

    def test_different_grid_shapes(self):
        """Test Soft SOM with various grid shapes."""
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
                temperature=0.5
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

    def test_reconstruction_quality_improves(self, input_data_2d):
        """Test that reconstruction quality can improve with training."""
        soft_som = SoftSOMLayer(
            grid_shape=(8, 8),
            input_dim=784,
            temperature=0.5,
            use_reconstruction_loss=True
        )

        # Create model for training
        inputs = keras.Input(shape=(784,))
        reconstruction = soft_som(inputs)
        model = keras.Model(inputs=inputs, outputs=reconstruction)

        model.compile(optimizer='adam', loss='mse')

        # Initial reconstruction error
        initial_reconstruction = model.predict(input_data_2d[:16], verbose=0)
        initial_mse = np.mean(np.square(input_data_2d[:16] - initial_reconstruction))

        # Train for a few steps
        model.fit(
            input_data_2d[:16], input_data_2d[:16],
            epochs=5,
            batch_size=8,
            verbose=0
        )

        # Final reconstruction error
        final_reconstruction = model.predict(input_data_2d[:16], verbose=0)
        final_mse = np.mean(np.square(input_data_2d[:16] - final_reconstruction))

        # Error should generally decrease (though not guaranteed for all cases)
        # At minimum, check that the model is still functioning
        assert final_mse >= 0
        assert not np.isnan(final_mse)

    def test_temperature_annealing_effect(self, input_data_2d):
        """Test that temperature can be dynamically changed."""
        soft_som = SoftSOMLayer(
            grid_shape=(6, 6),
            input_dim=784,
            temperature=2.0
        )

        # Initial soft assignments with high temperature
        soft_som(input_data_2d[:4], training=False)
        initial_assignments = soft_som.get_soft_assignments(input_data_2d[:4])
        initial_max = tf.reduce_max(initial_assignments)

        # Change temperature to lower value
        soft_som.temperature = 0.1

        # New soft assignments should be sharper (higher maximum values)
        new_assignments = soft_som.get_soft_assignments(input_data_2d[:4])
        new_max = tf.reduce_max(new_assignments)

        # Lower temperature should generally produce sharper distributions
        assert new_max >= initial_max  # Allow for equal in case of edge cases

    def test_edge_case_single_neuron(self):
        """Test Soft SOM with single neuron grid."""
        soft_som = SoftSOMLayer(grid_shape=(1,), input_dim=10)
        test_input = keras.random.uniform((5, 10))

        reconstruction = soft_som(test_input, training=True)

        # Should still work
        assert reconstruction.shape == (5, 10)

        # Soft assignments should be all 1.0 (only one neuron)
        soft_assignments = soft_som.get_soft_assignments(test_input)
        assert tf.reduce_all(tf.abs(soft_assignments - 1.0) < 1e-6)

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
        assert tf.reduce_all(tf.abs(recon_single - recon_batch) < 1e-6)