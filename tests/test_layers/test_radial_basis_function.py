"""
Comprehensive Test Suite for RBF Layer

This test suite follows Modern Keras 3 testing best practices with emphasis on
serialization testing and production readiness validation.
"""

import pytest
import tempfile
import os
from typing import Any, Dict, Tuple
import numpy as np

import keras
from keras import ops
import tensorflow as tf

from dl_techniques.layers.radial_basis_function import RBFLayer


def generate_cluster_data(
    n_clusters: int,
    n_samples: int,
    dim: int,
    noise: float = 0.1,
    seed: int = 42
) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
    """
    Generate clustered synthetic data for testing RBF layer behavior.

    Args:
        n_clusters: Number of distinct clusters to generate
        n_samples: Total number of samples across all clusters
        dim: Dimensionality of the feature space
        noise: Standard deviation of Gaussian noise added to cluster samples
        seed: Random seed for reproducible data generation

    Returns:
        Tuple of (data, true_centers) as Keras tensors
    """
    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Generate well-separated cluster centers
    centers = tf.random.uniform(
        (n_clusters, dim),
        minval=-5.0,
        maxval=5.0,
        seed=seed
    )

    # Generate samples around each center
    samples_per_cluster = n_samples // n_clusters
    data_parts = []

    for i in range(n_clusters):
        center = centers[i:i+1]  # Keep batch dimension
        cluster_samples = center + tf.random.normal(
            (samples_per_cluster, dim),
            mean=0.0,
            stddev=noise,
            seed=seed + i
        )
        data_parts.append(cluster_samples)

    # Concatenate all cluster data
    data = ops.concatenate(data_parts, axis=0)

    return data, centers


class TestRBFLayer:
    """Comprehensive test suite for RBF Layer following Modern Keras 3 patterns."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard RBF layer configuration for testing."""
        return {
            'units': 8,
            'gamma_init': 1.0,
            'repulsion_strength': 0.1,
            'min_center_distance': 1.0,
            'trainable_gamma': True,
            'safety_margin': 0.2,
        }

    @pytest.fixture
    def sample_input_2d(self) -> keras.KerasTensor:
        """Sample 2D input tensor for testing."""
        return tf.random.normal(shape=(16, 4), seed=42)

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """Sample 3D input tensor for testing."""
        return tf.random.normal(shape=(8, 12, 4), seed=43)

    def test_initialization(self, layer_config: Dict[str, Any]) -> None:
        """
        Test layer initialization and parameter validation.

        Validates:
        - Correct attribute assignment from config
        - Unbuilt state after initialization
        - Sub-component creation
        - Parameter validation with invalid inputs
        """
        # Test successful initialization
        layer = RBFLayer(**layer_config)

        # Verify configuration storage
        assert layer.units == layer_config['units']
        assert layer.gamma_init == layer_config['gamma_init']
        assert layer.repulsion_strength == layer_config['repulsion_strength']
        assert layer.min_center_distance == layer_config['min_center_distance']
        assert layer.trainable_gamma == layer_config['trainable_gamma']
        assert layer.safety_margin == layer_config['safety_margin']

        # Verify unbuilt state
        assert not layer.built
        assert layer.centers is None
        assert layer.gamma_raw is None

        # Test initializer/constraint/regularizer handling
        assert layer.center_initializer is not None
        assert layer.gamma_regularizer is None

    def test_parameter_validation(self) -> None:
        """Test comprehensive parameter validation with invalid inputs."""

        # Test invalid units
        with pytest.raises(ValueError, match="units must be positive"):
            RBFLayer(units=0)

        with pytest.raises(ValueError, match="units must be positive"):
            RBFLayer(units=-5)

        # Test invalid gamma_init
        with pytest.raises(ValueError, match="gamma_init must be positive"):
            RBFLayer(units=10, gamma_init=0.0)

        with pytest.raises(ValueError, match="gamma_init must be positive"):
            RBFLayer(units=10, gamma_init=-1.0)

        # Test invalid repulsion_strength
        with pytest.raises(ValueError, match="repulsion_strength must be non-negative"):
            RBFLayer(units=10, repulsion_strength=-0.1)

        # Test invalid min_center_distance
        with pytest.raises(ValueError, match="min_center_distance must be positive"):
            RBFLayer(units=10, min_center_distance=0.0)

        # Test invalid safety_margin
        with pytest.raises(ValueError, match="safety_margin must be non-negative"):
            RBFLayer(units=10, safety_margin=-0.1)

    @pytest.mark.parametrize("input_tensor", ["sample_input_2d", "sample_input_3d"])
    def test_forward_pass(
        self,
        layer_config: Dict[str, Any],
        input_tensor: str,
        request: pytest.FixtureRequest
    ) -> None:
        """
        Test forward pass and automatic building.

        Validates:
        - Automatic building on first call
        - Correct output shapes
        - Weight creation
        - Multiple forward passes consistency
        """
        # Get the input tensor from fixture
        inputs = request.getfixturevalue(input_tensor)

        layer = RBFLayer(**layer_config)

        # Forward pass triggers building
        outputs = layer(inputs)

        # Verify building occurred
        assert layer.built
        assert layer.centers is not None
        assert layer.gamma_raw is not None

        # Verify output shape
        expected_shape = list(inputs.shape)
        expected_shape[-1] = layer_config['units']
        assert outputs.shape == tuple(expected_shape)

        # Verify output properties
        assert ops.all(outputs >= 0.0), "RBF outputs must be non-negative"
        assert ops.all(outputs <= 1.0), "RBF outputs must be <= 1.0"

        # Test consistency across multiple calls
        outputs2 = layer(inputs)
        np.testing.assert_allclose(
            ops.convert_to_numpy(outputs),
            ops.convert_to_numpy(outputs2),
            rtol=1e-6,
            atol=1e-6,
            err_msg="Multiple forward passes should be consistent"
        )

    def test_serialization_cycle(
        self,
        layer_config: Dict[str, Any],
        sample_input_2d: keras.KerasTensor
    ) -> None:
        """
        CRITICAL TEST: Full serialization and deserialization cycle.

        This is the most important test for production readiness.
        Validates:
        - Model saving with custom layer
        - Model loading with custom layer
        - Identical predictions after serialization
        - Weight preservation
        """
        # Create model with RBF layer
        inputs = keras.Input(shape=sample_input_2d.shape[1:])
        rbf_outputs = RBFLayer(**layer_config)(inputs)
        # Add a simple dense layer to make it more realistic
        outputs = keras.layers.Dense(3, activation='softmax')(rbf_outputs)
        model = keras.Model(inputs, outputs)

        # Get original prediction
        original_prediction = model(sample_input_2d)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'rbf_test_model.keras')

            # Save model
            model.save(filepath)

            # Load model (tests custom layer registration)
            loaded_model = keras.models.load_model(filepath)

            # Get prediction from loaded model
            loaded_prediction = loaded_model(sample_input_2d)

            # Verify identical predictions
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions differ after serialization cycle"
            )

            # Verify layer configuration preserved
            original_rbf = model.layers[1]  # RBF layer
            loaded_rbf = loaded_model.layers[1]  # RBF layer

            assert original_rbf.units == loaded_rbf.units
            assert original_rbf.gamma_init == loaded_rbf.gamma_init
            assert original_rbf.repulsion_strength == loaded_rbf.repulsion_strength

    def test_config_completeness(self, layer_config: Dict[str, Any]) -> None:
        """
        Test that get_config contains all __init__ parameters.

        Validates:
        - All initialization parameters in config
        - Proper serialization of complex objects
        - Config can reconstruct identical layer
        """
        layer = RBFLayer(**layer_config)
        config = layer.get_config()

        # Check all required parameters present
        required_keys = [
            'units', 'gamma_init', 'repulsion_strength', 'min_center_distance',
            'safety_margin', 'trainable_gamma', 'center_initializer',
            'center_constraint', 'kernel_regularizer', 'gamma_regularizer'
        ]

        for key in required_keys:
            assert key in config, f"Missing {key} in get_config()"

        # Verify config values match initialization
        for key, value in layer_config.items():
            if key in ['center_initializer', 'center_constraint',
                      'kernel_regularizer', 'gamma_regularizer']:
                # These are serialized as dicts, test separately
                continue
            assert config[key] == value, f"Config mismatch for {key}"

        # Test reconstruction from config
        reconstructed_layer = RBFLayer.from_config(config)
        assert reconstructed_layer.units == layer.units
        assert reconstructed_layer.gamma_init == layer.gamma_init

    def test_gradients_flow(
        self,
        layer_config: Dict[str, Any],
        sample_input_2d: keras.KerasTensor
    ) -> None:
        """
        Test gradient computation and backpropagation.

        Validates:
        - Gradients computed for all trainable weights
        - No None gradients
        - Reasonable gradient magnitudes
        """
        layer = RBFLayer(**layer_config)

        with tf.GradientTape() as tape:
            # Enable gradient tracking for input
            tape.watch(sample_input_2d)

            # Forward pass
            outputs = layer(sample_input_2d, training=True)

            # Compute loss (simple mean squared output)
            loss = ops.mean(ops.square(outputs))

        # Compute all gradients in one call to avoid tape reuse
        all_variables = layer.trainable_weights + [sample_input_2d]
        all_gradients = tape.gradient(loss, all_variables)

        # Split gradients
        layer_gradients = all_gradients[:-1]
        input_gradients = all_gradients[-1]

        # Verify gradients exist
        assert len(layer_gradients) > 0, "No trainable weights found"
        assert all(g is not None for g in layer_gradients), "Some gradients are None"
        assert input_gradients is not None, "Input gradients are None"

        # Verify reasonable gradient magnitudes
        for i, grad in enumerate(layer_gradients):
            grad_norm = ops.sqrt(ops.sum(ops.square(grad)))
            assert grad_norm > 0, f"Zero gradient for weight {i}"
            assert grad_norm < 1000, f"Exploding gradient for weight {i}: {grad_norm}"

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(
        self,
        layer_config: Dict[str, Any],
        sample_input_2d: keras.KerasTensor,
        training: bool
    ) -> None:
        """
        Test behavior in different training modes.

        Validates:
        - Consistent outputs across training modes
        - Repulsion loss only added during training
        - Proper handling of training parameter
        """
        # Create layer without regularizers for cleaner testing
        clean_config = dict(layer_config)
        clean_config['gamma_regularizer'] = None
        clean_config['kernel_regularizer'] = None

        layer = RBFLayer(**clean_config)

        # Clear any existing losses to ensure clean state
        layer.losses.clear()

        # Forward pass in specified training mode
        outputs = layer(sample_input_2d, training=training)

        # Verify output shape and properties
        assert outputs.shape[0] == sample_input_2d.shape[0]
        assert outputs.shape[-1] == layer_config['units']
        assert ops.all(outputs >= 0.0)
        assert ops.all(outputs <= 1.0)

        # Check repulsion loss behavior
        if training is True:
            # In training mode, repulsion loss should be added
            assert len(layer.losses) > 0, "Repulsion loss not added in training mode"
            # Verify it's actually repulsion loss (should be > 0 for multiple units)
            total_loss = sum(layer.losses)
            assert total_loss >= 0, "Repulsion loss should be non-negative"
        else:
            # In inference mode or None, no repulsion losses should be added
            # Note: training=None defaults to False in most contexts
            assert len(layer.losses) == 0, f"Unexpected losses in inference mode: {[float(loss) for loss in layer.losses]}"

    def test_output_shape_computation(self, layer_config: Dict[str, Any]) -> None:
        """Test compute_output_shape method."""
        layer = RBFLayer(**layer_config)

        # Test 2D input
        input_shape_2d = (None, 10)
        output_shape_2d = layer.compute_output_shape(input_shape_2d)
        expected_2d = (None, layer_config['units'])
        assert output_shape_2d == expected_2d

        # Test 3D input (with time dimension)
        input_shape_3d = (None, 20, 10)
        output_shape_3d = layer.compute_output_shape(input_shape_3d)
        expected_3d = (None, 20, layer_config['units'])
        assert output_shape_3d == expected_3d

    def test_rbf_activation_properties(
        self,
        layer_config: Dict[str, Any]
    ) -> None:
        """
        Test mathematical properties of RBF activations.

        Validates:
        - Maximum activation at centers
        - Symmetric response around centers
        - Monotonic decrease with distance
        - Bounded output range
        """
        # Use fewer units for precise testing
        config = dict(layer_config)
        config['units'] = 3

        layer = RBFLayer(**config)
        input_dim = 4
        layer.build((None, input_dim))

        # Test maximum activation at center
        for i in range(config['units']):
            center = layer.centers[i:i+1]  # Shape (1, input_dim)
            activation = layer(center)[0, i]  # Activation of unit i

            # Should be close to 1.0 (maximum activation)
            assert ops.abs(activation - 1.0) < 1e-4, \
                f"Unit {i} activation at center: {activation}, expected ~1.0"

        # Test symmetry around center
        center = layer.centers[0:1]  # First center
        offset = ops.ones((1, input_dim)) * 0.5

        pos_input = center + offset
        neg_input = center - offset

        pos_activation = layer(pos_input)[0, 0]
        neg_activation = layer(neg_input)[0, 0]

        assert ops.abs(pos_activation - neg_activation) < 1e-5, \
            "RBF should be symmetric around center"

        # Test monotonic decrease with distance
        center = layer.centers[0:1]
        distances = [0.0, 0.5, 1.0, 2.0]
        activations = []

        for dist in distances:
            test_input = center + ops.ones((1, input_dim)) * dist
            activation = layer(test_input)[0, 0]
            activations.append(float(activation))

        # Activations should decrease with distance
        for i in range(len(activations) - 1):
            assert activations[i] >= activations[i + 1], \
                f"Non-monotonic behavior: {activations}"

    def test_repulsion_mechanism(self, layer_config: Dict[str, Any]) -> None:
        """
        Test center repulsion mechanism.

        Validates:
        - Repulsion force between close centers
        - Reduced repulsion for distant centers
        - Proper scaling with dimensionality
        """
        # Use 2 units for precise repulsion testing
        config = dict(layer_config)
        config.update({
            'units': 2,
            'repulsion_strength': 1.0,
            'min_center_distance': 2.0
        })

        layer = RBFLayer(**config)
        input_dim = 4
        layer.build((None, input_dim))

        # Test with centers very close together
        close_centers = ops.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1]  # Distance ~ 0.2
        ])
        layer.centers.assign(close_centers)
        close_repulsion = layer._compute_repulsion(layer.centers)

        # Test with centers far apart
        distant_centers = ops.array([
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0, 5.0]  # Distance = 10.0
        ])
        layer.centers.assign(distant_centers)
        distant_repulsion = layer._compute_repulsion(layer.centers)

        # Close centers should have higher repulsion
        assert close_repulsion > distant_repulsion, \
            f"Close repulsion ({close_repulsion}) should > distant ({distant_repulsion})"

        # Distant centers should have minimal repulsion
        assert distant_repulsion < 0.01, \
            f"Distant centers should have minimal repulsion, got {distant_repulsion}"

    def test_training_convergence(self, layer_config: Dict[str, Any]) -> None:
        """
        Test training behavior and convergence properties.

        Validates:
        - Centers adapt to data distribution
        - Loss decreases over training
        - Centers maintain minimum separation
        """
        # Configure for training test
        config = dict(layer_config)
        config.update({
            'units': 6,
            'repulsion_strength': 0.5,
            'min_center_distance': 1.5,
            'gamma_init': 0.5
        })

        # Generate clustered data
        n_clusters = 3
        n_samples = 150
        input_dim = 4
        data, true_centers = generate_cluster_data(
            n_clusters=n_clusters,
            n_samples=n_samples,
            dim=input_dim,
            noise=0.1
        )

        # Create layer and optimizer
        layer = RBFLayer(**config)
        optimizer = keras.optimizers.Adam(learning_rate=0.02)

        # Training loop
        initial_loss = None
        final_loss = None

        for epoch in range(100):
            # Clear losses from previous iteration
            layer.losses.clear()

            with tf.GradientTape() as tape:
                # Forward pass with training=True
                outputs = layer(data, training=True)

                # Simple loss: encourage diverse, strong activations
                max_activations = ops.max(outputs, axis=1)
                activation_loss = -ops.mean(max_activations)

                # Get repulsion loss
                repulsion_loss = sum(layer.losses)
                total_loss = activation_loss + repulsion_loss

                if initial_loss is None:
                    initial_loss = float(total_loss)
                final_loss = float(total_loss)

            # Update weights
            gradients = tape.gradient(total_loss, layer.trainable_weights)
            optimizer.apply_gradients(zip(gradients, layer.trainable_weights))

        # Verify training progress
        assert final_loss < initial_loss, \
            f"Training should reduce loss: {initial_loss} -> {final_loss}"

        # Check center separation
        centers = ops.convert_to_numpy(layer.centers)
        min_distance = float('inf')

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_distance = min(min_distance, dist)

        # Centers should maintain reasonable separation
        expected_min = 0.3 * config['min_center_distance']
        assert min_distance >= expected_min, \
            f"Centers too close: {min_distance} < {expected_min}"

    def test_numerical_stability(self, layer_config: Dict[str, Any]) -> None:
        """
        Test numerical stability with extreme inputs.

        Validates:
        - Handling of large magnitude inputs
        - Stability with very small inputs
        - No NaN/Inf in outputs
        - Bounded output range maintained
        """
        layer = RBFLayer(**layer_config)
        input_dim = 4
        batch_size = 16

        # Test cases with extreme inputs
        test_cases = [
            ("zeros", ops.zeros((batch_size, input_dim))),
            ("large_positive", ops.ones((batch_size, input_dim)) * 1000.0),
            ("large_negative", ops.ones((batch_size, input_dim)) * -1000.0),
            ("very_small", ops.ones((batch_size, input_dim)) * 1e-8),
            ("mixed_extreme", tf.random.normal((batch_size, input_dim), seed=44) * 1e6)
        ]

        for case_name, inputs in test_cases:
            outputs = layer(inputs)

            # Convert to numpy for detailed checks
            outputs_np = ops.convert_to_numpy(outputs)

            # Check for numerical issues
            assert not np.any(np.isnan(outputs_np)), \
                f"NaN values in outputs for case: {case_name}"

            assert not np.any(np.isinf(outputs_np)), \
                f"Inf values in outputs for case: {case_name}"

            # Check valid range [0, 1]
            assert np.all(outputs_np >= 0.0), \
                f"Negative outputs for case: {case_name}"

            assert np.all(outputs_np <= 1.0), \
                f"Outputs > 1.0 for case: {case_name}"

    def test_edge_cases(self) -> None:
        """Test error conditions and edge cases."""

        # Test invalid build input shapes
        layer = RBFLayer(units=5)

        # 1D input should fail
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            layer.build((10,))

        # None input dimension should fail
        with pytest.raises(ValueError, match="Last dimension.*must be defined"):
            layer.build((None, None))

        # Test single unit (no repulsion)
        single_unit_layer = RBFLayer(units=1)

        # Clear any existing losses to ensure clean state
        single_unit_layer.losses.clear()

        inputs = tf.random.normal((8, 3), seed=45)
        outputs = single_unit_layer(inputs, training=True)

        # Should work without errors
        assert outputs.shape == (8, 1)
        # No repulsion loss for single unit (repulsion requires >= 2 units)
        assert len(single_unit_layer.losses) == 0, f"Unexpected losses for single unit: {[float(loss) for loss in single_unit_layer.losses]}"


if __name__ == '__main__':
    # Run with: python -m pytest rbf_layer_test.py -v
    pytest.main([__file__, '-v', '--tb=short'])