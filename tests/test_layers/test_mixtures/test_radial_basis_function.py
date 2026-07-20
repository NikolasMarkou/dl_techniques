import pytest
import tempfile
import os
from typing import Any, Dict, Tuple
import numpy as np

import keras
from keras import ops
import tensorflow as tf


from dl_techniques.layers.mixtures.radial_basis_function import RBFLayer


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
        n_clusters: Number of distinct clusters to generate.
        n_samples: Total number of samples across all clusters.
        dim: Dimensionality of the feature space.
        noise: Standard deviation of Gaussian noise added to cluster samples.
        seed: Random seed for reproducible data generation.

    Returns:
        Tuple of (data, true_centers) as Keras tensors.
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
        center = centers[i:i + 1]  # Keep batch dimension
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
        """
        Standard RBF layer configuration for testing.

        Returns:
            A dictionary of configuration parameters.
        """
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
        """
        Sample 2D input tensor for testing.

        Returns:
            A tensor of shape (16, 4).
        """
        return tf.random.normal(shape=(16, 4), seed=42)

    @pytest.fixture
    def sample_input_3d(self) -> keras.KerasTensor:
        """
        Sample 3D input tensor for testing.

        Returns:
            A tensor of shape (8, 12, 4).
        """
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
            'center_constraint', 'kernel_regularizer', 'gamma_regularizer',
            'output_mode',
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
            # Bounded exponent prevents exploding gradients
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

            # Verify it's actually repulsion loss (should be > 0 for multiple units initialized randomly)
            total_loss = sum(layer.losses)
            assert float(total_loss) >= 0, "Repulsion loss should be non-negative"
        else:
            # In inference mode or None (defaults to inference usually), no repulsion losses
            assert len(
                layer.losses) == 0, f"Unexpected losses in inference mode: {[float(loss) for loss in layer.losses]}"

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
        """
        # Use fewer units for precise testing
        config = dict(layer_config)
        config['units'] = 3

        layer = RBFLayer(**config)
        input_dim = 4
        layer.build((None, input_dim))

        # Test maximum activation at center
        for i in range(config['units']):
            center = layer.centers[i:i + 1]  # Shape (1, input_dim)
            activation = layer(center)[0, i]  # Activation of unit i

            # Should be close to 1.0 (maximum activation: exp(0) = 1)
            assert float(ops.abs(activation - 1.0)) < 1e-4, \
                f"Unit {i} activation at center: {activation}, expected ~1.0"

        # Test symmetry around center
        center = layer.centers[0:1]  # First center
        offset = ops.ones((1, input_dim)) * 0.5

        pos_input = center + offset
        neg_input = center - offset

        pos_activation = layer(pos_input)[0, 0]
        neg_activation = layer(neg_input)[0, 0]

        assert float(ops.abs(pos_activation - neg_activation)) < 1e-5, \
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

        # Compute repulsion loss (now internal, uses self.centers)
        close_repulsion = layer._compute_repulsion_loss()

        # Test with centers far apart
        distant_centers = ops.array([
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 5.0, 5.0, 5.0]  # Distance = 10.0
        ])
        layer.centers.assign(distant_centers)

        distant_repulsion = layer._compute_repulsion_loss()

        # Close centers should have higher repulsion
        # Convert to float for safe assertion
        assert float(close_repulsion) > float(distant_repulsion), \
            f"Close repulsion ({close_repulsion}) should > distant ({distant_repulsion})"

        # Distant centers should have minimal repulsion
        # With dist=10.0 and min_dist=2.0, max(0, threshold - dist) should be 0
        assert float(distant_repulsion) < 1e-6, \
            f"Distant centers should have minimal repulsion, got {distant_repulsion}"

    def test_training_convergence(self, layer_config: Dict[str, Any]) -> None:
        """
        Test training behavior and convergence properties.

        Validates:
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

        for epoch in range(50):
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
        expected_min = 0.1 * config['min_center_distance']
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
        # Same extreme-input battery, in the opt-in normalized mode. The 'basis'
        # assertions below (range [0, 1], no NaN/Inf) hold for NRBF too, and the
        # normalized arm additionally must sum to exactly 1.0 along the last axis.
        norm_layer = RBFLayer(**layer_config, output_mode='normalized')
        input_dim = 4
        batch_size = 16

        # Test cases with extreme inputs
        test_cases = [
            ("zeros", ops.zeros((batch_size, input_dim))),
            ("large_positive", ops.ones((batch_size, input_dim)) * 1000.0),
            ("large_negative", ops.ones((batch_size, input_dim)) * -1000.0),
            ("very_small", ops.ones((batch_size, input_dim)) * 1e-8),
            # Random extreme values
            ("mixed_extreme", tf.random.normal((batch_size, input_dim), seed=44) * 1e5)
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

            norm_np = ops.convert_to_numpy(norm_layer(inputs))

            assert not np.any(np.isnan(norm_np)), \
                f"NaN in normalized outputs for case: {case_name}"
            assert not np.any(np.isinf(norm_np)), \
                f"Inf in normalized outputs for case: {case_name}"
            assert np.all(norm_np >= 0.0), \
                f"Negative normalized outputs for case: {case_name}"
            np.testing.assert_allclose(
                norm_np.sum(axis=-1),
                np.ones(batch_size, dtype='float32'),
                rtol=1e-6, atol=1e-6,
                err_msg=f"normalized outputs do not sum to 1.0 for case: {case_name}",
            )

    def test_edge_cases(self) -> None:
        """Test error conditions and edge cases."""

        # Test invalid build input shapes
        layer = RBFLayer(units=5)

        # 1D input should fail
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            layer.build((10,))

        # None input dimension should fail
        # Regex matched against: "The last dimension of the input must be defined."
        with pytest.raises(ValueError, match="last dimension.*must be defined"):
            layer.build((None, None))

        # Test single unit (no repulsion)
        single_unit_layer = RBFLayer(units=1, repulsion_strength=1.0)

        # Clear any existing losses to ensure clean state
        single_unit_layer.losses.clear()

        inputs = tf.random.normal((8, 3), seed=45)
        outputs = single_unit_layer(inputs, training=True)

        # Should work without errors
        assert outputs.shape == (8, 1)

        # No repulsion loss for single unit (repulsion requires >= 2 units)
        assert len(single_unit_layer.losses) == 0, \
            f"Unexpected losses for single unit: {[float(loss) for loss in single_unit_layer.losses]}"


class TestRBFLayerGraphMode:
    """Graph-compatibility regression: call() must trace with a symbolic flag."""

    def test_graph_mode_symbolic_training(self) -> None:
        """Regression for the bare ``if training and ...:`` graph-breaker."""
        layer = RBFLayer(units=8, repulsion_strength=0.1)
        x = tf.constant(np.random.normal(0, 1, (8, 16)).astype(np.float32))

        @tf.function
        def run(inp, training):
            return layer(inp, training=training)

        y_train = run(x, tf.constant(True))
        y_infer = run(x, tf.constant(False))
        assert tuple(y_train.shape) == (8, 8)
        assert tuple(y_infer.shape) == (8, 8)

    def test_symbolic_training_fires_repulsion_loss(self) -> None:
        """A SYMBOLIC training=True tensor must fire the repulsion add_loss (== the
        python-True value); symbolic False must contribute zero (the foot-gun fix).
        """
        layer = RBFLayer(units=8, repulsion_strength=0.5)
        x = tf.constant(np.random.normal(0, 1, (16, 16)).astype(np.float32))
        layer.build((16, 16))

        _ = layer(x, training=True)
        python_loss = float(ops.convert_to_numpy(tf.add_n(layer.losses)))

        @tf.function
        def step(inp, training):
            _ = layer(inp, training=training)
            return tf.add_n(layer.losses) if layer.losses else tf.constant(0.0)

        sym_true = float(step(x, tf.constant(True)))
        sym_false = float(step(x, tf.constant(False)))
        assert python_loss > 0.0
        assert np.isclose(sym_true, python_loss, atol=1e-6), \
            "symbolic training=True repulsion loss must equal the python-True value"
        assert sym_false == 0.0, "symbolic training=False must contribute zero loss"

    def test_mixed_float16_forward(self) -> None:
        """Forward must run under a mixed_float16 policy with float32 kernel math and a
        compute_dtype output (uniform with GMMLayer / KMeansLayer)."""
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            layer = RBFLayer(units=8, repulsion_strength=0.5)
            x = np.random.normal(0, 1, (8, 16)).astype(np.float32)
            y = layer(x)
            assert keras.backend.standardize_dtype(y.dtype) == "float16"
            y_np = np.asarray(ops.convert_to_numpy(y), dtype=np.float32)
            assert not np.isnan(y_np).any() and not np.isinf(y_np).any()
            # Parameters stay float32 (autocast=False).
            assert keras.backend.standardize_dtype(layer.centers.dtype) == "float32"
            assert keras.backend.standardize_dtype(layer.gamma_raw.dtype) == "float32"

            # --- NRBF regression guard for the reproduced fp16 NaN (D-002/F8) -----
            # This is the direct guard on the failure mode: in the 'basis' arm `phi`
            # bottoms out at the 50.0 exponent clip, exp(-50) ~ 1.93e-22, which is a
            # normal float32 but flushes to EXACT 0.0 in float16 -- and for ordinary
            # inputs it is far smaller still. A `phi / sum(phi)` normalization applied
            # to the float16 tensor is therefore 0/0 -> NaN for any input far from
            # every center. `x_far` below is exactly such an input (centers are
            # initialized in [-0.05, 0.05]; 1e4 is far outside every one of them), so
            # a naive/post-cast implementation makes this assertion FAIL, not pass
            # vacuously. Verified by running it against such an implementation.
            norm_layer = RBFLayer(units=8, repulsion_strength=0.5,
                                  output_mode='normalized')
            x_far = np.full((4, 16), 1e4, dtype=np.float32)
            for name, xs in (("near", x), ("far", x_far)):
                yn = norm_layer(xs)
                assert keras.backend.standardize_dtype(yn.dtype) == "float16"
                yn_np = np.asarray(ops.convert_to_numpy(yn), dtype=np.float32)
                assert not np.isnan(yn_np).any(), \
                    f"NaN in mixed_float16 normalized output ({name} input)"
                assert not np.isinf(yn_np).any(), \
                    f"Inf in mixed_float16 normalized output ({name} input)"
                # float16 has ~3 decimal digits, hence the loose tolerance here.
                np.testing.assert_allclose(
                    yn_np.sum(axis=-1), np.ones(xs.shape[0], dtype=np.float32),
                    rtol=1e-2, atol=1e-2,
                    err_msg=f"mixed_float16 normalized output ({name}) must sum to 1",
                )
            # D-008: far from every center, TRUE NRBF selects the NEAREST center --
            # that is precisely NRBF's defining property over plain RBF. It must NOT
            # be uniform 1/units: uniform is the signature of softmaxing the CLIPPED
            # exponent, which is a dead layer with exactly zero gradient. The earlier
            # version of this test asserted uniformity and so pinned the defect.
            far_np = np.asarray(
                ops.convert_to_numpy(norm_layer(x_far)), dtype=np.float32
            )
            assert far_np.max(axis=-1).min() > 0.9, (
                "far-from-all-centers NRBF collapsed toward uniform "
                f"(max prob {far_np.max(axis=-1).min()}); the normalized arm is being "
                "fed the CLIPPED exponent"
            )
            assert not np.allclose(
                far_np, np.full((4, 8), 1.0 / 8.0, dtype=np.float32), atol=1e-2
            ), "NRBF is uniform far from every center -- the 50.0 clip leaked in"
        finally:
            keras.mixed_precision.set_global_policy(original_policy)


class TestRBFLayerOutputMode:
    """The opt-in ``output_mode='normalized'`` (NRBF) surface.

    Covers the two things that can silently break: the DEFAULT ``'basis'`` arm
    drifting away from its pre-change behavior, and the normalized arm being
    implemented as a naive ``phi / sum(phi)`` division (a real NaN under
    ``mixed_float16``; see ``RBFLayer.call``'s D-002 anchor).
    """

    # Pinned pre-change capture (see plan step 2a). Weights are assigned
    # explicitly below, so this is a fixed numeric target rather than a re-run
    # of the implementation against itself: ANY change to the default arm's
    # arithmetic -- including an algebraically-equivalent rewrite of
    # exp(-min(dist_sq * gamma, 50.0)) -- moves these bits and fails the test.
    GOLDEN_CENTERS = np.array(
        [[0.0, 0.0], [1.0, 2.0], [-3.0, 0.5]], dtype='float32'
    )
    GOLDEN_GAMMA_RAW = np.array([0.5413248, 1.0, -0.5], dtype='float32')
    GOLDEN_X = np.array(
        [[0.5, -0.5], [2.0, 2.0], [1000.0, 1000.0]], dtype='float32'
    )
    GOLDEN_BASIS = np.array([
        [0.6065306663513184, 0.00019623443949967623, 0.0018705554539337754],
        [0.000335462624207139, 0.2689414620399475, 2.452020453347359e-06],
        [1.9287498933537385e-22, 1.9287498933537385e-22, 1.9287498933537385e-22],
    ], dtype='float32')

    def _golden_layer(self, **kwargs: Any) -> RBFLayer:
        """Build a fully deterministic RBFLayer with the pinned golden weights."""
        layer = RBFLayer(units=3, repulsion_strength=0.0, **kwargs)
        layer.build(self.GOLDEN_X.shape)
        layer.centers.assign(self.GOLDEN_CENTERS)
        layer.gamma_raw.assign(self.GOLDEN_GAMMA_RAW)
        return layer

    def test_golden_capture_is_deterministic(self) -> None:
        """Precondition: the golden comparison is only meaningful if the capture
        is reproducible. Two independent layers with the same assigned weights
        must agree BIT-exactly, otherwise the next test's assert_array_equal
        would be flaky rather than falsifying."""
        a = ops.convert_to_numpy(self._golden_layer()(self.GOLDEN_X))
        b = ops.convert_to_numpy(self._golden_layer()(self.GOLDEN_X))
        np.testing.assert_array_equal(
            a, b, err_msg="RBF forward pass is not deterministic"
        )

    def test_default_basis_matches_pre_change_capture(self) -> None:
        """SC10: the default arm still produces the PRE-change numbers.

        Tolerance note: this is ``assert_allclose(rtol=1e-6)``, not
        ``assert_array_equal``, purely because ``GOLDEN_BASIS`` was captured on
        one device and ``exp`` differs by up to ~1 ULP (measured: 1.1e-7 on two
        elements) between the CPU and CUDA kernels. That is a property of the
        capture medium, not a weakening of the claim: every behavior change this
        test exists to catch moves the values by ORDERS of magnitude (switching
        the default to softmax turns 1.93e-22 into 0.333), so a 1e-6 relative
        band cannot hide one. The strictly bit-exact half of the claim is
        asserted device-portably in the next test.
        """
        layer = self._golden_layer()
        assert layer.output_mode == 'basis', "default output_mode must be 'basis'"

        got = np.asarray(ops.convert_to_numpy(layer(self.GOLDEN_X)))
        np.testing.assert_allclose(
            got, self.GOLDEN_BASIS, rtol=1e-6, atol=0.0,
            err_msg=(
                "default 'basis' output drifted from the pre-change capture -- "
                "backward compatibility is broken"
            ),
        )

    def test_default_basis_is_bit_exact_with_reference_formula(self) -> None:
        """I4/SC10: the default arm is BIT-identical to ``exp(-min(d2*gamma, 50))``.

        The reference below is the documented formula, rebuilt from the layer's
        own weights with the same keras ops on the same device -- so
        ``assert_array_equal`` is legitimate here and portable, unlike a
        cross-device numeric capture. This is the assertion that fires on the
        refactors most likely to break bit-exactness while adding a branch
        nearby: hoisting the compute_dtype cast above the branch, factoring the
        ``exp`` into a shared helper, or algebraically rewriting the exponent.
        """
        layer = self._golden_layer()
        x = ops.convert_to_tensor(self.GOLDEN_X)

        dist_sq = ops.sum(
            ops.square(ops.expand_dims(x, axis=-2) - layer.centers), axis=-1
        )
        reference = ops.exp(-ops.minimum(dist_sq * layer.gamma, 50.0))

        np.testing.assert_array_equal(
            np.asarray(ops.convert_to_numpy(layer(self.GOLDEN_X))),
            np.asarray(ops.convert_to_numpy(reference)),
            err_msg=(
                "default 'basis' output is no longer bit-identical to "
                "exp(-min(dist_sq * gamma, 50.0))"
            ),
        )

    @staticmethod
    def _nrbf_reference(
        x: np.ndarray, centers: np.ndarray, gamma_raw: np.ndarray
    ) -> np.ndarray:
        """Textbook NRBF ``phi_k / sum_j phi_j``, in numerically-safe float64.

        Shares no code with the layer: it re-derives ``gamma = softplus(gamma_raw)``
        in numpy and computes the UNCLIPPED exponent, then shifts by the per-row
        minimum before exponentiating so that arbitrarily distant inputs never
        underflow the whole row to 0/0. The shift is exact -- it cancels in the
        ratio -- so this is the definition, not an approximation of it.
        """
        x64 = x.astype('float64')
        c64 = centers.astype('float64')
        gamma = np.log1p(np.exp(gamma_raw.astype('float64')))  # softplus

        dist_sq = ((x64[..., None, :] - c64) ** 2).sum(axis=-1)
        exponent = dist_sq * gamma                              # UNCLIPPED
        shifted = -(exponent - exponent.min(axis=-1, keepdims=True))
        phi = np.exp(shifted)
        return phi / phi.sum(axis=-1, keepdims=True)

    def test_normalized_matches_numpy_division_reference(self) -> None:
        """SC11: NRBF equals ``phi_k / sum_j phi_j``, computed independently.

        The reference is the textbook definition on the UNCLIPPED exponent, in
        float64, sharing no code with the layer. It catches a softmax over the
        wrong axis, a sign error on the exponent, a mode branch that silently
        falls through to 'basis', and -- via the third row of ``GOLDEN_X``, which
        sits 1000 units from every center -- the D-008 defect of normalizing the
        CLIPPED exponent (which would return uniform 1/3 there).
        """
        got = np.asarray(
            ops.convert_to_numpy(
                self._golden_layer(output_mode='normalized')(self.GOLDEN_X)
            )
        )
        expected = self._nrbf_reference(
            self.GOLDEN_X, self.GOLDEN_CENTERS, self.GOLDEN_GAMMA_RAW
        )

        np.testing.assert_allclose(
            got, expected, rtol=1e-6, atol=1e-6,
            err_msg="normalized output does not equal phi / sum(phi)",
        )
        # ...and it is genuinely DIFFERENT from the basis output, so the check
        # above cannot pass by the branch being a no-op.
        assert not np.allclose(got, self.GOLDEN_BASIS, atol=1e-6)

    @pytest.mark.parametrize("shape", [(16, 4), (8, 12, 4), (16, 128), (4, 6, 256)])
    def test_normalized_sums_to_one(self, shape: Tuple[int, ...]) -> None:
        """SC11: normalized activations sum to 1.0 along the unit axis, at any rank.

        D-008: the last two shapes are REALISTIC feature dimensions (128, 256),
        past the ``D * gamma > 50`` saturation threshold. Sum-to-1 held even in
        the defective clipped implementation -- that is exactly why this property
        alone was insufficient -- but the shapes are carried here so that every
        normalized test runs in the regime the layer is actually used in.
        """
        layer = RBFLayer(units=5, output_mode='normalized')
        x = np.random.RandomState(3).normal(0, 2, shape).astype('float32')
        y = np.asarray(ops.convert_to_numpy(layer(x)))

        assert y.shape == shape[:-1] + (5,)
        assert not np.isnan(y).any() and not np.isinf(y).any()
        np.testing.assert_allclose(
            y.sum(axis=-1), np.ones(shape[:-1], dtype='float32'),
            rtol=1e-6, atol=1e-6,
        )

    def test_normalized_far_from_all_centers_selects_nearest(self) -> None:
        """D-008: far from every center, NRBF must give the TRUE distribution.

        This test previously asserted the output was uniform ``1/units`` and so
        PINNED the defect: softmaxing the 50.0-clipped exponent makes every unit
        saturate at the same value, and uniform-with-zero-gradient looks healthy
        because it still sums to 1. Selecting the nearest center far from the
        data is NRBF's defining property over plain RBF, and it is what the
        module docstring, the README and D-002 all claim.

        Uses the reviewer's worked example: with centers at 0, 3 and 6 along the
        first axis and ``x = 20``, the true NRBF is ~[2.5e-89, 4.1e-41, 1.0].
        """
        centers = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]], dtype='float32')
        x = np.array([[20.0, 0.0]], dtype='float32')

        layer = RBFLayer(units=3, repulsion_strength=0.0, output_mode='normalized')
        layer.build(x.shape)
        layer.centers.assign(centers)
        gamma_raw = np.asarray(ops.convert_to_numpy(layer.gamma_raw))

        y = np.asarray(ops.convert_to_numpy(layer(x)))
        expected = self._nrbf_reference(x, centers, gamma_raw)

        assert not np.isnan(y).any(), "NaN from the far-from-all-centers input"
        assert not np.isinf(y).any(), "Inf from the far-from-all-centers input"

        # The distribution is essentially one-hot on the NEAREST center (index 2).
        np.testing.assert_allclose(
            y, expected, rtol=1e-6, atol=1e-12,
            err_msg="far-from-all-centers NRBF does not match the float64 reference",
        )
        assert int(np.argmax(y)) == 2, "NRBF did not select the nearest center"
        assert y[0, 2] > 0.999, f"nearest-center mass collapsed: {y[0, 2]}"
        # Explicitly NOT the uniform 1/3 the clipped implementation returned.
        assert not np.allclose(y, 1.0 / 3.0, atol=1e-3), (
            "NRBF returned uniform 1/units far from every center -- the normalized "
            "arm is being fed the CLIPPED exponent (D-008)"
        )

    @pytest.mark.parametrize("dim,gamma_init", [(4, 1.0), (128, 1.0), (256, 1.0),
                                                (16, 5.0)])
    def test_normalized_gradients_flow(self, dim: int, gamma_init: float) -> None:
        """Gradients reach both centers and gamma_raw through the softmax.

        D-008 -- THE test this suite was missing. The original ran only at
        ``dim=4``, which is BELOW the ``D * gamma > 50`` clip-saturation
        threshold, so it could not observe a defect whose trigger is
        dimensionality. Softmaxing the clipped exponent gives ``ops.minimum`` a
        structurally zero gradient in the saturated branch: at ``dim=128`` with
        stock defaults both trainable weights received gradient EXACTLY 0.0 while
        the forward output still looked healthy (uniform, summing to 1).

        The parametrization deliberately spans both sides of the threshold.
        Measured against the pre-fix implementation: (128, 1.0) and (256, 1.0)
        FAIL with gradient max exactly 0.0 on both weights, while (4, 1.0) and
        (16, 5.0) PASS -- so the two added high-dimension cases are the
        load-bearing ones, and the low-dimension cases document where the old
        test's blind spot began. (16, 5.0) is retained as a partial-saturation
        control: ``D*gamma = 80`` clips most but not all units, so a nonzero
        gradient survives and the case cannot detect the defect. That is the
        point -- it records that "some saturation" is not the trigger.
        """
        layer = RBFLayer(units=5, gamma_init=gamma_init, output_mode='normalized')
        x = tf.constant(
            np.random.RandomState(9).normal(0, 1, (16, dim)).astype('float32')
        )

        with tf.GradientTape() as tape:
            loss = ops.mean(ops.square(layer(x, training=True)))
        grads = tape.gradient(loss, layer.trainable_weights)

        assert len(grads) == 2, "expected gradients for centers and gamma_raw"
        for weight, grad in zip(layer.trainable_weights, grads):
            assert grad is not None, f"None gradient for {weight.name}"
            g = np.asarray(ops.convert_to_numpy(grad))
            assert not np.isnan(g).any(), f"NaN gradient for {weight.name}"
            assert not np.isinf(g).any(), f"Inf gradient for {weight.name}"
            assert np.abs(g).max() > 0.0, (
                f"vanished gradient for {weight.name} at dim={dim}, "
                f"gamma_init={gamma_init} -- the normalized arm is a dead layer "
                "here (D-008: do not softmax the clipped exponent)"
            )

    def test_normalized_output_varies_at_realistic_dimension(self) -> None:
        """D-008: the forward half of the dead-layer check, independent of autodiff.

        At ``dim=128`` the clipped implementation returned a CONSTANT vector
        ``[1/units] * units`` for every sample regardless of input. A layer whose
        output does not depend on its input is dead whether or not a gradient
        test is present, so this asserts input-dependence directly.
        """
        layer = RBFLayer(units=6, output_mode='normalized')
        x = np.random.RandomState(11).normal(0, 1, (32, 128)).astype('float32')
        y = np.asarray(ops.convert_to_numpy(layer(x)))

        assert not np.allclose(y, 1.0 / 6.0, atol=1e-4), (
            "normalized output is the constant uniform vector at dim=128 -- "
            "the layer is dead (D-008)"
        )
        # Output genuinely varies BOTH across units and across samples.
        assert y.std(axis=-1).min() > 1e-6, "output is uniform across units"
        assert y.std(axis=0).max() > 1e-6, "output does not depend on the input"

    def test_invalid_output_mode_raises(self) -> None:
        """Validation matches the house `<param> must be ..., got <value>` shape."""
        with pytest.raises(ValueError, match="output_mode must be"):
            RBFLayer(units=4, output_mode='assignments')
        with pytest.raises(ValueError, match="output_mode must be"):
            RBFLayer(units=4, output_mode='mixture')

    def test_config_round_trip_preserves_output_mode(self) -> None:
        """Per-key config checks (never full-dict equality) at a NON-default value."""
        layer = RBFLayer(units=7, gamma_init=2.0, output_mode='normalized')
        config = layer.get_config()

        assert 'output_mode' in config
        assert config['output_mode'] == 'normalized'

        restored = RBFLayer.from_config(config)
        assert restored.output_mode == 'normalized'
        assert restored.units == 7
        assert restored.gamma_init == 2.0

        # A default-constructed layer must still round-trip as 'basis'.
        assert RBFLayer(units=7).get_config()['output_mode'] == 'basis'

    def test_normalized_save_load_round_trip(self) -> None:
        """SC14 / guide 8.2: a REAL .keras archive round-trip, not a get_config cycle.

        Uses the NON-default output_mode on purpose: with the default, a
        from_config that dropped the key entirely would still reconstruct a
        matching layer and the test could not fail.
        """
        x = np.random.RandomState(21).normal(0, 1, (16, 4)).astype('float32')

        inputs = keras.Input(shape=(4,))
        rbf = RBFLayer(units=6, output_mode='normalized')(inputs)
        outputs = keras.layers.Dense(3, activation='softmax')(rbf)
        model = keras.Model(inputs, outputs)

        original = ops.convert_to_numpy(model(x))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'rbf_normalized.keras')
            model.save(filepath)
            loaded = keras.models.load_model(filepath)

            np.testing.assert_allclose(
                original, ops.convert_to_numpy(loaded(x)),
                rtol=1e-6, atol=1e-6,
                err_msg="predictions differ after the .keras round-trip",
            )

            reloaded_rbf = [
                lyr for lyr in loaded.layers if isinstance(lyr, RBFLayer)
            ]
            assert len(reloaded_rbf) == 1
            assert reloaded_rbf[0].output_mode == 'normalized', (
                "output_mode was not restored from the .keras archive"
            )

    def test_factory_honors_output_mode(self) -> None:
        """SC13: BOTH factory sites.

        Site (a): the layer's attribute is asserted, not merely that the call did
        not raise -- a missing MIXTURE_REGISTRY key makes create_mixture_layer
        SILENTLY drop the kwarg and return a basis-mode layer.
        Site (b): validate_mixture_config must accept RBF's own vocabulary and
        must NOT have been broadened for the sibling types.
        """
        from dl_techniques.layers.mixtures.factory import (
            create_mixture_layer, validate_mixture_config, MIXTURE_REGISTRY,
        )

        # (a) registry default mirrors the constructor default.
        assert MIXTURE_REGISTRY['rbf']['optional_params']['output_mode'] == 'basis'

        layer = create_mixture_layer('rbf', units=4, output_mode='normalized')
        assert layer.output_mode == 'normalized', (
            "create_mixture_layer dropped output_mode -- registry site missed"
        )
        assert create_mixture_layer('rbf', units=4).output_mode == 'basis'

        y = np.asarray(ops.convert_to_numpy(
            layer(np.random.RandomState(5).normal(0, 1, (8, 3)).astype('float32'))
        ))
        np.testing.assert_allclose(
            y.sum(axis=-1), np.ones(8, dtype='float32'), rtol=1e-6, atol=1e-6,
            err_msg="factory-built layer did not actually normalize",
        )

        # (b) validator is type-scoped in BOTH directions.
        validate_mixture_config('rbf', units=4, output_mode='normalized')
        validate_mixture_config('rbf', units=4, output_mode='basis')
        with pytest.raises(ValueError, match="output_mode"):
            validate_mixture_config('rbf', units=4, output_mode='assignments')
        validate_mixture_config('kmeans', n_clusters=4, output_mode='assignments')
        with pytest.raises(ValueError, match="output_mode"):
            validate_mixture_config('kmeans', n_clusters=4, output_mode='normalized')
        with pytest.raises(ValueError, match="output_mode"):
            validate_mixture_config('gmm', n_components=4, output_mode='basis')


if __name__ == '__main__':
    # Run with: python -m pytest test_radial_basis_function.py -v
    pytest.main([__file__, '-v', '--tb=short'])