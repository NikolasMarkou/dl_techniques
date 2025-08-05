"""
Test suite for KMeansLayer implementation.

This module provides comprehensive tests for the differentiable K-means clustering layer,
covering initialization, forward passes, training behavior, serialization, and edge cases.

The tests are designed to work with Keras 3.x and avoid TensorFlow-specific operations
to ensure compatibility across different backends (TensorFlow, JAX, PyTorch).
"""

import pytest
import numpy as np
import keras
from keras import ops
from typing import Dict, Any, List, Tuple, Union

from dl_techniques.layers.kmeans import KMeansLayer


# Test fixtures
@pytest.fixture
def random_seed() -> int:
    """Fixture for consistent random numbers across tests.

    Returns:
        Random seed value for reproducible tests
    """
    return 42


@pytest.fixture
def basic_config() -> Dict[str, Any]:
    """Fixture for basic layer configuration.

    Returns:
        Dictionary containing basic KMeansLayer configuration
    """
    return {
        "n_clusters": 4,
        "temperature": 0.1,
        "output_mode": "assignments",
        "cluster_axis": -1,
        "centroid_initializer": "glorot_normal"  # Use standard initializer for tests
    }


@pytest.fixture
def sample_data_2d() -> keras.KerasTensor:
    """Generate 2D sample data for testing.

    Returns:
        Sample tensor of shape (batch_size, features)
    """
    # Create deterministic test data
    np.random.seed(42)
    data = np.random.normal(0, 1, (32, 64)).astype(np.float32)
    return keras.ops.convert_to_tensor(data)


@pytest.fixture
def sample_data_4d() -> keras.KerasTensor:
    """Generate 4D sample data for testing (e.g., image-like data).

    Returns:
        Sample tensor of shape (batch_size, height, width, channels)
    """
    np.random.seed(42)
    data = np.random.normal(0, 1, (8, 28, 28, 3)).astype(np.float32)
    return keras.ops.convert_to_tensor(data)


class TestKMeansLayerInitialization:
    """Test suite for KMeansLayer initialization."""

    def test_valid_initialization(self, basic_config: Dict[str, Any]) -> None:
        """Test layer initialization with valid parameters.

        Args:
            basic_config: Basic configuration dictionary
        """
        layer = KMeansLayer(**basic_config)

        assert layer.n_clusters == basic_config["n_clusters"]
        assert layer.temperature == basic_config["temperature"]
        assert layer.output_mode == basic_config["output_mode"]
        assert layer.cluster_axis == [basic_config["cluster_axis"]]

    def test_orthonormal_initializer_fallback(self, random_seed: int) -> None:
        """Test that orthonormal initializer works and falls back appropriately."""
        from dl_techniques.initializers.orthonormal_initializer import OrthonormalInitializer

        # Test case where orthonormal should work (n_clusters <= feature_dims)
        layer_small = KMeansLayer(
            n_clusters=4,
            centroid_initializer=OrthonormalInitializer(seed=random_seed),
            random_seed=random_seed
        )

        # Build with small feature dimension
        test_input = keras.ops.ones((1, 10))  # 10 features, 4 clusters
        _ = layer_small(test_input)

        # Should use orthonormal initialization
        assert layer_small.centroids is not None

        # Test case where it should fall back (n_clusters > feature_dims)
        layer_large = KMeansLayer(
            n_clusters=20,  # More clusters than features
            centroid_initializer=OrthonormalInitializer(seed=random_seed),
            random_seed=random_seed
        )

        # Build with small feature dimension
        test_input_small = keras.ops.ones((1, 5))  # 5 features, 20 clusters
        _ = layer_large(test_input_small)

        # Should fall back to glorot_normal
        assert layer_large.centroids is not None

    @pytest.mark.parametrize("invalid_param,invalid_value,expected_error", [
        ("n_clusters", 0, "n_clusters must be a positive integer"),
        ("n_clusters", -1, "n_clusters must be a positive integer"),
        ("n_clusters", "invalid", "n_clusters must be a positive integer"),
        ("temperature", 0, "temperature must be positive"),
        ("temperature", -0.1, "temperature must be positive"),
        ("momentum", -0.1, "momentum must be in \\[0, 1\\)"),
        ("momentum", 1.5, "momentum must be in \\[0, 1\\)"),
        ("centroid_lr", 0, "centroid_lr must be in \\(0, 1\\]"),
        ("centroid_lr", 1.5, "centroid_lr must be in \\(0, 1\\]"),
        ("repulsion_strength", -0.1, "repulsion_strength must be non-negative"),
        ("min_distance", 0, "min_distance must be positive"),
        ("min_distance", -1, "min_distance must be positive"),
        ("output_mode", "invalid", "output_mode must be 'assignments' or 'mixture'")
    ])
    def test_invalid_initialization(
        self,
        invalid_param: str,
        invalid_value: Any,
        expected_error: str,
        basic_config: Dict[str, Any]
    ) -> None:
        """Test layer initialization with invalid parameters.

        Args:
            invalid_param: Parameter name to make invalid
            invalid_value: Invalid value to use
            expected_error: Expected error message substring
            basic_config: Basic valid configuration
        """
        invalid_params = {**basic_config, invalid_param: invalid_value}

        with pytest.raises(ValueError, match=expected_error):
            KMeansLayer(**invalid_params)

    def test_custom_initializers_and_regularizers(self) -> None:
        """Test custom initializers and regularizers."""
        custom_initializer = keras.initializers.RandomNormal(stddev=0.5)
        custom_regularizer = keras.regularizers.L2(0.01)

        layer = KMeansLayer(
            n_clusters=5,
            centroid_initializer=custom_initializer,
            centroid_regularizer=custom_regularizer
        )

        assert layer.centroid_initializer == custom_initializer
        assert layer.centroid_regularizer == custom_regularizer


class TestKMeansLayerShapes:
    """Test suite for KMeansLayer shape handling."""

    @pytest.mark.parametrize("input_shape,cluster_axis,expected_feature_dims", [
        ((32, 64), -1, 64),
        ((32, 28, 28, 3), -1, 3),
        ((32, 100, 128), 1, 100),
        ((32, 28, 28, 3), [1, 2], 28 * 28),
        ((16, 10, 20, 30), [1, 2, 3], 10 * 20 * 30)
    ])
    def test_build_with_different_shapes(
        self,
        input_shape: Tuple[int, ...],
        cluster_axis: Union[int, List[int]],
        expected_feature_dims: int,
        basic_config: Dict[str, Any]
    ) -> None:
        """Test layer building with different input shapes and clustering axes.

        Args:
            input_shape: Input tensor shape
            cluster_axis: Axis or axes to cluster on
            expected_feature_dims: Expected feature dimensions
            basic_config: Basic layer configuration
        """
        config = {**basic_config, "cluster_axis": cluster_axis}
        layer = KMeansLayer(**config)

        # Create input tensor
        inputs = keras.ops.ones(input_shape)

        # Build layer by calling it
        output = layer(inputs)

        # Check that layer built correctly
        assert layer.built
        assert layer.feature_dims == expected_feature_dims
        assert layer.centroids.shape == (basic_config["n_clusters"], expected_feature_dims)

    def test_compute_output_shape_assignments(self, basic_config: Dict[str, Any]) -> None:
        """Test output shape computation for assignments mode."""
        layer = KMeansLayer(**basic_config)

        # Single axis clustering
        input_shape = (32, 64)
        expected_shape = (32, basic_config["n_clusters"])
        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape

        # Multi-axis clustering
        layer_multi = KMeansLayer(**{**basic_config, "cluster_axis": [1, 2]})
        input_shape_multi = (32, 28, 28, 3)
        expected_shape_multi = (32, basic_config["n_clusters"], 3)
        computed_shape_multi = layer_multi.compute_output_shape(input_shape_multi)
        assert computed_shape_multi == expected_shape_multi

    def test_compute_output_shape_mixture(self, basic_config: Dict[str, Any]) -> None:
        """Test output shape computation for mixture mode."""
        config = {**basic_config, "output_mode": "mixture"}
        layer = KMeansLayer(**config)

        input_shape = (32, 64)
        expected_shape = input_shape  # Same as input for mixture mode
        computed_shape = layer.compute_output_shape(input_shape)
        assert computed_shape == expected_shape


class TestKMeansLayerForwardPass:
    """Test suite for KMeansLayer forward pass behavior."""

    @pytest.mark.parametrize("output_mode", ["assignments", "mixture"])
    def test_output_modes(
        self,
        output_mode: str,
        sample_data_2d: keras.KerasTensor,
        basic_config: Dict[str, Any]
    ) -> None:
        """Test different output modes produce correct outputs.

        Args:
            output_mode: Output mode to test
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        config = {**basic_config, "output_mode": output_mode}
        layer = KMeansLayer(**config)

        output = layer(sample_data_2d)

        if output_mode == "assignments":
            # Check that outputs are valid probabilities
            assert output.shape == (sample_data_2d.shape[0], basic_config["n_clusters"])

            # Check that assignments sum to 1 (within tolerance)
            assignment_sums = ops.sum(output, axis=-1)
            expected_sums = ops.ones_like(assignment_sums)

            # Use numpy for assertion since keras ops might not have allclose
            assert np.allclose(
                ops.convert_to_numpy(assignment_sums),
                ops.convert_to_numpy(expected_sums),
                rtol=1e-5
            )

            # Check range [0, 1]
            assert ops.convert_to_numpy(ops.min(output)) >= 0
            assert ops.convert_to_numpy(ops.max(output)) <= 1

        else:  # mixture mode
            # Check output shape matches input shape
            assert output.shape == sample_data_2d.shape

    def test_different_temperatures(
        self,
        sample_data_2d: keras.KerasTensor,
        basic_config: Dict[str, Any]
    ) -> None:
        """Test that temperature affects assignment softness.

        Args:
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        # Remove temperature from config to avoid duplicate
        config_base = {k: v for k, v in basic_config.items() if k != 'temperature'}

        # Low temperature (harder assignments)
        layer_hard = KMeansLayer(**config_base, temperature=0.01)
        output_hard = layer_hard(sample_data_2d)

        # High temperature (softer assignments)
        layer_soft = KMeansLayer(**config_base, temperature=1.0)
        output_soft = layer_soft(sample_data_2d)

        # Compute entropy of assignments
        eps = 1e-7
        hard_entropy = -ops.mean(
            ops.sum(output_hard * ops.log(output_hard + eps), axis=-1)
        )
        soft_entropy = -ops.mean(
            ops.sum(output_soft * ops.log(output_soft + eps), axis=-1)
        )

        # Hard assignments should have lower entropy
        assert ops.convert_to_numpy(hard_entropy) < ops.convert_to_numpy(soft_entropy)

    def test_training_vs_inference_mode(
        self,
        sample_data_2d: keras.KerasTensor,
        basic_config: Dict[str, Any]
    ) -> None:
        """Test behavior difference between training and inference modes.

        Args:
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        layer = KMeansLayer(**basic_config, random_seed=42, centroid_lr=0.5)  # Higher LR for visible changes

        # Get initial centroids
        _ = layer(sample_data_2d, training=False)  # Build layer
        initial_centroids = ops.convert_to_numpy(layer.centroids)

        # Multiple training steps to ensure visible changes
        for _ in range(5):
            _ = layer(sample_data_2d, training=True)
        updated_centroids = ops.convert_to_numpy(layer.centroids)

        # Centroids should have changed during training
        centroid_changes = np.abs(updated_centroids - initial_centroids)
        max_change = np.max(centroid_changes)
        assert max_change > 1e-3, f"Max centroid change {max_change} too small - centroids not updating"

        # Reset centroids - should produce different values
        layer.reset_centroids()
        reset_centroids = ops.convert_to_numpy(layer.centroids)

        # Check reset produced different values
        reset_vs_initial = np.abs(reset_centroids - initial_centroids)
        reset_vs_updated = np.abs(reset_centroids - updated_centroids)

        # At least one should be significantly different
        assert (np.max(reset_vs_initial) > 1e-3 or np.max(reset_vs_updated) > 1e-3), \
            "Reset centroids should be different from previous values"


class TestKMeansLayerTraining:
    """Test suite for KMeansLayer training dynamics."""

    def test_centroid_updates_with_momentum(
        self,
        sample_data_2d: keras.KerasTensor,
        basic_config: Dict[str, Any]
    ) -> None:
        """Test that centroids update correctly with momentum.

        Args:
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        layer = KMeansLayer(**basic_config, momentum=0.9, centroid_lr=0.3, random_seed=42)  # Higher LR

        # Initial forward pass to build layer
        _ = layer(sample_data_2d, training=False)
        initial_centroids = ops.convert_to_numpy(layer.centroids)
        initial_momentum = ops.convert_to_numpy(layer.centroid_momentum)

        # Should start with zero momentum
        assert np.allclose(initial_momentum, 0, atol=1e-6)

        # Multiple training steps to accumulate momentum
        for _ in range(3):
            _ = layer(sample_data_2d, training=True)

        updated_centroids = ops.convert_to_numpy(layer.centroids)
        updated_momentum = ops.convert_to_numpy(layer.centroid_momentum)

        # Check for visible changes
        centroid_change = np.max(np.abs(updated_centroids - initial_centroids))
        momentum_change = np.max(np.abs(updated_momentum - initial_momentum))

        assert centroid_change > 1e-3, f"Centroids should change during training (change: {centroid_change})"
        assert momentum_change > 1e-6, f"Momentum should accumulate during training (change: {momentum_change})"

    def test_repulsion_forces(self, basic_config: Dict[str, Any]) -> None:
        """Test that repulsion forces prevent centroid collapse.

        Args:
            basic_config: Basic layer configuration
        """
        # Create layer with strong repulsion
        layer = KMeansLayer(
            **basic_config,
            repulsion_strength=0.5,  # Moderate strength
            min_distance=1.0,
            centroid_lr=0.2,  # Higher learning rate
            random_seed=42
        )

        # Create data that might cause collapse (all points similar)
        similar_data = keras.ops.ones((32, 64)) * 0.1 + keras.random.normal((32, 64)) * 0.01

        # Get initial centroid distances
        _ = layer(similar_data, training=False)  # Build layer
        initial_centroids = ops.convert_to_numpy(layer.centroids)

        # Multiple training steps with potentially collapsing data
        for _ in range(10):
            _ = layer(similar_data, training=True)

        # Check that centroids maintain reasonable separation
        final_centroids = ops.convert_to_numpy(layer.centroids)

        # Compute pairwise distances between centroids
        distances = []
        for i in range(final_centroids.shape[0]):
            for j in range(i + 1, final_centroids.shape[0]):
                dist = np.linalg.norm(final_centroids[i] - final_centroids[j])
                distances.append(dist)

        # Should have reasonable separation (not collapsed)
        min_distance = min(distances) if distances else 1.0
        mean_distance = np.mean(distances) if distances else 1.0

        assert min_distance > 0.05, f"Minimum centroid distance too small: {min_distance}"
        assert mean_distance > 0.1, f"Mean centroid distance too small: {mean_distance}"


class TestKMeansLayerSerialization:
    """Test suite for KMeansLayer serialization and deserialization."""

    def test_get_config(self, basic_config: Dict[str, Any]) -> None:
        """Test layer configuration serialization.

        Args:
            basic_config: Basic layer configuration
        """
        layer = KMeansLayer(**basic_config)
        config = layer.get_config()

        # Check all important parameters are saved
        assert config["n_clusters"] == basic_config["n_clusters"]
        assert config["temperature"] == basic_config["temperature"]
        assert config["output_mode"] == basic_config["output_mode"]
        assert config["cluster_axis"] == [basic_config["cluster_axis"]]

    def test_from_config(self, basic_config: Dict[str, Any]) -> None:
        """Test layer recreation from configuration.

        Args:
            basic_config: Basic layer configuration
        """
        original_layer = KMeansLayer(**basic_config)
        config = original_layer.get_config()

        # Recreate layer from config
        recreated_layer = KMeansLayer.from_config(config)

        # Check all parameters match
        assert recreated_layer.n_clusters == original_layer.n_clusters
        assert recreated_layer.temperature == original_layer.temperature
        assert recreated_layer.output_mode == original_layer.output_mode
        assert recreated_layer.cluster_axis == original_layer.cluster_axis

    def test_build_config_serialization(self, sample_data_2d: keras.KerasTensor, basic_config: Dict[str, Any]) -> None:
        """Test build configuration serialization for model saving.

        Args:
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        layer = KMeansLayer(**basic_config)

        # Build the layer
        _ = layer(sample_data_2d)

        # Get build config
        build_config = layer.get_build_config()
        assert "input_shape" in build_config
        assert build_config["input_shape"] == sample_data_2d.shape

        # Test building from config
        new_layer = KMeansLayer(**basic_config)
        new_layer.build_from_config(build_config)

        assert new_layer.built
        assert new_layer.feature_dims == layer.feature_dims


class TestKMeansLayerEdgeCases:
    """Test suite for KMeansLayer edge cases and error handling."""

    def test_single_cluster(self, sample_data_2d: keras.KerasTensor) -> None:
        """Test behavior with single cluster.

        Args:
            sample_data_2d: Sample 2D input data
        """
        layer = KMeansLayer(n_clusters=1, centroid_initializer="glorot_normal")
        output = layer(sample_data_2d)

        # All assignments should be 1.0 for single cluster
        expected = ops.ones_like(output)
        assert np.allclose(
            ops.convert_to_numpy(output),
            ops.convert_to_numpy(expected),
            rtol=1e-5
        )

    def test_more_clusters_than_samples(self) -> None:
        """Test behavior when n_clusters > batch_size."""
        # Small batch, many clusters
        small_data = keras.random.normal((2, 10))
        layer = KMeansLayer(n_clusters=5, centroid_initializer="glorot_normal")

        # Should still work without errors
        output = layer(small_data, training=True)
        assert output.shape == (2, 5)

    def test_reset_centroids_functionality(self, sample_data_2d: keras.KerasTensor, basic_config: Dict[str, Any]) -> None:
        """Test centroid reset functionality.

        Args:
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        layer = KMeansLayer(**basic_config)

        # Build layer
        _ = layer(sample_data_2d)
        original_centroids = ops.convert_to_numpy(layer.centroids)

        # Reset with custom values
        new_centroids = keras.random.normal((basic_config["n_clusters"], sample_data_2d.shape[-1]))
        layer.reset_centroids(new_centroids)

        # Check centroids changed
        reset_centroids = ops.convert_to_numpy(layer.centroids)
        assert np.allclose(reset_centroids, ops.convert_to_numpy(new_centroids))

        # Reset without providing new values (reinitialize with random)
        layer.reset_centroids()
        reinitialized_centroids = ops.convert_to_numpy(layer.centroids)

        # Check that reinitialized values are different
        diff_from_reset = np.abs(reinitialized_centroids - reset_centroids)
        diff_from_original = np.abs(reinitialized_centroids - original_centroids)

        # Should be different from both previous states
        assert np.max(diff_from_reset) > 1e-3, "Reinitialized centroids should differ from reset centroids"
        assert np.max(diff_from_original) > 1e-3, "Reinitialized centroids should differ from original centroids"

    def test_cluster_centers_property(self, sample_data_2d: keras.KerasTensor, basic_config: Dict[str, Any]) -> None:
        """Test cluster_centers property access.

        Args:
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        layer = KMeansLayer(**basic_config)

        # Before building, should be None
        assert layer.cluster_centers is None

        # After building, should return centroids
        _ = layer(sample_data_2d)
        centers = layer.cluster_centers

        assert centers is not None
        assert centers.shape == (basic_config["n_clusters"], sample_data_2d.shape[-1])

        # Should be the same as centroids attribute
        assert np.allclose(
            ops.convert_to_numpy(centers),
            ops.convert_to_numpy(layer.centroids)
        )

    def test_invalid_centroid_reset(self, sample_data_2d: keras.KerasTensor, basic_config: Dict[str, Any]) -> None:
        """Test error handling in centroid reset.

        Args:
            sample_data_2d: Sample 2D input data
            basic_config: Basic layer configuration
        """
        layer = KMeansLayer(**basic_config)

        # Should raise error if layer not built
        with pytest.raises(ValueError, match="Layer must be built"):
            layer.reset_centroids()

        # Build layer
        _ = layer(sample_data_2d)

        # Should raise error with wrong shape
        wrong_shape_centroids = keras.random.normal((basic_config["n_clusters"], 10))  # Wrong feature dim
        with pytest.raises(ValueError, match="new_centroids must have shape"):
            layer.reset_centroids(wrong_shape_centroids)


class TestKMeansLayerIntegration:
    """Test suite for KMeansLayer integration with Keras models."""

    def test_in_sequential_model(self, basic_config: Dict[str, Any]) -> None:
        """Test KMeansLayer integration in Sequential model.

        Args:
            basic_config: Basic layer configuration
        """
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(64,)),
            KMeansLayer(**basic_config),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Test compilation
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Test forward pass
        test_input = keras.random.normal((16, 64))
        output = model(test_input)

        assert output.shape == (16, 10)

    def test_in_functional_model(self, basic_config: Dict[str, Any]) -> None:
        """Test KMeansLayer integration in Functional API model.

        Args:
            basic_config: Basic layer configuration
        """
        inputs = keras.layers.Input(shape=(64,))
        x = keras.layers.Dense(128, activation='relu')(inputs)
        x = KMeansLayer(**basic_config)(x)
        outputs = keras.layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Test forward pass
        test_input = keras.random.normal((16, 64))
        output = model(test_input)

        assert output.shape == (16, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])