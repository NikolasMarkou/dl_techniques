import pytest
import numpy as np
import tensorflow as tf
from dl_techniques.layers.kmeans import KMeansLayer

# ---------------------------------------------------------------------

@pytest.fixture
def random_seed():
    """Fixture for consistent random numbers."""
    return 42


@pytest.fixture
def basic_config():
    """Fixture for basic layer configuration."""
    return {
        "n_clusters": 4,
        "temperature": 0.1,
        "output_mode": "assignments",
        "cluster_axis": -1
    }

# ---------------------------------------------------------------------


class TestDifferentiableKMeansLayer:
    """Test suite for DifferentiableKMeansLayer."""

    def test_initialization(self, basic_config):
        """Test layer initialization with valid parameters."""
        layer = KMeansLayer(**basic_config)
        assert layer.n_clusters == basic_config["n_clusters"]
        assert layer.temperature == basic_config["temperature"]
        assert layer.output_mode == basic_config["output_mode"]
        assert layer.cluster_axis == [basic_config["cluster_axis"]]

    @pytest.mark.parametrize("invalid_config", [
        {"n_clusters": 0},
        {"n_clusters": -1},
        {"temperature": 0},
        {"temperature": -0.1},
        {"output_mode": "invalid"}
    ])
    def test_invalid_initialization(self, invalid_config, basic_config):
        """Test layer initialization with invalid parameters."""
        invalid_params = {**basic_config, **invalid_config}
        with pytest.raises(ValueError):
            KMeansLayer(**invalid_params)

    @pytest.mark.parametrize("input_shape,cluster_axis", [
        ((32, 64), -1),
        ((32, 28, 28, 3), -1),
        ((32, 100, 128), 1),
        ((32, 28, 28, 3), [1, 2])
    ])
    def test_build_shapes(self, input_shape, cluster_axis, basic_config):
        """Test layer building with different input shapes."""
        config = {**basic_config, "cluster_axis": cluster_axis}
        layer = KMeansLayer(**config)

        inputs = tf.random.normal(input_shape)
        output = layer(inputs)

        # Check output shape
        expected_shape = list(input_shape)
        if isinstance(cluster_axis, list):
            # For multiple axes, collapse them into first axis in list
            for axis in reversed(sorted(cluster_axis)):
                expected_shape.pop(axis)
            expected_shape.insert(cluster_axis[0], basic_config["n_clusters"])
        else:
            expected_shape[cluster_axis] = basic_config["n_clusters"]

        assert output.shape == tuple(expected_shape)

    @pytest.mark.parametrize("output_mode", ["assignments", "mixture"])
    def test_output_modes(self, output_mode, basic_config):
        """Test different output modes."""
        config = {**basic_config, "output_mode": output_mode}
        layer = KMeansLayer(**config)

        inputs = tf.random.normal((32, 64))
        output = layer(inputs)

        if output_mode == "assignments":
            # Check that outputs sum to 1 for assignments
            sums = tf.reduce_sum(output, axis=-1)
            tf.debugging.assert_near(sums, tf.ones_like(sums), rtol=1e-5)
            # Check range [0, 1]
            assert tf.reduce_all(output >= 0)
            assert tf.reduce_all(output <= 1)
        else:  # mixture mode
            # Check output shape matches input shape
            assert output.shape == inputs.shape

    def test_training_updates(self, basic_config, random_seed):
        """Test that centroids update during training."""
        tf.random.set_seed(random_seed)
        layer = KMeansLayer(**basic_config, random_seed=random_seed)

        inputs = tf.random.normal((32, 64))

        # Get initial centroids
        _ = layer(inputs)  # This builds the layer
        initial_centroids = tf.identity(layer.centroids)

        # Training step
        _ = layer(inputs, training=True)

        # Centroids should have changed
        assert not tf.reduce_all(tf.equal(initial_centroids, layer.centroids))

    def test_temperature_effect(self, basic_config):
        """Test that temperature affects the softness of assignments."""
        inputs = tf.random.normal((32, 64))

        # Remove temperature from basic config to avoid duplicate parameter
        config_without_temp = {k: v for k, v in basic_config.items() if k != 'temperature'}

        # Low temperature (harder assignments)
        layer_hard = KMeansLayer(**config_without_temp, temperature=0.01)
        output_hard = layer_hard(inputs)

        # High temperature (softer assignments)
        layer_soft = KMeansLayer(**config_without_temp, temperature=1.0)
        output_soft = layer_soft(inputs)

        # Hard assignments should be more extreme (closer to 0 or 1)
        hard_entropy = -tf.reduce_mean(
            tf.reduce_sum(output_hard * tf.math.log(output_hard + 1e-7), axis=-1)
        )
        soft_entropy = -tf.reduce_mean(
            tf.reduce_sum(output_soft * tf.math.log(output_soft + 1e-7), axis=-1)
        )

        assert hard_entropy < soft_entropy

    def test_serialization(self, basic_config):
        """Test layer serialization and deserialization."""
        layer = KMeansLayer(**basic_config)
        config = layer.get_config()

        # Recreate layer from config
        new_layer = KMeansLayer.from_config(config)

        # Check all parameters are preserved
        assert new_layer.n_clusters == layer.n_clusters
        assert new_layer.temperature == layer.temperature
        assert new_layer.output_mode == layer.output_mode
        assert new_layer.cluster_axis == layer.cluster_axis
        assert new_layer.random_seed == layer.random_seed

# ---------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__])
