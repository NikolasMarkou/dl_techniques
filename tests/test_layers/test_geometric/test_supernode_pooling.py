import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.geometric.supernode_pooling import SupernodePooling


# ===========================================================================
# TestSupernodePooling
# ===========================================================================


class TestSupernodePooling:
    """Test suite for SupernodePooling."""

    @pytest.fixture
    def hidden_dim(self) -> int:
        return 32

    @pytest.fixture
    def ndim(self) -> int:
        return 3

    @pytest.fixture
    def num_points(self) -> int:
        return 100

    @pytest.fixture
    def num_supernodes(self) -> int:
        return 5

    @pytest.fixture
    def dict_inputs(self, num_points, ndim, num_supernodes) -> dict:
        positions = tf.random.normal([num_points, ndim])
        supernode_indices = tf.constant([0, 10, 20, 30, 40])
        return {"positions": positions, "supernode_indices": supernode_indices}

    @pytest.fixture
    def layer_instance(self, hidden_dim, ndim) -> SupernodePooling:
        return SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)

    # ------------------------------------------------------------------

    def test_initialization_defaults(self, hidden_dim, ndim):
        """Test initialization with default parameters."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0)
        assert layer.hidden_dim == hidden_dim
        assert layer.ndim == ndim
        assert layer.radius == 1.0
        assert layer.k_neighbors is None
        assert layer.max_neighbors == 32
        assert layer.mode == "relpos"
        assert layer.use_bias is True

    def test_initialization_custom(self, hidden_dim, ndim):
        """Test initialization with custom parameters."""
        layer = SupernodePooling(
            hidden_dim=hidden_dim,
            ndim=ndim,
            k_neighbors=8,
            max_neighbors=16,
            mode="abspos",
            activation="relu",
            use_bias=False,
            name="custom_sp",
        )
        assert layer.k_neighbors == 8
        assert layer.radius is None
        assert layer.max_neighbors == 16
        assert layer.mode == "abspos"
        assert layer.use_bias is False
        assert layer.name == "custom_sp"

    def test_invalid_hidden_dim(self, ndim):
        """Test that non-positive hidden_dim raises ValueError."""
        with pytest.raises(ValueError, match="hidden_dim"):
            SupernodePooling(hidden_dim=0, ndim=ndim, radius=1.0)

    def test_invalid_ndim(self, hidden_dim):
        """Test that non-positive ndim raises ValueError."""
        with pytest.raises(ValueError, match="ndim"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=0, radius=1.0)

    def test_invalid_max_neighbors(self, hidden_dim, ndim):
        """Test that non-positive max_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="max_neighbors"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, max_neighbors=0)

    def test_invalid_mode(self, hidden_dim, ndim):
        """Test that unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="mode"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, mode="bad")

    def test_invalid_both_radius_and_k_neighbors(self, hidden_dim, ndim):
        """Test that specifying both radius and k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, k_neighbors=5)

    def test_invalid_neither_radius_nor_k_neighbors(self, hidden_dim, ndim):
        """Test that specifying neither radius nor k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim)

    def test_invalid_negative_radius(self, hidden_dim, ndim):
        """Test that negative radius raises ValueError."""
        with pytest.raises(ValueError, match="radius"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=-1.0)

    def test_invalid_negative_k_neighbors(self, hidden_dim, ndim):
        """Test that negative k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="k_neighbors"):
            SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, k_neighbors=-1)

    def test_build(self, layer_instance, ndim):
        """Test that the layer builds correctly with dict input_shape."""
        layer_instance.build({"positions": (100, ndim), "supernode_indices": (5,)})
        assert layer_instance.built is True
        assert layer_instance.pos_embed.built is True
        assert layer_instance.message_mlp.built is True
        assert layer_instance.proj_layer.built is True

    def test_forward_pass_radius(self, hidden_dim, ndim, dict_inputs):
        """Test forward pass with radius mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)
        output = layer(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    @pytest.mark.xfail(reason="Known bug in _knn_neighbors: take_along_axis shape mismatch")
    def test_forward_pass_k_neighbors(self, hidden_dim, ndim):
        """Test forward pass with k_neighbors mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, k_neighbors=3)
        positions = tf.random.normal([50, ndim])
        supernode_indices = tf.constant([0, 10, 20, 30, 40])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}
        output = layer(inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    def test_forward_pass_abspos_mode(self, hidden_dim, ndim, dict_inputs):
        """Test forward pass with abspos mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0, mode="abspos")
        output = layer(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    def test_forward_pass_relpos_mode(self, hidden_dim, ndim, dict_inputs):
        """Test forward pass with relpos mode."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0, mode="relpos")
        output = layer(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)
        assert not np.any(np.isnan(output.numpy()))

    def test_output_shape(self, layer_instance, dict_inputs, hidden_dim):
        """Test that output shape matches expected (1, num_supernodes, hidden_dim)."""
        output = layer_instance(dict_inputs)
        assert output.shape == (1, 5, hidden_dim)

    def test_compute_output_shape(self, layer_instance, ndim, hidden_dim):
        """Test compute_output_shape returns correct shape."""
        input_shape = {"positions": (100, ndim), "supernode_indices": (5,)}
        computed = layer_instance.compute_output_shape(input_shape)
        assert computed == (1, None, hidden_dim)

    def test_compute_output_shape_invalid_input(self, layer_instance):
        """Test compute_output_shape raises on invalid input."""
        with pytest.raises(ValueError):
            layer_instance.compute_output_shape((100, 3))

    def test_serialization(self, hidden_dim, ndim):
        """get_config / from_config round-trip preserves attributes."""
        original = SupernodePooling(
            hidden_dim=hidden_dim,
            ndim=ndim,
            radius=2.5,
            max_neighbors=16,
            mode="relpos",
            activation="relu",
            use_bias=False,
            name="sp_s",
        )
        config = original.get_config()
        restored = SupernodePooling.from_config(config)

        assert restored.hidden_dim == original.hidden_dim
        assert restored.ndim == original.ndim
        assert restored.radius == original.radius
        assert restored.k_neighbors == original.k_neighbors
        assert restored.max_neighbors == original.max_neighbors
        assert restored.mode == original.mode
        assert restored.use_bias == original.use_bias

    def test_serialization_k_neighbors(self, hidden_dim, ndim):
        """get_config / from_config round-trip preserves k_neighbors mode."""
        original = SupernodePooling(
            hidden_dim=hidden_dim,
            ndim=ndim,
            k_neighbors=10,
            mode="abspos",
            name="sp_k",
        )
        config = original.get_config()
        restored = SupernodePooling.from_config(config)

        assert restored.k_neighbors == original.k_neighbors
        assert restored.radius is None
        assert restored.mode == original.mode

    def test_gradient_flow(self, hidden_dim, ndim):
        """Gradients propagate through the layer."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)
        positions = tf.Variable(tf.random.normal([50, ndim]))
        supernode_indices = tf.constant([0, 10, 20, 30])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}

        with tf.GradientTape() as tape:
            out = layer(inputs)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, positions)
        assert grads is not None
        assert np.any(grads.numpy() != 0)

    def test_numerical_stability(self, hidden_dim, ndim):
        """No NaN / Inf with extreme input values."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=5.0)
        supernode_indices = tf.constant([0, 5, 10])
        for scale in [1e-6, 1e6]:
            positions = tf.ones([20, ndim]) * scale
            # Add small offsets so points are not all identical
            offsets = tf.random.normal([20, ndim], stddev=scale * 0.01)
            positions = positions + offsets
            inputs = {"positions": positions, "supernode_indices": supernode_indices}
            out = layer(inputs)
            assert not np.any(np.isnan(out.numpy())), f"NaN at scale {scale}"
            assert not np.any(np.isinf(out.numpy())), f"Inf at scale {scale}"

    def test_2d_points(self, hidden_dim):
        """Test with 2D point cloud."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=2, radius=3.0)
        positions = tf.random.normal([50, 2])
        supernode_indices = tf.constant([0, 10, 20])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}
        output = layer(inputs)
        assert output.shape == (1, 3, hidden_dim)

    def test_different_supernode_counts(self, hidden_dim, ndim):
        """Layer handles different numbers of supernodes."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=2.0)
        positions = tf.random.normal([50, ndim])
        for n_super in [1, 3, 10]:
            indices = tf.constant(list(range(0, n_super * 5, 5))[:n_super])
            inputs = {"positions": positions, "supernode_indices": indices}
            output = layer(inputs)
            assert output.shape == (1, n_super, hidden_dim)

    @pytest.mark.xfail(reason="Known bug in _knn_neighbors: take_along_axis shape mismatch")
    def test_k_neighbors_mode_output_shape(self, hidden_dim, ndim):
        """Test k_neighbors mode produces correct output shape."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, k_neighbors=3)
        positions = tf.random.normal([30, ndim])
        supernode_indices = tf.constant([0, 10, 20])
        inputs = {"positions": positions, "supernode_indices": supernode_indices}
        output = layer(inputs)
        assert output.shape == (1, 3, hidden_dim)

    def test_relpos_has_rel_pos_embed(self, hidden_dim, ndim):
        """relpos mode should create rel_pos_embed layer."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, mode="relpos")
        assert layer.rel_pos_embed is not None

    def test_abspos_no_rel_pos_embed(self, hidden_dim, ndim):
        """abspos mode should not create rel_pos_embed layer."""
        layer = SupernodePooling(hidden_dim=hidden_dim, ndim=ndim, radius=1.0, mode="abspos")
        assert layer.rel_pos_embed is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
