import pytest
import numpy as np
import tensorflow as tf
import keras
import os
import tempfile

from dl_techniques.layers.geometric.point_cloud_autoencoder import (
    PointCloudAutoencoder,
    CorrespondenceNetwork,
)


# ===========================================================================
# TestPointCloudAutoencoder
# ===========================================================================


class TestPointCloudAutoencoder:
    """Test suite for PointCloudAutoencoder.

    NOTE: PointCloudAutoencoder uses keras.ops.get_graph_feature() which does
    not exist in standard Keras. Therefore forward pass, model save/load, and
    gradient flow tests are NOT included. Only initialization, invalid params,
    serialization, and compute_output_shape are tested.
    """

    @pytest.fixture
    def k_neighbors(self) -> int:
        return 20

    # ------------------------------------------------------------------

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        layer = PointCloudAutoencoder()
        assert layer.k_neighbors == 20

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = PointCloudAutoencoder(
            k_neighbors=10,
            name="custom_pca",
        )
        assert layer.k_neighbors == 10
        assert layer.name == "custom_pca"

    def test_invalid_k_neighbors_zero(self):
        """Test that zero k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="k_neighbors"):
            PointCloudAutoencoder(k_neighbors=0)

    def test_invalid_k_neighbors_negative(self):
        """Test that negative k_neighbors raises ValueError."""
        with pytest.raises(ValueError, match="k_neighbors"):
            PointCloudAutoencoder(k_neighbors=-5)

    def test_compute_output_shape(self):
        """Test compute_output_shape returns correct structure."""
        layer = PointCloudAutoencoder(k_neighbors=10)
        source_shape = (8, 64, 3)
        target_shape = (8, 32, 3)
        result = layer.compute_output_shape((source_shape, target_shape))

        reconstructions, local_features, global_features = result

        # Reconstructions
        assert reconstructions == ((8, 64, 3), (8, 32, 3))
        # Local features
        assert local_features == ((8, 64, 1024), (8, 32, 1024))
        # Global features
        assert global_features == ((8, 2048), (8, 2048))

    def test_compute_output_shape_before_build(self):
        """Test compute_output_shape works before layer is built."""
        layer = PointCloudAutoencoder(k_neighbors=20)
        source_shape = (None, 128, 3)
        target_shape = (None, 128, 3)
        result = layer.compute_output_shape((source_shape, target_shape))

        reconstructions, local_features, global_features = result

        assert reconstructions == ((None, 128, 3), (None, 128, 3))
        assert local_features == ((None, 128, 1024), (None, 128, 1024))
        assert global_features == ((None, 2048), (None, 2048))

    def test_compute_output_shape_different_num_points(self):
        """Test compute_output_shape with different source/target sizes."""
        layer = PointCloudAutoencoder(k_neighbors=5)
        source_shape = (4, 100, 3)
        target_shape = (4, 200, 3)
        result = layer.compute_output_shape((source_shape, target_shape))

        reconstructions, local_features, global_features = result

        assert reconstructions[0] == (4, 100, 3)
        assert reconstructions[1] == (4, 200, 3)
        assert local_features[0] == (4, 100, 1024)
        assert local_features[1] == (4, 200, 1024)

    def test_serialization(self, k_neighbors):
        """get_config / from_config round-trip preserves attributes."""
        original = PointCloudAutoencoder(
            k_neighbors=k_neighbors,
            name="pca_s",
        )
        config = original.get_config()
        restored = PointCloudAutoencoder.from_config(config)

        assert restored.k_neighbors == original.k_neighbors

    def test_serialization_custom(self):
        """get_config / from_config round-trip with custom k_neighbors."""
        original = PointCloudAutoencoder(k_neighbors=15, name="pca_custom")
        config = original.get_config()

        assert config["k_neighbors"] == 15

        restored = PointCloudAutoencoder.from_config(config)
        assert restored.k_neighbors == 15


# ===========================================================================
# TestCorrespondenceNetwork
# ===========================================================================


class TestCorrespondenceNetwork:
    """Test suite for CorrespondenceNetwork."""

    @pytest.fixture
    def num_gaussians(self) -> int:
        return 8

    @pytest.fixture
    def num_points(self) -> int:
        return 16

    @pytest.fixture
    def local_dim(self) -> int:
        return 32

    @pytest.fixture
    def global_dim(self) -> int:
        return 64

    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def input_tensors(self, batch_size, num_points, local_dim, global_dim):
        local_features = tf.random.normal([batch_size, num_points, local_dim])
        global_features = tf.random.normal([batch_size, global_dim])
        return local_features, global_features

    @pytest.fixture
    def layer_instance(self, num_gaussians) -> CorrespondenceNetwork:
        return CorrespondenceNetwork(num_gaussians=num_gaussians)

    # ------------------------------------------------------------------

    def test_initialization_defaults(self, num_gaussians):
        """Test initialization with default parameters."""
        layer = CorrespondenceNetwork(num_gaussians=num_gaussians)
        assert layer.num_gaussians == num_gaussians

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        layer = CorrespondenceNetwork(
            num_gaussians=16,
            name="custom_corr",
        )
        assert layer.num_gaussians == 16
        assert layer.name == "custom_corr"

    def test_invalid_num_gaussians_zero(self):
        """Test that zero num_gaussians raises ValueError."""
        with pytest.raises(ValueError, match="num_gaussians"):
            CorrespondenceNetwork(num_gaussians=0)

    def test_invalid_num_gaussians_negative(self):
        """Test that negative num_gaussians raises ValueError."""
        with pytest.raises(ValueError, match="num_gaussians"):
            CorrespondenceNetwork(num_gaussians=-3)

    def test_build(self, layer_instance, input_tensors):
        """Test that the layer builds correctly."""
        layer_instance(input_tensors)
        assert layer_instance.built is True
        assert layer_instance.mlp.built is True

    def test_output_shape(self, layer_instance, input_tensors, batch_size, num_points, num_gaussians):
        """Test output shape matches expected dimensions."""
        output = layer_instance(input_tensors)
        assert output.shape == (batch_size, num_points, num_gaussians)

    def test_compute_output_shape(self, layer_instance, input_tensors, batch_size, num_points, num_gaussians, local_dim, global_dim):
        """Test compute_output_shape matches actual output."""
        layer_instance(input_tensors)
        local_shape = (batch_size, num_points, local_dim)
        global_shape = (batch_size, global_dim)
        computed = layer_instance.compute_output_shape((local_shape, global_shape))
        assert computed == (batch_size, num_points, num_gaussians)

    def test_compute_output_shape_before_build(self, num_gaussians):
        """Test compute_output_shape works before layer is built."""
        layer = CorrespondenceNetwork(num_gaussians=num_gaussians)
        local_shape = (None, 64, 128)
        global_shape = (None, 256)
        result = layer.compute_output_shape((local_shape, global_shape))
        assert result == (None, 64, num_gaussians)

    def test_output_is_probability_distribution(self, layer_instance, input_tensors):
        """Test that output rows sum to 1 (valid probability distribution)."""
        output = layer_instance(input_tensors)
        output_np = keras.ops.convert_to_numpy(output)

        # Each point's assignment over gaussians should sum to ~1
        row_sums = np.sum(output_np, axis=-1)
        np.testing.assert_allclose(
            row_sums,
            np.ones_like(row_sums),
            rtol=1e-5, atol=1e-5,
            err_msg="Softmax output rows should sum to 1",
        )

    def test_output_non_negative(self, layer_instance, input_tensors):
        """Test that all output values are non-negative (probabilities)."""
        output = layer_instance(input_tensors)
        output_np = keras.ops.convert_to_numpy(output)
        assert np.all(output_np >= 0), "Softmax outputs should be non-negative"

    def test_numerical_stability(self, num_gaussians):
        """No NaN / Inf with extreme input values."""
        layer = CorrespondenceNetwork(num_gaussians=num_gaussians)
        for scale in [1e-8, 1e8]:
            local_feat = tf.ones([2, 8, 16]) * scale
            global_feat = tf.ones([2, 32]) * scale
            out = layer((local_feat, global_feat))
            out_np = keras.ops.convert_to_numpy(out)
            assert not np.any(np.isnan(out_np)), f"NaN at scale {scale}"
            assert not np.any(np.isinf(out_np)), f"Inf at scale {scale}"

    def test_different_batch_sizes(self, num_gaussians):
        """Layer handles variable batch sizes."""
        layer = CorrespondenceNetwork(num_gaussians=num_gaussians)
        for bs in [1, 4, 8]:
            local_feat = tf.random.normal([bs, 16, 32])
            global_feat = tf.random.normal([bs, 64])
            out = layer((local_feat, global_feat))
            assert out.shape[0] == bs

    def test_serialization(self, num_gaussians):
        """get_config / from_config round-trip preserves attributes."""
        original = CorrespondenceNetwork(
            num_gaussians=num_gaussians,
            name="corr_s",
        )
        config = original.get_config()
        restored = CorrespondenceNetwork.from_config(config)

        assert restored.num_gaussians == original.num_gaussians

    def test_model_save_load(self, num_gaussians, num_points, local_dim, global_dim, batch_size):
        """Save / load through Keras .keras format preserves outputs."""
        local_feat = tf.random.normal([batch_size, num_points, local_dim])
        global_feat = tf.random.normal([batch_size, global_dim])

        inp_local = keras.Input(shape=(num_points, local_dim))
        inp_global = keras.Input(shape=(global_dim,))
        out = CorrespondenceNetwork(num_gaussians=num_gaussians, name="corr")(
            [inp_local, inp_global]
        )
        model = keras.Model(inputs=[inp_local, inp_global], outputs=out)

        original_pred = model.predict([local_feat, global_feat], verbose=0)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)

        loaded_pred = loaded.predict([local_feat, global_feat], verbose=0)
        np.testing.assert_allclose(
            original_pred, loaded_pred, rtol=1e-5, atol=1e-5,
            err_msg="Predictions should match after save/load",
        )

    def test_gradient_flow(self, num_gaussians):
        """Gradients propagate through the layer."""
        layer = CorrespondenceNetwork(num_gaussians=num_gaussians)
        local_feat = tf.Variable(tf.random.normal([2, 8, 16]))
        global_feat = tf.Variable(tf.random.normal([2, 32]))
        with tf.GradientTape() as tape:
            out = layer((local_feat, global_feat))
            loss = tf.reduce_mean(tf.square(out))
        grads_local = tape.gradient(loss, local_feat)
        assert grads_local is not None
        assert np.any(grads_local.numpy() != 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
