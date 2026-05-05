import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.embedding.albert_factorized_embedding import (
    AlbertFactorizedEmbedding,
)


class TestAlbertFactorizedEmbedding:
    """Test suite for AlbertFactorizedEmbedding."""

    @pytest.fixture
    def small_layer(self) -> AlbertFactorizedEmbedding:
        return AlbertFactorizedEmbedding(
            vocab_size=64, bottleneck_dim=8, output_dim=32,
        )

    def test_forward_shape(self, small_layer):
        ids = tf.constant([[0, 1, 5, 63]], dtype=tf.int32)
        out = small_layer(ids)
        assert out.shape == (1, 4, 32)

    def test_param_count_compression_at_realistic_scale(self):
        """For vocab=50K, D=768, k=128: ~6x smaller than dense Embedding."""
        layer = AlbertFactorizedEmbedding(
            vocab_size=50_000, bottleneck_dim=128, output_dim=768,
        )
        ids = tf.zeros((1, 4), dtype=tf.int32)
        _ = layer(ids)  # build
        n_params = layer.count_params()
        n_dense = 50_000 * 768
        assert n_dense // n_params >= 5, (
            f"Expected >=5x compression, got {n_dense / n_params:.2f}x"
        )

    @pytest.mark.parametrize("vocab,k,D", [
        (1, 8, 16), (0, 8, 16), (-1, 8, 16),
        (64, 0, 16), (64, -1, 16), (64, 8, 0), (64, 8, -1),
    ])
    def test_invalid_args_raise(self, vocab, k, D):
        with pytest.raises(ValueError):
            AlbertFactorizedEmbedding(
                vocab_size=vocab, bottleneck_dim=k, output_dim=D,
            )

    def test_distinct_tokens_distinct_embeddings(self, small_layer):
        small_layer.build((None,))
        # Randomize weights so distinct tokens give distinct embeddings.
        for w in small_layer.weights:
            w.assign(tf.random.normal(w.shape, seed=7))
        ids = tf.constant([0, 1, 2, 63], dtype=tf.int32)
        out = small_layer(ids).numpy()
        assert not np.allclose(out[0], out[1])
        assert not np.allclose(out[2], out[3])

    def test_serialization_roundtrip(self):
        layer = AlbertFactorizedEmbedding(
            vocab_size=128, bottleneck_dim=16, output_dim=64,
        )
        ids = tf.constant([[0, 1, 127]], dtype=tf.int32)
        original = layer(ids).numpy()

        config = layer.get_config()
        reconstructed = AlbertFactorizedEmbedding.from_config(config)
        reconstructed(ids)
        for w_orig, w_new in zip(layer.weights, reconstructed.weights):
            w_new.assign(w_orig.numpy())
        new_out = reconstructed(ids).numpy()
        np.testing.assert_allclose(original, new_out, atol=1e-6)

    def test_save_load_keras(self):
        inp = keras.Input(shape=(8,), dtype="int32")
        x = AlbertFactorizedEmbedding(
            vocab_size=256, bottleneck_dim=16, output_dim=32,
        )(inp)
        out = keras.layers.GlobalAveragePooling1D()(x)
        model = keras.Model(inp, out)

        ids = np.random.randint(0, 256, size=(4, 8)).astype("int32")
        original = model(ids).numpy()

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "albert.keras")
            model.save(path)
            loaded = keras.models.load_model(
                path,
                custom_objects={
                    "AlbertFactorizedEmbedding": AlbertFactorizedEmbedding,
                },
            )
        loaded_out = loaded(ids).numpy()
        np.testing.assert_allclose(original, loaded_out, atol=1e-6)

    def test_gradient_flow(self):
        layer = AlbertFactorizedEmbedding(
            vocab_size=64, bottleneck_dim=8, output_dim=16,
        )
        ids = tf.constant([[0, 1, 2, 3]], dtype=tf.int32)
        with tf.GradientTape() as tape:
            out = layer(ids)
            loss = tf.reduce_sum(out)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) >= 2  # inner embedding + projection kernel
        for g in grads:
            assert g is not None
            assert tf.reduce_any(tf.not_equal(g, 0.0))

    def test_compute_output_shape(self, small_layer):
        assert small_layer.compute_output_shape((None, 32)) == (None, 32, 32)
        assert small_layer.compute_output_shape((4, 8)) == (4, 8, 32)

    def test_regularizer_serialization(self):
        layer = AlbertFactorizedEmbedding(
            vocab_size=64, bottleneck_dim=8, output_dim=16,
            embeddings_regularizer=keras.regularizers.L2(1e-4),
            projection_regularizer=keras.regularizers.L2(1e-5),
        )
        config = layer.get_config()
        rebuilt = AlbertFactorizedEmbedding.from_config(config)
        assert isinstance(
            rebuilt.embeddings_regularizer, keras.regularizers.L2,
        )
        assert isinstance(
            rebuilt.projection_regularizer, keras.regularizers.L2,
        )
