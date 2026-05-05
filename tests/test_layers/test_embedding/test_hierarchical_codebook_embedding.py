import pytest
import numpy as np
import tensorflow as tf
import keras
import tempfile
import os

from dl_techniques.layers.embedding.hierarchical_codebook_embedding import (
    HierarchicalCodebookEmbedding,
)


class TestHierarchicalCodebookEmbedding:
    """Test suite for HierarchicalCodebookEmbedding."""

    @pytest.fixture
    def small_layer(self) -> HierarchicalCodebookEmbedding:
        # vocab=64 -> 6 bits; K=2, chunk_bits=3, M=8 -> exactly addresses 64.
        return HierarchicalCodebookEmbedding(
            vocab_size=64, output_dim=16, num_chunks=2, chunk_bits=3,
            use_layer_norm=False,
        )

    @pytest.fixture
    def realistic_layer(self) -> HierarchicalCodebookEmbedding:
        # 50K-style vocab, K=2 with auto chunk_bits.
        return HierarchicalCodebookEmbedding(
            vocab_size=50_000, output_dim=128, num_chunks=2,
        )

    def test_param_count_compression(self, realistic_layer):
        """K=2 codebooks for 50K vocab should be ~100x smaller than dense."""
        ids = tf.zeros((1, 4), dtype=tf.int32)
        _ = realistic_layer(ids)  # build
        n_params = realistic_layer.count_params()
        n_dense = 50_000 * 128
        assert n_params < n_dense // 50, (
            f"Expected >50x compression, got {n_dense / n_params:.1f}x "
            f"({n_params:,} vs {n_dense:,})"
        )

    def test_chunk_bits_auto(self):
        """Auto chunk_bits rounds up to cover vocab_size bits."""
        # vocab=50261 -> 16 bits. K=4 -> chunk_bits=4. K=3 -> chunk_bits=6.
        layer4 = HierarchicalCodebookEmbedding(50_261, 16, num_chunks=4)
        assert layer4.chunk_bits == 4
        layer3 = HierarchicalCodebookEmbedding(50_261, 16, num_chunks=3)
        assert layer3.chunk_bits == 6  # 16/3 -> ceil(5.33) = 6

    def test_invalid_chunk_capacity(self):
        """num_chunks * chunk_bits < bits-needed must raise."""
        with pytest.raises(ValueError, match="cannot address"):
            HierarchicalCodebookEmbedding(
                vocab_size=1024, output_dim=8,
                num_chunks=2, chunk_bits=4,  # 8 bits < 10 needed
            )

    @pytest.mark.parametrize("vocab,K", [(1, 2), (0, 2), (-1, 2)])
    def test_invalid_vocab(self, vocab, K):
        with pytest.raises(ValueError):
            HierarchicalCodebookEmbedding(
                vocab_size=vocab, output_dim=8, num_chunks=K,
            )

    def test_forward_shape(self, small_layer):
        ids = tf.constant([[0, 1, 7, 8, 63]], dtype=tf.int32)
        out = small_layer(ids)
        assert out.shape == (1, 5, 16)

    def test_distinct_tokens_distinct_embeddings(self, small_layer):
        """Tokens differing in EVERY chunk should produce different embeds."""
        # Build with non-zero codebooks
        small_layer.build((None,))
        for cb in small_layer.codebooks:
            cb.assign(tf.random.normal(cb.shape, seed=42))
        # Token 0 = chunks (0, 0). Token 9 = chunks (1, 1) at chunk_bits=3.
        ids = tf.constant([0, 9], dtype=tf.int32)
        out = small_layer(ids).numpy()
        assert not np.allclose(out[0], out[1])

    def test_sibling_tokens_share_chunks(self, small_layer):
        """Tokens differing only in low-bit chunk: high-bit codebook
        contribution is identical."""
        small_layer.build((None,))
        for cb in small_layer.codebooks:
            cb.assign(tf.random.normal(cb.shape, seed=0))
        # Token 0 = (chunk_0=0, chunk_1=0); Token 1 = (chunk_0=1, chunk_1=0).
        # They share the high-bit codebook lookup E_1[0].
        ids = tf.constant([0, 1, 8, 9], dtype=tf.int32)
        out = small_layer(ids).numpy()
        # Difference 0 -> 1 should equal difference 8 -> 9 (both flip
        # chunk_0 from 0 to 1, with shared chunk_1 contributions cancelling).
        diff_a = out[1] - out[0]
        diff_b = out[3] - out[2]
        np.testing.assert_allclose(diff_a, diff_b, atol=1e-6)

    def test_serialization_roundtrip(self):
        layer = HierarchicalCodebookEmbedding(
            vocab_size=128, output_dim=16, num_chunks=2, chunk_bits=4,
            use_layer_norm=True,
        )
        ids = tf.constant([[0, 1, 127]], dtype=tf.int32)
        original_out = layer(ids).numpy()

        config = layer.get_config()
        reconstructed = HierarchicalCodebookEmbedding.from_config(config)
        # Build with same shape and copy weights
        reconstructed(ids)
        for w_orig, w_new in zip(layer.weights, reconstructed.weights):
            w_new.assign(w_orig.numpy())
        new_out = reconstructed(ids).numpy()
        np.testing.assert_allclose(original_out, new_out, atol=1e-6)

    def test_save_load_keras(self):
        """Full model save/load via .keras format."""
        inp = keras.Input(shape=(8,), dtype="int32")
        x = HierarchicalCodebookEmbedding(
            vocab_size=256, output_dim=32, num_chunks=2, chunk_bits=4,
        )(inp)
        out = keras.layers.GlobalAveragePooling1D()(x)
        model = keras.Model(inp, out)

        ids = np.random.randint(0, 256, size=(4, 8)).astype("int32")
        original = model(ids).numpy()

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "hce.keras")
            model.save(path)
            loaded = keras.models.load_model(
                path,
                custom_objects={
                    "HierarchicalCodebookEmbedding":
                        HierarchicalCodebookEmbedding,
                },
            )
        loaded_out = loaded(ids).numpy()
        np.testing.assert_allclose(original, loaded_out, atol=1e-6)

    def test_gradient_flow(self):
        layer = HierarchicalCodebookEmbedding(
            vocab_size=64, output_dim=16, num_chunks=2, chunk_bits=3,
            use_layer_norm=False,
        )
        ids = tf.constant([[0, 1, 2, 3]], dtype=tf.int32)
        with tf.GradientTape() as tape:
            out = layer(ids)
            loss = tf.reduce_sum(out)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) == 2  # 2 codebooks
        for g in grads:
            assert g is not None
            assert tf.reduce_any(tf.not_equal(g, 0.0))

    def test_compute_output_shape(self, small_layer):
        assert small_layer.compute_output_shape((None, 32)) == (None, 32, 16)
        assert small_layer.compute_output_shape((4, 8)) == (4, 8, 16)

    def test_layer_norm_applied(self):
        layer = HierarchicalCodebookEmbedding(
            vocab_size=64, output_dim=128, num_chunks=2, chunk_bits=3,
            use_layer_norm=True,
        )
        ids = tf.random.uniform((8, 16), 0, 64, dtype=tf.int32)
        out = layer(ids).numpy()
        # LayerNorm gives ~unit variance per token.
        per_token_std = out.std(axis=-1)
        assert per_token_std.mean() > 0.5
        assert per_token_std.mean() < 1.5

    def test_no_layer_norm(self):
        layer = HierarchicalCodebookEmbedding(
            vocab_size=64, output_dim=16, num_chunks=2, chunk_bits=3,
            use_layer_norm=False,
        )
        ids = tf.constant([[0, 1, 2]], dtype=tf.int32)
        out = layer(ids)
        assert out.shape == (1, 3, 16)
