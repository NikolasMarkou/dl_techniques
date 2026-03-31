import pytest
import tempfile
import os
import numpy as np
import keras
from keras import ops
from typing import Any, Dict

from dl_techniques.layers.reasoning.hrm_sparse_puzzle_embedding import SparsePuzzleEmbedding


class TestSparsePuzzleEmbedding:
    """Comprehensive test suite for SparsePuzzleEmbedding."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Standard configuration for testing."""
        return {
            'num_embeddings': 100,
            'embedding_dim': 32,
            'batch_size': 4,
        }

    @pytest.fixture
    def minimal_config(self) -> Dict[str, Any]:
        """Minimal configuration for testing."""
        return {
            'num_embeddings': 10,
            'embedding_dim': 8,
            'batch_size': 2,
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Sample puzzle IDs input."""
        return ops.convert_to_tensor(np.array([0, 5, 3, 9], dtype=np.int32))

    def test_initialization(self, layer_config):
        """Test layer initialization."""
        layer = SparsePuzzleEmbedding(**layer_config)

        assert layer.num_embeddings == 100
        assert layer.embedding_dim == 32
        assert layer.batch_size == 4
        assert not layer.built

    def test_forward_pass_inference(self, layer_config, sample_input):
        """Test forward pass in inference mode."""
        layer = SparsePuzzleEmbedding(**layer_config)

        output = layer(sample_input, training=False)

        assert layer.built
        assert output.shape == (4, 32)

    def test_forward_pass_training(self, layer_config, sample_input):
        """Test forward pass in training mode with caching."""
        layer = SparsePuzzleEmbedding(**layer_config)

        output = layer(sample_input, training=True)

        assert output.shape == (4, 32)
        # Verify local caches were updated
        cached_ids = ops.convert_to_numpy(layer.local_ids)
        np.testing.assert_array_equal(cached_ids, [0, 5, 3, 9])

    def test_embedding_lookup_correctness(self, minimal_config):
        """Test that embedding lookup returns correct vectors."""
        layer = SparsePuzzleEmbedding(**minimal_config)

        ids = ops.convert_to_tensor(np.array([0, 1], dtype=np.int32))
        output = layer(ids, training=False)

        # Manual lookup should match
        expected = ops.take(layer.embeddings, ids, axis=0)
        np.testing.assert_allclose(
            ops.convert_to_numpy(output),
            ops.convert_to_numpy(expected),
            atol=1e-7
        )

    def test_different_initializers(self, sample_input):
        """Test with different embedding initializers."""
        for init in ['zeros', 'ones', 'random_normal']:
            layer = SparsePuzzleEmbedding(
                num_embeddings=100,
                embedding_dim=16,
                batch_size=4,
                embeddings_initializer=init,
            )
            output = layer(sample_input, training=False)
            assert output.shape == (4, 16)

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full serialization cycle."""
        inputs = keras.Input(shape=(), dtype='int32')
        outputs = SparsePuzzleEmbedding(**layer_config)(inputs, training=False)
        model = keras.Model(inputs, outputs)

        original_pred = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_sparse_puzzle.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_input)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_pred),
                ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6,
                err_msg="SparsePuzzleEmbedding predictions differ after serialization"
            )

    def test_config_completeness(self, layer_config):
        """Test that get_config contains all __init__ parameters."""
        layer = SparsePuzzleEmbedding(**layer_config)
        config = layer.get_config()

        expected_keys = {
            'num_embeddings', 'embedding_dim', 'batch_size',
            'embeddings_initializer', 'embeddings_regularizer',
        }

        for key in expected_keys:
            assert key in config, f"Missing {key} in get_config()"

        assert config['num_embeddings'] == layer_config['num_embeddings']
        assert config['embedding_dim'] == layer_config['embedding_dim']
        assert config['batch_size'] == layer_config['batch_size']

    def test_config_roundtrip(self, layer_config):
        """Test config-based reconstruction."""
        layer = SparsePuzzleEmbedding(**layer_config)
        config = layer.get_config()

        reconstructed = SparsePuzzleEmbedding.from_config(config)

        assert reconstructed.num_embeddings == layer.num_embeddings
        assert reconstructed.embedding_dim == layer.embedding_dim
        assert reconstructed.batch_size == layer.batch_size

    def test_gradients_flow(self, layer_config, sample_input):
        """Test gradient computation."""
        import tensorflow as tf

        layer = SparsePuzzleEmbedding(**layer_config)

        with tf.GradientTape() as tape:
            output = layer(sample_input, training=True)
            loss = ops.mean(ops.square(output))

        gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(gradients) > 0
        assert any(g is not None for g in gradients)

    def test_compute_output_shape(self, layer_config):
        """Test output shape computation."""
        layer = SparsePuzzleEmbedding(**layer_config)

        input_shape = (4,)
        output_shape = layer.compute_output_shape(input_shape)
        assert output_shape == (4, 32)

    def test_edge_cases(self):
        """Test error conditions."""
        with pytest.raises(ValueError, match="num_embeddings must be positive"):
            SparsePuzzleEmbedding(num_embeddings=0, embedding_dim=8, batch_size=2)

        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            SparsePuzzleEmbedding(num_embeddings=10, embedding_dim=0, batch_size=2)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            SparsePuzzleEmbedding(num_embeddings=10, embedding_dim=8, batch_size=0)

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, minimal_config, training):
        """Test behavior in different training modes."""
        layer = SparsePuzzleEmbedding(**minimal_config)
        ids = ops.convert_to_tensor(np.array([0, 1], dtype=np.int32))

        output = layer(ids, training=training)
        assert output.shape == (2, 8)
