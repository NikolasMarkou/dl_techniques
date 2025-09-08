import pytest
import keras
import numpy as np
import tensorflow as tf
import tempfile
import os
from keras import ops
from typing import Dict, Any

# Assuming the file is named `code.py` for this example
from dl_techniques.models.modern_bert_blt.components import (
    ByteTokenizer,
    HashNGramEmbedding,
    ModernBertBltEmbeddings,
)

# ---------------------------------------------------------------------
# Tests for Component 1: ByteTokenizer
# ---------------------------------------------------------------------
class TestByteTokenizer:
    def test_initialization(self):
        """Test layer initialization with default and custom values."""
        # Default
        tokenizer = ByteTokenizer()
        assert tokenizer.vocab_size == 260
        assert tokenizer.byte_offset == 4
        assert not tokenizer.trainable

        # Custom
        tokenizer = ByteTokenizer(vocab_size=300, byte_offset=10)
        assert tokenizer.vocab_size == 300
        assert tokenizer.byte_offset == 10

    def test_config_completeness(self):
        """Test that get_config contains all __init__ parameters."""
        config_params = {"vocab_size": 300, "byte_offset": 10}
        tokenizer = ByteTokenizer(**config_params)
        config = tokenizer.get_config()
        for key, value in config_params.items():
            assert key in config
            assert config[key] == value

    @pytest.mark.parametrize(
        "text, add_bos, add_eos, expected_tokens",
        [
            # FIX: Use correct raw byte value for 'i' (105).
            ("Hi", False, False, [72, 105]),  # 'H' is 72, 'i' is 105.
            ("Hi", True, False, [1, 72, 105]),
            ("Hi", False, True, [72, 105, 2]),
            ("Hi", True, True, [1, 72, 105, 2]),
            ("", True, True, [1, 2]),
            # Multi-byte char '€' -> [226, 130, 172]
            ("€", False, False, [226, 130, 172]),
        ],
    )
    def test_text_to_bytes(self, text, add_bos, add_eos, expected_tokens):
        """Test encoding of text to byte token IDs."""
        tokenizer = ByteTokenizer(byte_offset=4)
        # Apply offset to expected tokens
        expected = [t if t < 4 else t + 4 for t in expected_tokens]
        tokens = tokenizer.text_to_bytes(text, add_bos=add_bos, add_eos=add_eos)
        assert tokens == expected

    @pytest.mark.parametrize(
        "tokens, expected_text",
        [
            # FIX: Use correct raw byte value for 'i' (105) to match expected "Hi World".
            ([72, 105, 32, 87, 111, 114, 108, 100], "Hi World"),
            ([1, 72, 105, 2], "Hi"),  # With special tokens
            ([226, 130, 172], "€"),  # Multi-byte
            ([0, 0, 0], ""),  # Only special tokens
            ([], ""),  # Empty list
        ]
    )
    def test_tokens_to_text(self, tokens, expected_text):
        """Test decoding of byte token IDs to text."""
        tokenizer = ByteTokenizer(byte_offset=4)
        # Apply offset to input tokens to simulate real input to the method
        offset_tokens = [t if t < 4 else t + 4 for t in tokens]
        text = tokenizer.tokens_to_text(offset_tokens)
        assert text == expected_text

    def test_round_trip_conversion(self):
        """Test that encoding and then decoding returns the original text."""
        tokenizer = ByteTokenizer()
        original_text = "Hello, world! This is a test with € and other symbols."
        tokens = tokenizer.text_to_bytes(original_text, add_bos=True, add_eos=True)
        decoded_text = tokenizer.tokens_to_text(tokens)
        assert decoded_text == original_text

    def test_serialization(self):
        """Test saving and loading the stateless layer."""
        layer = ByteTokenizer(vocab_size=300, byte_offset=10)
        config = layer.get_config()
        reloaded_layer = ByteTokenizer.from_config(config)

        assert reloaded_layer.vocab_size == 300
        assert reloaded_layer.byte_offset == 10

        # Verify functionality is the same
        text = "test"
        original_tokens = layer.text_to_bytes(text, False, False)
        reloaded_tokens = reloaded_layer.text_to_bytes(text, False, False)
        assert original_tokens == reloaded_tokens


# ---------------------------------------------------------------------
# Tests for Component 2: HashNGramEmbedding
# ---------------------------------------------------------------------
class TestHashNGramEmbedding:
    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        return {
            "hash_vocab_size": 100,
            "embed_dim": 16,
            "ngram_sizes": [2, 3],
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        return ops.array([[4, 5, 6, 7]], dtype="int32")

    def test_initialization(self, layer_config):
        """Test layer initialization and sub-layer creation."""
        layer = HashNGramEmbedding(**layer_config)
        assert layer.hash_vocab_size == 100
        assert layer.embed_dim == 16
        assert layer.ngram_sizes == [2, 3]
        assert not layer.built
        assert "2" in layer.hash_embeddings
        assert "3" in layer.hash_embeddings
        assert isinstance(layer.hash_embeddings["2"], keras.layers.Embedding)

    def test_initialization_errors(self):
        """Test error conditions during initialization."""
        with pytest.raises(ValueError):
            HashNGramEmbedding(hash_vocab_size=0, embed_dim=16, ngram_sizes=[2])
        with pytest.raises(ValueError):
            HashNGramEmbedding(hash_vocab_size=10, embed_dim=16, ngram_sizes=[])
        with pytest.raises(ValueError):
            HashNGramEmbedding(hash_vocab_size=10, embed_dim=16, ngram_sizes=[2, 0])

    def test_forward_pass_and_shape(self, layer_config, sample_input):
        """Test the forward pass and verify the output shape."""
        layer = HashNGramEmbedding(**layer_config)
        output = layer(sample_input)

        assert layer.built
        expected_shape = (
            sample_input.shape[0],
            sample_input.shape[1],
            layer_config["embed_dim"],
        )
        assert output.shape == expected_shape

    def test_hashing_logic(self):
        """Verify the polynomial hashing calculation is correct."""
        inputs = ops.array([[10, 20]], dtype="int64")
        hash_vocab_size = 1000
        base = 257

        # Manual calculation for n=2
        # Padded input: [0, 10, 20]
        # 1st ngram: [0, 10] -> hash = (0*257 + 10) % 1000 = 10
        # 2nd ngram: [10, 20] -> hash = (10*257 + 20) % 1000 = 2590 % 1000 = 590
        expected_hashes_n2 = np.array([[10, 590]])

        layer = HashNGramEmbedding(hash_vocab_size, 16, [2])

        # We can't directly see the indices, but we can verify by setting weights
        embedding_layer = layer.hash_embeddings['2']
        embedding_layer.build((None, 2))

        # Create weights that map index to itself
        weights = np.arange(hash_vocab_size).reshape(-1, 1) * np.ones((1, 16))
        embedding_layer.set_weights([weights])

        output_n2 = layer._compute_ngram_embeddings(inputs, 2)

        # Check if the output embedding vector corresponds to the hashed index
        np.testing.assert_allclose(output_n2[:, 0, 0], expected_hashes_n2[0, 0])
        np.testing.assert_allclose(output_n2[:, 1, 0], expected_hashes_n2[0, 1])

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Full save and load cycle."""
        # 1. Create original layer in a model
        inputs = keras.Input(shape=sample_input.shape[1:], dtype="int32")
        layer_output = HashNGramEmbedding(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)

        # 2. Get prediction from original
        original_prediction = model(sample_input)

        # 3. Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # 4. Verify identical outputs
            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
                err_msg="Predictions differ after serialization",
            )
            # Verify sub-layers were rebuilt
            assert loaded_model.layers[1].built
            assert loaded_model.layers[1].hash_embeddings['2'].built
            assert loaded_model.layers[1].hash_embeddings['3'].built

    def test_gradients_flow(self, layer_config, sample_input):
        """Test that gradients are computed for trainable variables."""
        layer = HashNGramEmbedding(**layer_config)
        with tf.GradientTape() as tape:
            output = layer(sample_input, training=True)
            loss = ops.mean(output ** 2)

        gradients = tape.gradient(loss, layer.trainable_variables)
        assert len(gradients) == len(layer_config["ngram_sizes"])  # One for each emb table
        assert all(g is not None for g in gradients)


# ---------------------------------------------------------------------
# Tests for Component 3: ModernBertBltEmbeddings
# ---------------------------------------------------------------------
class TestModernBertBltEmbeddings:
    @pytest.fixture
    def base_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 260,
            "hidden_size": 32,
            "max_position_embeddings": 64,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": 0.1,
            "normalization_type": "layer_norm"
        }

    @pytest.fixture
    def hash_config(self) -> Dict[str, Any]:
        return {
            "use_hash_embeddings": True,
            "hash_vocab_size": 100,
            "ngram_sizes": [2],
            "hash_embedding_dim": 16
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        return keras.random.randint(minval=4, maxval=259, shape=(2, 10))

    def test_initialization_no_hash(self, base_config):
        """Test initialization without hash embeddings."""
        layer = ModernBertBltEmbeddings(use_hash_embeddings=False, **base_config)
        assert not layer.use_hash_embeddings
        assert layer.hash_embeddings is None
        assert layer.hash_projection is None
        assert isinstance(layer.byte_embeddings, keras.layers.Embedding)
        assert isinstance(layer.layer_norm, keras.layers.LayerNormalization)

    def test_initialization_with_hash(self, base_config, hash_config):
        """Test initialization with hash embeddings."""
        layer = ModernBertBltEmbeddings(**base_config, **hash_config)
        assert layer.use_hash_embeddings
        assert isinstance(layer.hash_embeddings, HashNGramEmbedding)
        assert isinstance(layer.hash_projection, keras.layers.Dense)

    def test_initialization_hash_error(self, base_config):
        """Test ValueError when hash params are missing."""
        with pytest.raises(ValueError, match="must be provided"):
            ModernBertBltEmbeddings(use_hash_embeddings=True, **base_config)

    def test_forward_pass_no_hash(self, base_config, sample_input):
        """Test forward pass and shape without hash embeddings."""
        layer = ModernBertBltEmbeddings(use_hash_embeddings=False, **base_config)
        output = layer(sample_input)
        assert layer.built
        expected_shape = (
            sample_input.shape[0],
            sample_input.shape[1],
            base_config["hidden_size"],
        )
        assert output.shape == expected_shape

    def test_forward_pass_with_hash(self, base_config, hash_config, sample_input):
        """Test forward pass and shape with hash embeddings."""
        config = {**base_config, **hash_config}
        layer = ModernBertBltEmbeddings(**config)
        output = layer(sample_input)
        assert layer.built
        expected_shape = (
            sample_input.shape[0],
            sample_input.shape[1],
            config["hidden_size"],
        )
        assert output.shape == expected_shape

    @pytest.mark.parametrize("use_hash", [True, False])
    def test_serialization_cycle(self, base_config, hash_config, sample_input, use_hash):
        """CRITICAL TEST: Full save/load cycle for both configurations."""
        config = {**base_config, "use_hash_embeddings": use_hash}
        if use_hash:
            config.update(hash_config)

        inputs = keras.Input(shape=sample_input.shape[1:], dtype="int32")
        layer_output = ModernBertBltEmbeddings(**config)(inputs)
        model = keras.Model(inputs, layer_output)

        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "bert_blt_model.keras")
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            # Verify sub-layers were built correctly
            loaded_layer = loaded_model.layers[1]
            assert loaded_layer.built
            assert loaded_layer.byte_embeddings.built
            assert loaded_layer.position_embeddings.built
            if use_hash:
                assert loaded_layer.hash_embeddings.built
                assert loaded_layer.hash_projection.built

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6,
                atol=1e-6,
            )

    def test_gradients_flow(self, base_config, hash_config, sample_input):
        """Test gradient flow through all components."""
        config = {**base_config, **hash_config}
        layer = ModernBertBltEmbeddings(**config)

        with tf.GradientTape() as tape:
            output = layer(sample_input, training=True)
            loss = ops.mean(output ** 2)

        gradients = tape.gradient(loss, layer.trainable_variables)

        # Expected variables:
        # 1 (byte_emb) + 1 (pos_emb) = 2
        # 1 (hash_emb for n=2)
        # 2 (hash_proj kernel + bias)
        # 2 (layernorm gamma + beta)
        # Total = 7
        num_expected_vars = 2 + len(hash_config["ngram_sizes"]) + 2 + 2
        assert len(layer.trainable_variables) == num_expected_vars
        assert len(gradients) == num_expected_vars
        assert all(g is not None for g in gradients)

    def test_convenience_methods(self, base_config):
        """Test encode_text and decode_tokens methods."""
        layer = ModernBertBltEmbeddings(use_hash_embeddings=False, **base_config)
        text = "Test 123"

        # Test encode
        encoded = layer.encode_text(text, max_length=15, add_special_tokens=True)
        assert encoded.shape == (1, 15)
        # BOS + "Test 123" + EOS = 1 + 8 + 1 = 10 tokens. 5 padding tokens.
        assert ops.sum(ops.cast(encoded != 0, "int32")) == 10

        # Test decode
        decoded_text = layer.decode_tokens(encoded[0])
        assert decoded_text == text

    @pytest.mark.parametrize("training", [True, False])
    def test_training_modes(self, base_config, sample_input, training):
        """Test behavior in different training modes (for dropout)."""
        # Set dropout high to make differences more likely
        config = {**base_config, "hidden_dropout_prob": 0.9}
        layer = ModernBertBltEmbeddings(use_hash_embeddings=False, **config)

        # Pass data through once to build the layer
        _ = layer(sample_input)

        # Get two outputs in training mode
        output1 = layer(sample_input, training=True)
        output2 = layer(sample_input, training=True)

        # Get two outputs in inference mode
        output_eval1 = layer(sample_input, training=False)
        output_eval2 = layer(sample_input, training=False)

        # In training mode, outputs should differ due to dropout
        assert not np.allclose(
            ops.convert_to_numpy(output1), ops.convert_to_numpy(output2)
        )

        # In inference mode, outputs should be identical
        np.testing.assert_allclose(
            ops.convert_to_numpy(output_eval1), ops.convert_to_numpy(output_eval2)
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])