"""Tests for the GPT-2 model implementation.

Covers initialization, parameter validation, forward pass, weight tying,
causal masking, serialization, variant creation, and gradient flow.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.gpt2.gpt2 import GPT2


def _random_ids(shape, vocab_size):
    """Generate random integer token IDs."""
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Minimal GPT-2 config for fast testing."""
    return {
        "vocab_size": 256,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "max_seq_len": 32,
        "dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
    }


@pytest.fixture
def tiny_model(tiny_config):
    """Pre-built tiny GPT-2 model."""
    model = GPT2(**tiny_config)
    dummy = _random_ids((1, tiny_config["max_seq_len"]), tiny_config["vocab_size"])
    model(dummy, training=False)
    return model


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------


class TestGPT2Initialization:
    """Test GPT-2 model initialization and parameter validation."""

    def test_basic_initialization(self, tiny_config):
        model = GPT2(**tiny_config)
        assert model.vocab_size == 256
        assert model.embed_dim == 64
        assert model.depth == 2
        assert model.num_heads == 4
        assert model.max_seq_len == 32
        assert model.decoder is not None

    def test_default_initialization(self):
        model = GPT2()
        assert model.vocab_size == 100277
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12
        assert model.max_seq_len == 1024

    def test_parameter_validation_divisibility(self):
        with pytest.raises(ValueError, match="embed_dim.*must be divisible by num_heads"):
            GPT2(vocab_size=256, embed_dim=100, num_heads=12, depth=2)

    def test_parameter_validation_negative_vocab(self):
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            GPT2(vocab_size=-1, embed_dim=64, depth=2, num_heads=4)

    def test_parameter_validation_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate must be between"):
            GPT2(vocab_size=256, embed_dim=64, depth=2, num_heads=4, dropout_rate=1.5)

    def test_parameter_validation_attention_dropout(self):
        with pytest.raises(ValueError, match="attention_dropout_rate must be between"):
            GPT2(vocab_size=256, embed_dim=64, depth=2, num_heads=4, attention_dropout_rate=-0.1)

    def test_pre_norm_configuration(self, tiny_model):
        """Verify the decoder uses pre-layer normalization."""
        assert tiny_model.decoder.normalization_position == "pre"


# ---------------------------------------------------------------------
# Forward Pass Tests
# ---------------------------------------------------------------------


class TestGPT2ForwardPass:
    """Test GPT-2 forward pass and output shapes."""

    def test_forward_pass_tensor_input(self, tiny_config):
        model = GPT2(**tiny_config)
        batch_size, seq_len = 2, 16
        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)

        assert "logits" in outputs
        assert "last_hidden_state" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, tiny_config["vocab_size"])
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, tiny_config["embed_dim"])

    def test_forward_pass_dict_input(self, tiny_config):
        model = GPT2(**tiny_config)
        batch_size, seq_len = 2, 16
        inputs = {
            "input_ids": _random_ids((batch_size, seq_len), tiny_config["vocab_size"]),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int32),
        }
        outputs = model(inputs, training=False)

        assert outputs["logits"].shape == (batch_size, seq_len, tiny_config["vocab_size"])

    def test_forward_pass_dict_missing_input_ids(self, tiny_config):
        model = GPT2(**tiny_config)
        with pytest.raises(ValueError, match="input_ids"):
            model({"attention_mask": np.ones((2, 16), dtype=np.int32)})

    def test_forward_pass_batch_size_one(self, tiny_config):
        model = GPT2(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, 8, tiny_config["vocab_size"])

    def test_forward_pass_full_sequence(self, tiny_config):
        model = GPT2(**tiny_config)
        seq_len = tiny_config["max_seq_len"]
        input_ids = _random_ids((1, seq_len), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)
        assert outputs["logits"].shape == (1, seq_len, tiny_config["vocab_size"])


# ---------------------------------------------------------------------
# Weight Tying Tests
# ---------------------------------------------------------------------


class TestGPT2WeightTying:
    """Test that LM head reuses token embedding weights."""

    def test_weight_tying(self, tiny_config):
        model = GPT2(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        model(input_ids, training=False)

        embedding_weights = model.decoder.word_embeddings.embeddings
        assert embedding_weights.shape == (tiny_config["vocab_size"], tiny_config["embed_dim"])

    def test_logits_use_embedding_weights(self, tiny_config):
        """Verify logits are computed via matmul with embedding weights."""
        model = GPT2(**tiny_config)
        input_ids = _random_ids((1, 8), tiny_config["vocab_size"])
        outputs = model(input_ids, training=False)

        hidden_states = outputs["last_hidden_state"]
        embedding_weights = model.decoder.word_embeddings.embeddings
        expected_logits = keras.ops.matmul(
            hidden_states, keras.ops.transpose(embedding_weights)
        )

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(outputs["logits"]),
            keras.ops.convert_to_numpy(expected_logits),
            atol=1e-5,
        )


# ---------------------------------------------------------------------
# Causal Masking Tests
# ---------------------------------------------------------------------


class TestGPT2CausalMasking:
    """Test that causal masking prevents attending to future tokens."""

    def test_causal_masking_later_tokens_differ(self, tiny_config):
        """Later tokens should differ when context changes."""
        model = GPT2(**tiny_config)

        # Same prefix up to position 3, then diverge
        seq1 = np.array([[1, 2, 3, 10, 20, 30, 40, 50]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 99, 99, 99, 99, 99]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)

        logits1 = keras.ops.convert_to_numpy(out1["logits"])
        logits2 = keras.ops.convert_to_numpy(out2["logits"])

        # Tokens at position 4+ should differ (different preceding tokens)
        assert not np.allclose(logits1[0, 4], logits2[0, 4], atol=1e-3)

        # Logits at positions 0-2 should be closer than at position 4+
        # (positions 0-2 see identical context under causal masking)
        diff_early = np.abs(logits1[0, 2] - logits2[0, 2]).max()
        diff_late = np.abs(logits1[0, 5] - logits2[0, 5]).max()
        assert diff_early < diff_late


# ---------------------------------------------------------------------
# Serialization Tests
# ---------------------------------------------------------------------


class TestGPT2Serialization:
    """Test GPT-2 model serialization and deserialization."""

    def test_get_config(self, tiny_config):
        model = GPT2(**tiny_config)
        config = model.get_config()

        assert config["vocab_size"] == tiny_config["vocab_size"]
        assert config["embed_dim"] == tiny_config["embed_dim"]
        assert config["depth"] == tiny_config["depth"]
        assert config["num_heads"] == tiny_config["num_heads"]
        assert config["max_seq_len"] == tiny_config["max_seq_len"]

    def test_from_config_roundtrip(self, tiny_config):
        model = GPT2(**tiny_config)
        config = model.get_config()
        model2 = GPT2.from_config(config)

        assert model2.vocab_size == model.vocab_size
        assert model2.embed_dim == model.embed_dim
        assert model2.depth == model.depth
        assert model2.num_heads == model.num_heads

    def test_compute_output_shape(self, tiny_config):
        model = GPT2(**tiny_config)
        shapes = model.compute_output_shape((None, 16))
        assert shapes["logits"] == (None, 16, tiny_config["vocab_size"])
        assert shapes["last_hidden_state"] == (None, 16, tiny_config["embed_dim"])


# ---------------------------------------------------------------------
# Variant Tests
# ---------------------------------------------------------------------


class TestGPT2Variants:
    """Test GPT-2 model variant creation."""

    def test_from_variant_tiny(self):
        model = GPT2.from_variant("tiny")
        assert model.embed_dim == 256
        assert model.depth == 4
        assert model.num_heads == 4

    def test_from_variant_small(self):
        model = GPT2.from_variant("small")
        assert model.embed_dim == 768
        assert model.depth == 12
        assert model.num_heads == 12

    def test_from_variant_with_overrides(self):
        model = GPT2.from_variant("tiny", dropout_rate=0.2)
        assert model.dropout_rate == 0.2
        assert model.embed_dim == 256

    def test_from_variant_unknown(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            GPT2.from_variant("nonexistent")

    def test_from_variant_pretrained_missing_file(self):
        with pytest.raises(FileNotFoundError):
            GPT2.from_variant("tiny", pretrained="/nonexistent/path.keras")

    def test_all_variants_have_required_keys(self):
        required_keys = {"embed_dim", "depth", "num_heads", "max_seq_len"}
        for name, config in GPT2.MODEL_VARIANTS.items():
            for key in required_keys:
                assert key in config, f"Variant '{name}' missing key '{key}'"


# ---------------------------------------------------------------------
# Gradient Flow Tests
# ---------------------------------------------------------------------


class TestGPT2GradientFlow:
    """Test gradient flow through the model."""

    def test_gradient_flow(self, tiny_config):
        model = GPT2(**tiny_config)
        batch_size, seq_len = 2, 8
        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        labels = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])

        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            logits = outputs["logits"]
            loss = keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True
            )
            loss = keras.ops.mean(loss)

        gradients = tape.gradient(loss, model.trainable_variables)

        for var, grad in zip(model.trainable_variables, gradients):
            assert grad is not None, f"No gradient for {var.name}"

    def test_training_step(self, tiny_config):
        """Test that model can do a training step with GradientTape."""
        model = GPT2(**tiny_config)
        batch_size, seq_len = 4, 8
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        input_ids = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])
        labels = _random_ids((batch_size, seq_len), tiny_config["vocab_size"])

        # Get initial weights snapshot
        _ = model(input_ids, training=False)
        initial_weights = [w.numpy().copy() for w in model.trainable_weights[:2]]

        with tf.GradientTape() as tape:
            outputs = model(input_ids, training=True)
            loss = loss_fn(labels, outputs["logits"])

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Weights should have changed
        updated_weights = [w.numpy() for w in model.trainable_weights[:2]]
        for w_init, w_updated in zip(initial_weights, updated_weights):
            assert not np.allclose(w_init, w_updated), "Weights did not update"
