import pytest
import keras
import numpy as np
import tensorflow as tf
import tempfile
import os
from keras import ops
from typing import Dict, Any

# ---------------------------------------------------------------------

from dl_techniques.models.modern_bert.components import (
    ModernBertEmbeddings,
    ModernBertAttention,
    ModernBertEncoderLayer,
)
# ---------------------------------------------------------------------

class TestModernBertEmbeddings:
    """Comprehensive test suite for the ModernBertEmbeddings layer."""

    @pytest.fixture
    def layer_config(self) -> Dict[str, Any]:
        """Provides a standard configuration for the embedding layer."""
        return {
            "vocab_size": 1000,
            "hidden_size": 64,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": 0.1,
            "use_bias": True,
        }

    @pytest.fixture
    def sample_input(self) -> Dict[str, keras.KerasTensor]:
        """Provides sample input tensors for the layer."""
        batch_size, seq_len = 4, 16
        return {
            "input_ids": ops.convert_to_tensor(
                np.random.randint(0, 1000, size=(batch_size, seq_len)), dtype="int32"
            ),
            "token_type_ids": ops.zeros((batch_size, seq_len), dtype="int32"),
        }

    def test_initialization(self, layer_config):
        """Tests that the layer initializes correctly with its sub-layers."""
        layer = ModernBertEmbeddings(**layer_config)
        assert not layer.built
        assert isinstance(layer.word_embeddings, keras.layers.Embedding)
        assert isinstance(layer.token_type_embeddings, keras.layers.Embedding)
        assert isinstance(layer.layer_norm, keras.layers.LayerNormalization)
        assert isinstance(layer.dropout, keras.layers.Dropout)
        assert layer.hidden_size == layer_config["hidden_size"]

    def test_forward_pass(self, layer_config, sample_input):
        """Tests the forward pass and correct output shape."""
        layer = ModernBertEmbeddings(**layer_config)
        output = layer(**sample_input)
        assert layer.built
        expected_shape = (
            sample_input["input_ids"].shape[0],
            sample_input["input_ids"].shape[1],
            layer_config["hidden_size"],
        )
        assert output.shape == expected_shape

    def test_forward_pass_no_token_types(self, layer_config, sample_input):
        """Tests that the layer functions correctly without token_type_ids."""
        layer = ModernBertEmbeddings(**layer_config)
        output = layer(input_ids=sample_input["input_ids"])
        expected_shape = (
            sample_input["input_ids"].shape[0],
            sample_input["input_ids"].shape[1],
            layer_config["hidden_size"],
        )
        assert output.shape == expected_shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Verifies the full save/load cycle."""
        # 1. Create original layer in a model
        # FIX: Add names to Input layers to resolve warnings
        inputs = {
            "input_ids": keras.Input(shape=(None,), dtype="int32", name="input_ids"),
            "token_type_ids": keras.Input(shape=(None,), dtype="int32", name="token_type_ids"),
        }
        layer_output = ModernBertEmbeddings(**layer_config)(**inputs)
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
                err_msg="Predictions differ after serialization cycle",
            )

    def test_config_completeness(self, layer_config):
        """Tests that get_config contains all __init__ parameters."""
        layer = ModernBertEmbeddings(**layer_config)
        config = layer.get_config()
        for key in layer_config:
            assert key in config, f"Missing key '{key}' in get_config()"

    def test_gradients_flow(self, layer_config, sample_input):
        """Tests that gradients can be computed through the layer."""
        layer = ModernBertEmbeddings(**layer_config)
        with tf.GradientTape() as tape:
            # Note: Must watch the input tensors for gradient calculation
            # as they are not tf.Variable by default.
            tape.watch(sample_input["input_ids"])
            tape.watch(sample_input["token_type_ids"])
            output = layer(**sample_input)
            loss = ops.mean(ops.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in gradients)
        # FIX: Correct number of trainable variables is 4
        # (word_emb, token_type_emb, norm_gamma, norm_beta)
        assert len(gradients) == 4

    @pytest.mark.parametrize("training", [True, False, None])
    def test_training_modes(self, layer_config, sample_input, training):
        """Tests that the layer runs in all training modes."""
        layer = ModernBertEmbeddings(**layer_config)
        output = layer(**sample_input, training=training)
        assert output is not None


class TestModernBertAttention:
    """Comprehensive test suite for the ModernBertAttention layer."""

    @pytest.fixture(params=[True, False])
    def is_global(self, request):
        return request.param

    @pytest.fixture
    def layer_config(self, is_global) -> Dict[str, Any]:
        """Provides a standard configuration for the attention layer."""
        return {
            "hidden_size": 32,
            "num_heads": 4,
            "attention_probs_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "rope_theta": 10000.0,
            "max_seq_len": 128,
            "local_attention_window_size": 8,
            "use_bias": False,
            "is_global": is_global,
        }

    @pytest.fixture
    def sample_input(self, layer_config) -> keras.KerasTensor:
        """Provides a sample input tensor for the layer."""
        batch_size, seq_len = 2, 16
        return keras.random.normal(shape=(batch_size, seq_len, layer_config["hidden_size"]))

    def test_initialization(self, layer_config):
        """Tests correct initialization of attention and RoPE sub-layers."""
        layer = ModernBertAttention(**layer_config)
        assert not layer.built
        assert isinstance(layer.mha, keras.layers.MultiHeadAttention)
        assert "RotaryPositionEmbedding" in str(type(layer.rotary_embedding))
        assert layer.is_global == layer_config["is_global"]

    def test_forward_pass(self, layer_config, sample_input):
        """Tests the forward pass and preserved output shape."""
        layer = ModernBertAttention(**layer_config)
        output = layer(sample_input)
        assert layer.built
        assert output.shape == sample_input.shape

    def test_forward_pass_with_mask(self, layer_config, sample_input):
        """Tests the forward pass with an attention mask."""
        mask = ops.ones((sample_input.shape[0], sample_input.shape[1]), dtype="int32")
        layer = ModernBertAttention(**layer_config)
        output = layer(sample_input, attention_mask=mask)
        assert output.shape == sample_input.shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Verifies the full save/load cycle."""
        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = ModernBertAttention(**layer_config)(inputs)
        model = keras.Model(inputs, layer_output)
        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_config_completeness(self, layer_config):
        """Tests that get_config contains all __init__ parameters."""
        layer = ModernBertAttention(**layer_config)
        config = layer.get_config()
        for key in layer_config:
            assert key in config, f"Missing key '{key}' in get_config()"

    def test_gradients_flow(self, layer_config, sample_input):
        """Tests that gradients can be computed through the layer."""
        layer = ModernBertAttention(**layer_config)
        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = ops.mean(ops.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)
        assert all(g is not None for g in gradients)
        assert len(gradients) == 4 # Q, K, V, Output projections

    def test_edge_cases(self):
        """Tests error conditions, like indivisible hidden_size."""
        with pytest.raises(ValueError, match="must be divisible"):
            ModernBertAttention(hidden_size=32, num_heads=5, is_global=True, rope_theta=1.0, max_seq_len=1, local_attention_window_size=1, use_bias=True, attention_probs_dropout_prob=0.0, initializer_range=0.02)


class TestModernBertEncoderLayer:
    """Comprehensive test suite for the ModernBertEncoderLayer."""

    @pytest.fixture(params=[True, False])
    def is_global(self, request):
        return request.param

    @pytest.fixture
    def layer_config(self, is_global) -> Dict[str, Any]:
        """Provides a full config dict as expected by the encoder layer's kwargs."""
        return {
            "is_global": is_global,
            "hidden_size": 32,
            "num_heads": 4,
            "intermediate_size": 64,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "use_bias": True,
            "rope_theta_local": 10000.0,
            "rope_theta_global": 50000.0,
            "max_seq_len": 128,
            "local_attention_window_size": 8,
        }

    @pytest.fixture
    def sample_input(self, layer_config) -> keras.KerasTensor:
        """Provides a sample input tensor for the layer."""
        batch_size, seq_len = 2, 16
        return keras.random.normal(shape=(batch_size, seq_len, layer_config["hidden_size"]))

    def test_initialization(self, layer_config):
        """Tests correct initialization of all sub-layers."""
        config_copy = layer_config.copy()
        is_global_val = config_copy.pop("is_global")
        layer = ModernBertEncoderLayer(is_global=is_global_val, **config_copy)

        assert not layer.built
        assert isinstance(layer.attention_norm, keras.layers.LayerNormalization)
        assert isinstance(layer.attention, ModernBertAttention)
        assert isinstance(layer.ffn_norm, keras.layers.LayerNormalization)
        assert "GeGLUFFN" in str(type(layer.ffn))
        assert layer.is_global == is_global_val

    def test_forward_pass(self, layer_config, sample_input):
        """Tests the forward pass and preserved output shape."""
        config_copy = layer_config.copy()
        is_global_val = config_copy.pop("is_global")
        layer = ModernBertEncoderLayer(is_global=is_global_val, **config_copy)
        output = layer(sample_input)

        assert layer.built
        assert output.shape == sample_input.shape

    def test_forward_pass_with_mask(self, layer_config, sample_input):
        """Tests the forward pass with an attention mask."""
        mask = ops.ones((sample_input.shape[0], sample_input.shape[1]), dtype="int32")
        config_copy = layer_config.copy()
        is_global_val = config_copy.pop("is_global")
        layer = ModernBertEncoderLayer(is_global=is_global_val, **config_copy)
        output = layer(sample_input, attention_mask=mask)

        assert output.shape == sample_input.shape

    def test_serialization_cycle(self, layer_config, sample_input):
        """CRITICAL TEST: Verifies the full save/load cycle."""
        config_copy = layer_config.copy()
        is_global_val = config_copy.pop("is_global")

        inputs = keras.Input(shape=sample_input.shape[1:])
        layer_output = ModernBertEncoderLayer(is_global=is_global_val, **config_copy)(inputs)
        model = keras.Model(inputs, layer_output)
        original_prediction = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.keras")
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_prediction = loaded_model(sample_input)

            np.testing.assert_allclose(
                ops.convert_to_numpy(original_prediction),
                ops.convert_to_numpy(loaded_prediction),
                rtol=1e-6, atol=1e-6
            )

    def test_config_completeness(self, layer_config):
        """Tests that get_config contains all __init__ parameters."""
        config_copy = layer_config.copy()
        is_global_val = config_copy.pop("is_global")
        layer = ModernBertEncoderLayer(is_global=is_global_val, **config_copy)
        config = layer.get_config()

        for key in layer_config:
            assert key in config, f"Missing key '{key}' in get_config()"

    def test_gradients_flow(self, layer_config, sample_input):
        """Tests that gradients can be computed through the layer."""
        config_copy = layer_config.copy()
        is_global_val = config_copy.pop("is_global")
        layer = ModernBertEncoderLayer(is_global=is_global_val, **config_copy)

        with tf.GradientTape() as tape:
            output = layer(sample_input)
            loss = ops.mean(ops.square(output))
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert all(g is not None for g in gradients)
        assert len(gradients) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])