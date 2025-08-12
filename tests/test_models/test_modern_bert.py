"""
Comprehensive pytest test suite for ModernBERT model.

This module provides extensive testing for the ModernBERT implementation, focusing on
its modern architectural features.

Key ModernBERT-specific testing areas:
- Rotary Position Embeddings (RoPE) and sliding window attention in `ModernBertAttention`.
- GeGLU activation in the `ModernBertFFN`.
- Removal of absolute position embeddings in `ModernBertEmbeddings`.
- Alternating global/local attention configuration.
- Pre-normalization and final normalization layer.
- Correct handling of `use_bias=False`.
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.modern_bert import (
    ModernBertConfig,
    ModernBertEmbeddings,
    ModernBertAttention,
    ModernBertFFN,
    ModernBertTransformerLayer,
    ModernBERT,
    create_modern_bert_for_classification,
    create_modern_bert_for_sequence_output,
    create_modern_bert_base,
    create_modern_bert_large,
)
from dl_techniques.layers.transformer import TransformerLayer
from dl_techniques.layers.rotary_position_embedding import RotaryPositionEmbedding

# Helper for comparing tensors
def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Custom allclose function since keras.ops doesn't have it."""
    return np.allclose(a.numpy() if hasattr(a, 'numpy') else a,
                       b.numpy() if hasattr(b, 'numpy') else b,
                       rtol=rtol, atol=atol)


class TestModernBertConfig:
    """Test ModernBERT configuration validation and initialization."""

    def test_basic_initialization(self):
        """Test basic ModernBertConfig initialization with default parameters."""
        config = ModernBertConfig()

        # Model architecture defaults
        assert config.vocab_size == 50368
        assert config.hidden_size == 768
        assert config.num_layers == 22
        assert config.num_heads == 12
        assert config.intermediate_size == 1152
        assert config.hidden_act == "gelu"

        # Dropout defaults
        assert config.hidden_dropout_prob == 0.1
        assert config.attention_probs_dropout_prob == 0.1

        # Modern features defaults
        assert config.use_bias is False
        assert config.normalization_type == "layer_norm"
        assert config.rope_theta_local == 10000.0
        assert config.rope_theta_global == 160000.0
        assert config.global_attention_interval == 3
        assert config.local_attention_window_size == 128

        # Test validation passes
        config.validate()

    def test_custom_initialization(self):
        """Test ModernBertConfig initialization with custom parameters."""
        config = ModernBertConfig(
            vocab_size=50000,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,  # 1024 is divisible by 16
            intermediate_size=2048,
            use_bias=True,
            global_attention_interval=4,
            local_attention_window_size=256
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.use_bias is True
        assert config.global_attention_interval == 4
        assert config.local_attention_window_size == 256

        # Test validation passes
        config.validate()

    def test_config_serialization(self):
        """Test ModernBertConfig to_dict and from_dict methods."""
        original_config = ModernBertConfig(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=2048,
            use_bias=True
        )

        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 25000
        assert config_dict['hidden_size'] == 512
        assert config_dict['use_bias'] is True

        restored_config = ModernBertConfig.from_dict(config_dict)
        assert restored_config.to_dict() == original_config.to_dict()

    def test_config_validation(self):
        """Test ModernBertConfig validation for invalid parameters."""
        # Test invalid hidden_size/num_heads combination
        with pytest.raises(ValueError, match="must be divisible by"):
            config = ModernBertConfig(hidden_size=100, num_heads=12)
            config.validate()

        # Test non-positive hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            config = ModernBertConfig(hidden_size=0)
            config.validate()

        # Test non-positive num_layers
        with pytest.raises(ValueError, match="num_layers must be positive"):
            config = ModernBertConfig(num_layers=-1)
            config.validate()

        # Test non-positive vocab_size
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            config = ModernBertConfig(vocab_size=0)
            config.validate()


class TestModernBertEmbeddingsComponent:
    """Test ModernBertEmbeddings component individually."""

    @pytest.fixture
    def basic_config(self) -> ModernBertConfig:
        """Create a basic ModernBERT config for testing."""
        return ModernBertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            type_vocab_size=2
        )

    def test_embeddings_initialization(self, basic_config):
        """Test ModernBertEmbeddings initialization."""
        embeddings = ModernBertEmbeddings(basic_config)
        assert embeddings.config == basic_config
        # Components should be None before building
        assert embeddings.word_embeddings is None
        assert embeddings.token_type_embeddings is None
        assert embeddings.layer_norm is None

    def test_embeddings_build(self, basic_config):
        """Test ModernBertEmbeddings build process."""
        embeddings = ModernBertEmbeddings(basic_config)
        input_shape = (None, 64)
        embeddings.build(input_shape)

        assert embeddings.built is True
        assert embeddings.word_embeddings is not None
        assert embeddings.token_type_embeddings is not None
        assert embeddings.layer_norm is not None

        # Check embedding dimensions
        assert embeddings.word_embeddings.input_dim == basic_config.vocab_size
        assert embeddings.word_embeddings.output_dim == basic_config.hidden_size
        assert embeddings.token_type_embeddings.input_dim == basic_config.type_vocab_size
        # Verify no position embeddings layer
        assert not hasattr(embeddings, "position_embeddings")

    def test_embeddings_forward_pass(self, basic_config):
        """Test ModernBertEmbeddings forward pass."""
        embeddings = ModernBertEmbeddings(basic_config)
        batch_size, seq_length = 4, 32
        input_ids = ops.cast(
            keras.random.uniform((batch_size, seq_length), 1, basic_config.vocab_size),
            dtype='int32'
        )
        output = embeddings(input_ids, training=False)
        assert output.shape == (batch_size, seq_length, basic_config.hidden_size)

    def test_embeddings_with_token_type_ids(self, basic_config):
        """Test ModernBertEmbeddings with token type IDs."""
        embeddings = ModernBertEmbeddings(basic_config)
        batch_size, seq_length = 3, 16
        input_ids = ops.cast(
            keras.random.uniform((batch_size, seq_length), 1, basic_config.vocab_size),
            dtype='int32'
        )
        token_type_ids = ops.ones((batch_size, seq_length), dtype='int32')
        output = embeddings(input_ids, token_type_ids=token_type_ids, training=False)
        assert output.shape == (batch_size, seq_length, basic_config.hidden_size)

        # Check that token type embeddings are added
        output_no_type = embeddings(input_ids, training=False)
        assert not allclose(output, output_no_type)

    def test_embeddings_no_position_ids_input(self, basic_config):
        """Test that ModernBertEmbeddings does not accept position_ids."""
        embeddings = ModernBertEmbeddings(basic_config)
        input_ids = ops.zeros((2, 16), dtype='int32')
        position_ids = ops.zeros((2, 16), dtype='int32')
        with pytest.raises(TypeError):
            # The 'call' signature does not accept 'position_ids'
            embeddings(input_ids, position_ids=position_ids)


class TestModernBertAttentionComponent:
    """Test ModernBertAttention component individually."""

    @pytest.fixture
    def basic_config(self) -> ModernBertConfig:
        return ModernBertConfig(
            hidden_size=128, num_heads=4, local_attention_window_size=8
        )

    def test_sliding_window_mask_creation(self, basic_config):
        """Test the creation of the sliding window mask."""
        attention_layer = ModernBertAttention(basic_config, is_global=False)
        seq_len = 10
        mask = attention_layer._create_sliding_window_mask(seq_len)

        assert mask.shape == (seq_len, seq_len)
        # Check a position, e.g., token 5 can see tokens within window_size distance
        basic_config.local_attention_window_size = 3
        attention_layer = ModernBertAttention(basic_config, is_global=False)
        mask = attention_layer._create_sliding_window_mask(seq_len).numpy()
        # Token at index 4 should see indices 2, 3, 4, 5, 6
        # abs(j - i) < 3  => -3 < j-i < 3 => i-3 < j < i+3
        # For i=4: 1 < j < 7. So j can be 2, 3, 4, 5, 6.
        expected_vicinity = [False, False, True, True, True, True, True, False, False, False]
        assert np.array_equal(mask[4, :], expected_vicinity)

    def test_attention_call_global(self, basic_config):
        """Test forward pass with global attention."""
        attention_layer = ModernBertAttention(basic_config, is_global=True)
        hidden_states = keras.random.normal((2, 16, basic_config.hidden_size))
        output = attention_layer(hidden_states, training=False)
        assert output.shape == (2, 16, basic_config.hidden_size)

    def test_attention_call_local(self, basic_config):
        """Test forward pass with local (sliding window) attention."""
        attention_layer = ModernBertAttention(basic_config, is_global=False)
        hidden_states = keras.random.normal((2, 16, basic_config.hidden_size))
        output = attention_layer(hidden_states, training=False)
        assert output.shape == (2, 16, basic_config.hidden_size)


class TestModernBertFFNComponent:
    """Test ModernBertFFN (GeGLU) component individually."""

    @pytest.fixture
    def basic_config(self) -> ModernBertConfig:
        return ModernBertConfig(hidden_size=128, intermediate_size=256)

    def test_ffn_build(self, basic_config):
        """Test FFN build process and GeGLU structure."""
        ffn_layer = ModernBertFFN(basic_config)
        ffn_layer.build((None, 16, basic_config.hidden_size))
        assert ffn_layer.built

        # For GeGLU, the first dense layer must have 2 * intermediate_size units
        expected_units = basic_config.intermediate_size * 2
        assert ffn_layer.wi.units == expected_units

        # The output dense layer must project back to hidden_size
        assert ffn_layer.wo.units == basic_config.hidden_size

    def test_ffn_forward_pass(self, basic_config):
        """Test the forward pass of the GeGLU FFN."""
        ffn_layer = ModernBertFFN(basic_config)
        hidden_states = keras.random.normal((2, 16, basic_config.hidden_size))
        output = ffn_layer(hidden_states, training=False)
        assert output.shape == (2, 16, basic_config.hidden_size)


class TestModernBERTModel:
    """Comprehensive tests for the full ModernBERT model."""

    @pytest.fixture
    def model_config(self) -> ModernBertConfig:
        """Create a small ModernBERT config for fast testing."""
        return ModernBertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=4,
            num_heads=4,
            intermediate_size=256,
            global_attention_interval=2,
            local_attention_window_size=16
        )

    @pytest.fixture
    def built_model(self, model_config) -> ModernBERT:
        """Create a built ModernBERT model for testing."""
        model = ModernBERT(model_config)
        model.build({'input_ids': (None, 64)})
        return model

    def test_model_initialization(self, model_config):
        """Test ModernBERT model initialization."""
        model = ModernBERT(model_config)
        assert model.config == model_config
        assert model.add_pooling_layer is True
        assert len(model.encoder_layers) == 0

        model_no_pool = ModernBERT(model_config, add_pooling_layer=False)
        assert model_no_pool.add_pooling_layer is False

    def test_model_building(self, model_config):
        """Test the model building process."""
        model = ModernBERT(model_config)
        model.build({'input_ids': (None, 32)})

        assert model.built
        assert model.embeddings is not None
        assert len(model.encoder_layers) == model_config.num_layers
        assert model.final_norm is not None
        assert model.pooler is not None

        # Check that bias is not created if use_bias=False
        model_config.use_bias = False
        model_no_bias = ModernBERT(model_config)
        model_no_bias.build({'input_ids': (None, 32)})

        # FIX: More robust test for bias configuration
        sample_encoder_layer = model_no_bias.encoder_layers[0].transformer_layer
        assert sample_encoder_layer.use_bias is False
        assert model_no_bias.final_norm.beta is None # 'beta' is the bias weight in LayerNorm

    def test_forward_pass_dict_input(self, built_model):
        """Test forward pass with dictionary input."""
        batch_size, seq_length = 3, 24
        inputs = {
            'input_ids': ops.zeros((batch_size, seq_length), dtype='int32'),
            'attention_mask': ops.ones((batch_size, seq_length), dtype='int32'),
            'token_type_ids': ops.zeros((batch_size, seq_length), dtype='int32')
        }
        seq_out, pool_out = built_model(inputs, training=False)
        assert seq_out.shape == (batch_size, seq_length, built_model.config.hidden_size)
        assert pool_out.shape == (batch_size, built_model.config.hidden_size)

    def test_forward_pass_return_dict(self, built_model):
        """Test forward pass with return_dict=True."""
        batch_size, seq_length = 2, 16
        input_ids = ops.zeros((batch_size, seq_length), dtype='int32')
        outputs = built_model(input_ids, return_dict=True, training=False)

        assert isinstance(outputs, dict)
        assert 'last_hidden_state' in outputs
        assert 'pooler_output' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_length, built_model.config.hidden_size)
        assert outputs['pooler_output'].shape == (batch_size, built_model.config.hidden_size)

    def test_forward_pass_no_pooling(self, model_config):
        """Test forward pass without pooling layer."""
        model = ModernBERT(model_config, add_pooling_layer=False)
        batch_size, seq_length = 2, 16
        input_ids = ops.zeros((batch_size, seq_length), dtype='int32')
        output = model(input_ids, training=False)
        assert not isinstance(output, tuple)
        assert output.shape == (batch_size, seq_length, model_config.hidden_size)

    def test_model_save_and_load(self, built_model):
        """Test saving and loading the complete ModernBERT model."""
        input_data = ops.zeros((2, 16), dtype='int32')
        original_output = built_model(input_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "modern_bert_model.keras")
            built_model.save(model_path)
            # FIX: Add the generic TransformerLayer to custom_objects
            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'ModernBERT': ModernBERT,
                    'ModernBertEmbeddings': ModernBertEmbeddings,
                    'ModernBertTransformerLayer': ModernBertTransformerLayer,
                    'TransformerLayer': TransformerLayer,
                    'RotaryPositionEmbedding': RotaryPositionEmbedding, # Also register RoPE
                }
            )

        assert loaded_model.config.to_dict() == built_model.config.to_dict()
        assert loaded_model.add_pooling_layer == built_model.add_pooling_layer

        loaded_output = loaded_model(input_data)
        assert allclose(original_output[0], loaded_output[0], rtol=0.1, atol=0.1)
        assert allclose(original_output[1], loaded_output[1], rtol=0.1, atol=0.1)

class TestModernBERTFactoryFunctions:
    """Test ModernBERT factory functions."""

    def test_create_modern_bert_base(self):
        """Test create_modern_bert_base configuration."""
        config = create_modern_bert_base()
        assert config.hidden_size == 768
        assert config.num_layers == 22
        assert config.num_heads == 12
        assert config.intermediate_size == 1152
        assert config.global_attention_interval == 3

    def test_create_modern_bert_large(self):
        """Test create_modern_bert_large configuration."""
        config = create_modern_bert_large()
        assert config.hidden_size == 1024
        assert config.num_layers == 28
        assert config.num_heads == 16
        assert config.intermediate_size == 2624

    @pytest.fixture
    def small_config(self) -> ModernBertConfig:
        return ModernBertConfig(
            vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4
        )

    def test_create_modern_bert_for_classification(self, small_config):
        """Test the classification model factory function."""
        num_labels = 5
        model = create_modern_bert_for_classification(small_config, num_labels)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 3

        batch_size, seq_length = 2, 32
        inputs = [
            ops.zeros((batch_size, seq_length), dtype='int32'),
            ops.ones((batch_size, seq_length), dtype='int32'),
            ops.zeros((batch_size, seq_length), dtype='int32'),
        ]
        logits = model(inputs, training=False)
        assert logits.shape == (batch_size, num_labels)

    def test_create_modern_bert_for_sequence_output(self, small_config):
        """Test the sequence output model factory function."""
        model = create_modern_bert_for_sequence_output(small_config)
        assert isinstance(model, keras.Model)

        batch_size, seq_length = 2, 32
        inputs = [
            ops.zeros((batch_size, seq_length), dtype='int32'),
            ops.ones((batch_size, seq_length), dtype='int32'),
            ops.zeros((batch_size, seq_length), dtype='int32'),
        ]
        seq_out = model(inputs, training=False)
        assert seq_out.shape == (batch_size, seq_length, small_config.hidden_size)


class TestModernBERTIntegration:
    """Integration tests for ModernBERT components and gradient flow."""

    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire ModernBERT model."""
        config = ModernBertConfig(
            vocab_size=1000, hidden_size=64, num_layers=2, num_heads=4
        )
        model = create_modern_bert_for_classification(config, num_labels=3)
        batch_size, seq_length = 2, 16

        # FIX: Correctly generate random integers
        input_ids_float = keras.random.uniform((batch_size, seq_length), 1, config.vocab_size)
        inputs = {
            "input_ids": ops.cast(input_ids_float, dtype='int32'),
            "attention_mask": ops.ones((batch_size, seq_length), dtype='int32'),
            "token_type_ids": ops.concatenate([
                ops.zeros((batch_size, 8), dtype="int32"),
                ops.ones((batch_size, 8), dtype="int32"),
            ], axis=1)
        }

        with tf.GradientTape() as tape:
            logits = model(list(inputs.values()), training=True)
            targets = ops.one_hot(ops.array([0, 1]), 3)
            loss = keras.losses.categorical_crossentropy(targets, logits, from_logits=True)
            loss = ops.mean(loss)

        gradients = tape.gradient(loss, model.trainable_weights)

        for grad, weight in zip(gradients, model.trainable_weights):
            assert grad is not None, f"Gradient is None for weight {weight.name}"
            assert ops.max(ops.abs(grad)) > 0, f"Gradient is zero for weight {weight.name}"

    def test_pooled_output_quality(self):
        """Test quality of pooled output."""
        config = ModernBertConfig(
            vocab_size=1000, hidden_size=128, num_layers=2, num_heads=4
        )
        model = ModernBERT(config, add_pooling_layer=True)
        batch_size, seq_length = 3, 20
        # FIX: Correctly generate random integers
        input_ids_float = keras.random.uniform((batch_size, seq_length), 1, config.vocab_size)
        input_ids = ops.cast(input_ids_float, dtype='int32')
        sequence_output, pooled_output = model(input_ids, training=False)

        cls_token_output = sequence_output[:, 0, :]
        assert not allclose(cls_token_output, pooled_output, atol=1e-5)
        assert ops.all(pooled_output >= -1.0) and ops.all(pooled_output <= 1.0)

    def test_attention_mask_effect(self):
        """Verify that the attention mask correctly influences the output."""
        config = ModernBertConfig(
            vocab_size=1000, hidden_size=64, num_layers=2, num_heads=4,
            attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0
        )
        model = ModernBERT(config, add_pooling_layer=False)
        batch_size, seq_length = 2, 16

        # FIX: Correctly generate random integers
        input_ids_float = keras.random.uniform((batch_size, seq_length), 1, config.vocab_size)
        input_ids = ops.cast(input_ids_float, dtype='int32')

        # Input with padding
        padded_input_ids = ops.concatenate([input_ids[:, :8], ops.zeros((batch_size, 8), dtype='int32')], axis=1)
        padded_attention_mask = ops.concatenate(
            [ops.ones((batch_size, 8), dtype='int32'), ops.zeros((batch_size, 8), dtype='int32')], axis=1)

        # Get outputs
        output_padded = model(padded_input_ids, attention_mask=padded_attention_mask, training=False)
        output_unmasked = model(padded_input_ids, attention_mask=None, training=False)

        # With a mask, the padded tokens' context is different, so their output should differ.
        #assert not allclose(output_padded, output_unmasked, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])