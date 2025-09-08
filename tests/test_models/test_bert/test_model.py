"""
Comprehensive pytest test suite for refactored BERT (Bidirectional Encoder Representations from Transformers) model.

This module provides extensive testing for the BERT implementation including:
- Model initialization and parameter validation
- Architecture building and shape consistency
- Forward pass functionality with different input formats
- Encoding operations and feature extraction
- Serialization and deserialization
- Error handling and edge cases
- Factory function testing for classification and sequence output models
- Integration testing
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.bert.model import (
    BERT,
    create_bert_for_classification,
    create_bert_for_sequence_output,
    create_bert_base_uncased,
    create_bert_large_uncased,
    create_bert_with_rms_norm,
    create_bert_with_advanced_features
)


class TestBERTModelInitialization:
    """Test BERT model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BERT model initialization."""
        config_dict = create_bert_base_uncased()
        config_dict.update(hidden_size=256, num_layers=6, num_heads=8)
        model = BERT(**config_dict)

        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert model.add_pooling_layer is True
        assert not model.built

        # Components should be created in __init__
        assert model.embeddings is not None
        assert len(model.encoder_layers) == 6
        assert model.pooler is not None

        # But should not be built yet
        assert not model.embeddings.built
        for layer in model.encoder_layers:
            assert not layer.built

    def test_initialization_without_pooling(self):
        """Test BERT model initialization without pooling layer."""
        config_dict = create_bert_base_uncased()
        model = BERT(**config_dict, add_pooling_layer=False)

        assert model.add_pooling_layer is False
        assert model.pooler is None

    def test_parameter_validation(self):
        """Test Bert parameter validation for invalid values."""
        config_dict = create_bert_base_uncased()

        # Test invalid hidden_size/num_heads combination
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            config_copy = config_dict.copy()
            config_copy.update(hidden_size=100, num_heads=12)
            BERT(**config_copy)

        # Test negative hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            config_copy = config_dict.copy()
            config_copy.update(hidden_size=-100)
            BERT(**config_copy)

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="hidden_dropout_prob must be between"):
            config_copy = config_dict.copy()
            config_copy.update(hidden_dropout_prob=1.5)
            BERT(**config_copy)

    def test_initialization_with_custom_config(self):
        """Test BERT model initialization with custom configuration."""
        config_dict = {
            "vocab_size": 25000,
            "hidden_size": 512,
            "num_layers": 8,
            "num_heads": 8,
            "intermediate_size": 2048,
            "hidden_dropout_prob": 0.2,
            "attention_probs_dropout_prob": 0.1,
            "normalization_type": "rms_norm",
            "normalization_position": "pre"
        }
        model = BERT(**config_dict)

        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert model.num_layers == 8
        assert model.hidden_dropout_prob == 0.2
        assert model.normalization_type == "rms_norm"


class TestBERTModelBuilding:
    """Test BERT model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Create a basic BERT config for testing."""
        return {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "intermediate_size": 1024,
            "max_position_embeddings": 128
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality."""
        model = BERT(**basic_config)
        batch_size, seq_length = 2, 32
        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config['vocab_size']
            ),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)

        assert model.built
        assert model.embeddings.built
        assert len(model.encoder_layers) == basic_config['num_layers']
        for layer in model.encoder_layers:
            assert layer.built

        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (batch_size, seq_length, basic_config['hidden_size'])
        assert pooled_output.shape == (batch_size, basic_config['hidden_size'])

    def test_build_without_pooling_layer(self, basic_config):
        """Test building BERT model without pooling layer."""
        model = BERT(**basic_config, add_pooling_layer=False)
        input_ids = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=basic_config['vocab_size']),
            dtype='int32'
        )
        output = model(input_ids, training=False)
        assert model.built
        assert model.pooler is None
        assert output.shape == (2, 32, basic_config['hidden_size'])

    def test_transformer_layers_configuration(self, basic_config):
        """Test that TransformerLayers are configured correctly."""
        model = BERT(**basic_config)
        input_ids = ops.cast(
            keras.random.uniform((1, 16), minval=1, maxval=basic_config['vocab_size']),
            dtype='int32'
        )
        _ = model(input_ids, training=False)

        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config['hidden_size']
            assert layer.num_heads == basic_config['num_heads']
            assert layer.intermediate_size == basic_config['intermediate_size']
            assert layer.name == f"encoder_layer_{i}"


class TestBERTModelForwardPass:
    """Test BERT model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> BERT:
        """Create a built BERT model for testing."""
        config_dict = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 3,
            "num_heads": 8,
            "intermediate_size": 1024,
            "max_position_embeddings": 128
        }
        model = BERT(**config_dict)
        sample_input = ops.cast(
            keras.random.uniform((1, 16), minval=1, maxval=config_dict['vocab_size']),
            dtype='int32'
        )
        _ = model(sample_input, training=False)
        return model

    def test_forward_pass_input_ids_only(self, built_model):
        """Test forward pass with only input IDs."""
        batch_size, seq_length = 4, 32
        input_ids = ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=built_model.vocab_size),
            dtype='int32'
        )
        sequence_output, pooled_output = built_model(input_ids, training=False)
        assert sequence_output.shape == (batch_size, seq_length, built_model.hidden_size)
        assert pooled_output.shape == (batch_size, built_model.hidden_size)

    def test_forward_pass_dict_input(self, built_model):
        """Test forward pass with dictionary input."""
        batch_size, seq_length = 3, 24
        input_ids = ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=built_model.vocab_size),
            dtype='int32'
        )
        inputs = {
            'input_ids': input_ids,
            'attention_mask': ops.ones((batch_size, seq_length), dtype='int32'),
            'token_type_ids': ops.zeros((batch_size, seq_length), dtype='int32')
        }
        sequence_output, pooled_output = built_model(inputs, training=False)
        assert sequence_output.shape == (batch_size, seq_length, built_model.hidden_size)
        assert pooled_output.shape == (batch_size, built_model.hidden_size)

    def test_forward_pass_return_dict(self, built_model):
        """Test forward pass with return_dict=True."""
        batch_size, seq_length = 2, 16
        input_ids = ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=built_model.vocab_size),
            dtype='int32'
        )
        outputs = built_model(input_ids, return_dict=True, training=False)
        assert isinstance(outputs, dict)
        assert 'last_hidden_state' in outputs
        assert 'pooler_output' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_length, built_model.hidden_size)
        assert outputs['pooler_output'].shape == (batch_size, built_model.hidden_size)

    def test_forward_pass_no_pooling(self):
        """Test forward pass without pooling layer."""
        config_dict = {"vocab_size": 1000, "hidden_size": 256, "num_layers": 2, "num_heads": 8}
        model = BERT(**config_dict, add_pooling_layer=False)
        input_ids = ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        outputs = model(input_ids, training=False)
        assert outputs.shape == (2, 16, config_dict['hidden_size'])


class TestBERTModelSerialization:
    """Test BERT model serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        config_dict = {
            "vocab_size": 25000, "hidden_size": 512, "num_layers": 8, "num_heads": 8,
            "intermediate_size": 2048, "hidden_dropout_prob": 0.15,
            "normalization_type": "rms_norm"
        }
        model = BERT(**config_dict)
        model_config = model.get_config()
        assert isinstance(model_config, dict)
        assert model_config['vocab_size'] == 25000
        assert model_config['hidden_size'] == 512
        assert model_config['num_layers'] == 8
        assert model_config['hidden_dropout_prob'] == 0.15
        assert model_config['normalization_type'] == "rms_norm"

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        config_dict = {
            "vocab_size": 1000, "hidden_size": 256, "num_layers": 2, "num_heads": 8,
            "intermediate_size": 512
        }
        model = BERT(**config_dict, add_pooling_layer=True)
        input_ids = ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        original_outputs = model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_bert.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

            assert isinstance(original_outputs, tuple)
            assert isinstance(loaded_outputs, tuple)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs[0]),
                keras.ops.convert_to_numpy(loaded_outputs[0]),
                rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs[1]),
                keras.ops.convert_to_numpy(loaded_outputs[1]),
                rtol=1e-5, atol=1e-6
            )


class TestBERTEdgeCases:
    """Test BERT model edge cases and error handling."""

    @pytest.fixture
    def config_dict(self):
        return {"vocab_size": 1000, "hidden_size": 128, "num_layers": 2, "num_heads": 8}

    def test_minimum_sequence_length(self, config_dict):
        """Test BERT with minimum sequence length."""
        model = BERT(**config_dict)
        input_ids = ops.cast([[42]], dtype='int32')
        sequence_output, pooled_output = model(input_ids, training=False)
        assert sequence_output.shape == (1, 1, 128)
        assert pooled_output.shape == (1, 128)

    def test_single_sample_batch(self, config_dict):
        """Test BERT with batch size of 1."""
        model = BERT(**config_dict)
        input_ids = ops.cast(
            keras.random.uniform((1, 32), minval=1, maxval=1000), dtype='int32'
        )
        sequence_output, pooled_output = model(input_ids, training=False)
        assert sequence_output.shape == (1, 32, 128)
        assert pooled_output.shape == (1, 128)


class TestBERTFactoryFunctions:
    """Test BERT factory functions."""

    def test_create_bert_for_classification(self):
        """Test create_bert_for_classification factory function."""
        config_dict = {"vocab_size": 1000, "hidden_size": 256, "num_layers": 3, "num_heads": 8}
        num_labels = 5
        model = create_bert_for_classification(config_dict, num_labels)
        assert isinstance(model, keras.Model)
        input_ids = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        attention_mask = ops.ones((2, 32), dtype='int32')
        token_type_ids = ops.zeros((2, 32), dtype='int32')
        logits = model([input_ids, attention_mask, token_type_ids], training=False)
        assert logits.shape == (2, num_labels)

    def test_create_bert_for_sequence_output(self):
        """Test create_bert_for_sequence_output factory function."""
        config_dict = {"vocab_size": 1000, "hidden_size": 256, "num_layers": 3, "num_heads": 8}
        model = create_bert_for_sequence_output(config_dict)
        assert isinstance(model, keras.Model)
        input_ids = ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        attention_mask = ops.ones((2, 32), dtype='int32')
        token_type_ids = ops.zeros((2, 32), dtype='int32')
        sequence_output = model([input_ids, attention_mask, token_type_ids], training=False)
        assert sequence_output.shape == (2, 32, config_dict['hidden_size'])

    def test_create_bert_base_uncased(self):
        """Test create_bert_base_uncased configuration."""
        config_dict = create_bert_base_uncased()
        assert config_dict['vocab_size'] == 30522
        assert config_dict['hidden_size'] == 768
        assert config_dict['num_layers'] == 12

    def test_create_bert_large_uncased(self):
        """Test create_bert_large_uncased configuration."""
        config_dict = create_bert_large_uncased()
        assert config_dict['vocab_size'] == 30522
        assert config_dict['hidden_size'] == 1024
        assert config_dict['num_layers'] == 24

    def test_create_bert_with_rms_norm(self):
        """Test create_bert_with_rms_norm configurations."""
        config = create_bert_with_rms_norm(size="base", normalization_position="pre")
        assert config['hidden_size'] == 768
        assert config['normalization_type'] == "rms_norm"
        assert config['normalization_position'] == "pre"


class TestBERTIntegration:
    """Integration tests for BERT components working together."""

    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire BERT model."""
        config_dict = {
            "vocab_size": 1000, "hidden_size": 64, "num_layers": 2, "num_heads": 8,
            "intermediate_size": 256
        }
        model = create_bert_for_classification(config_dict, num_labels=3)
        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
        # Use varied token_type_ids to ensure gradient flow to token type embeddings
        token_type_ids = ops.concatenate([
            ops.zeros((batch_size, seq_length // 2), dtype="int32"),
            ops.ones((batch_size, seq_length - seq_length // 2), dtype="int32"),
        ], axis=1)

        with tf.GradientTape() as tape:
            logits = model([input_ids, attention_mask, token_type_ids], training=True)
            targets = ops.one_hot(ops.array([0, 2]), 3)
            loss = ops.mean(keras.losses.categorical_crossentropy(targets, logits))

        gradients = tape.gradient(loss, model.trainable_weights)
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > len(model.trainable_weights) * 0.5
        grad_norms = [ops.sqrt(ops.sum(ops.square(g))) for g in non_none_grads]
        assert all(norm > 0.0 for norm in grad_norms)


class TestBERTAdvancedFeatures:
    """Test advanced BERT features from dl-techniques framework."""

    def test_create_bert_with_advanced_features(self):
        """Test create_bert_with_advanced_features function."""
        config_dict = create_bert_with_advanced_features(
            size="base", ffn_type="swiglu", use_stochastic_depth=True
        )
        config_dict.update(num_layers=2, hidden_size=256, num_heads=8, vocab_size=1000)
        model = BERT(**config_dict)
        input_ids = ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        sequence_output, pooled_output = model(input_ids, training=False)
        assert sequence_output.shape == (2, 16, config_dict['hidden_size'])

    def test_different_ffn_types(self):
        """Test BERT with different FFN types."""
        base_config = {"vocab_size": 1000, "hidden_size": 128, "num_layers": 2, "num_heads": 8}
        for ffn_type in ["mlp", "swiglu"]:
            config_dict = {**base_config, "ffn_type": ffn_type}
            model = BERT(**config_dict, add_pooling_layer=False)
            input_ids = ops.cast(keras.random.uniform((2, 16), minval=1, maxval=1000), dtype='int32')
            output = model(input_ids, training=False)
            assert output.shape == (2, 16, 128)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])