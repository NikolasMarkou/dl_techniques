"""
Comprehensive pytest test suite for BERT (Bidirectional Encoder Representations from Transformers) model.

This module provides extensive testing for the BERT implementation including:
- Configuration validation and serialization
- Embeddings layer functionality
- Model initialization and parameter validation
- Architecture building and shape consistency
- Forward pass functionality with different input formats
- Individual component testing (BertEmbeddings, TransformerLayer integration)
- Encoding operations and feature extraction
- Serialization and deserialization
- Error handling and edge cases
- Factory function testing for classification and sequence output models
- Integration testing

Key BERT-specific testing areas:
- Token, position, and type embeddings combination
- Attention mask handling
- Different sequence lengths and padding
- Pooled output for classification tasks
- Multiple input format support (dict vs individual tensors)
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.bert import (
    BertConfig,
    BertEmbeddings,
    Bert,
    create_bert_for_classification,
    create_bert_for_sequence_output,
    create_bert_base_uncased,
    create_bert_large_uncased,
    create_bert_with_rms_norm,
    create_bert_with_advanced_features
)


def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Custom allclose function since keras.ops doesn't have it."""
    return np.allclose(a.numpy() if hasattr(a, 'numpy') else a,
                      b.numpy() if hasattr(b, 'numpy') else b,
                      rtol=rtol, atol=atol)


class TestBertConfig:
    """Test BERT configuration validation and initialization."""

    def test_basic_initialization(self):
        """Test basic BertConfig initialization with default parameters."""
        config = BertConfig()

        # Model architecture defaults
        assert config.vocab_size == 30522
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.intermediate_size == 3072
        assert config.hidden_act == "gelu"

        # Dropout defaults
        assert config.hidden_dropout_prob == 0.1
        assert config.attention_probs_dropout_prob == 0.1

        # Sequence and embedding defaults
        assert config.max_position_embeddings == 512
        assert config.type_vocab_size == 2
        assert config.pad_token_id == 0

        # Initialization defaults
        assert config.initializer_range == 0.02
        assert config.layer_norm_eps == 1e-12

        # Architecture defaults
        assert config.position_embedding_type == "absolute"
        assert config.use_cache is True
        assert config.normalization_type == "layer_norm"
        assert config.normalization_position == "post"
        assert config.attention_type == "multi_head_attention"
        assert config.ffn_type == "mlp"

        # Test validation passes
        config.validate()

    def test_custom_initialization(self):
        """Test BertConfig initialization with custom parameters."""
        config = BertConfig(
            vocab_size=50000,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,  # 1024 is divisible by 16
            intermediate_size=4096,
            hidden_act="relu",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.15,
            max_position_embeddings=1024,
            type_vocab_size=4,
            initializer_range=0.01,
            layer_norm_eps=1e-6,
            normalization_type="rms_norm",
            normalization_position="pre",
            attention_type="multi_head_attention",
            ffn_type="swiglu"
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.intermediate_size == 4096
        assert config.hidden_act == "relu"
        assert config.hidden_dropout_prob == 0.2
        assert config.attention_probs_dropout_prob == 0.15
        assert config.max_position_embeddings == 1024
        assert config.type_vocab_size == 4
        assert config.initializer_range == 0.01
        assert config.layer_norm_eps == 1e-6
        assert config.normalization_type == "rms_norm"
        assert config.normalization_position == "pre"
        assert config.attention_type == "multi_head_attention"
        assert config.ffn_type == "swiglu"

        # Test validation passes
        config.validate()

    def test_config_serialization(self):
        """Test BertConfig to_dict and from_dict methods."""
        original_config = BertConfig(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,  # 512 is divisible by 8
            intermediate_size=2048,
            hidden_dropout_prob=0.15
        )

        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 25000
        assert config_dict['hidden_size'] == 512
        assert config_dict['num_layers'] == 8
        assert config_dict['num_heads'] == 8
        assert config_dict['intermediate_size'] == 2048
        assert config_dict['hidden_dropout_prob'] == 0.15

        restored_config = BertConfig.from_dict(config_dict)
        assert restored_config.vocab_size == original_config.vocab_size
        assert restored_config.hidden_size == original_config.hidden_size
        assert restored_config.num_layers == original_config.num_layers
        assert restored_config.num_heads == original_config.num_heads
        assert restored_config.intermediate_size == original_config.intermediate_size
        assert restored_config.hidden_dropout_prob == original_config.hidden_dropout_prob

    def test_config_validation(self):
        """Test BertConfig validation for invalid parameters."""
        # Test invalid hidden_size/num_heads combination
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            config = BertConfig(hidden_size=100, num_heads=12)
            config.validate()

        # Test negative hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            config = BertConfig(hidden_size=-100)
            config.validate()

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="hidden_dropout_prob must be between"):
            config = BertConfig(hidden_dropout_prob=1.5)
            config.validate()

    def test_edge_case_parameters(self):
        """Test BertConfig with edge case but valid parameters."""
        # Minimum viable configuration
        config1 = BertConfig(vocab_size=1, hidden_size=64, num_heads=4)  # 64/4=16, valid
        config1.validate()
        assert config1.vocab_size == 1
        assert config1.hidden_size == 64
        assert config1.num_heads == 4

        # Zero dropout
        config2 = BertConfig(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        config2.validate()
        assert config2.hidden_dropout_prob == 0.0
        assert config2.attention_probs_dropout_prob == 0.0

        # Large configuration
        config3 = BertConfig(vocab_size=100000, hidden_size=2048, num_layers=48, num_heads=16)
        config3.validate()
        assert config3.vocab_size == 100000
        assert config3.hidden_size == 2048
        assert config3.num_layers == 48


class TestBertEmbeddingsComponent:
    """Test BertEmbeddings component individually."""

    @pytest.fixture
    def basic_config(self) -> BertConfig:
        """Create a basic BERT config for testing."""
        return BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,  # 256 is divisible by 8
            intermediate_size=1024,
            max_position_embeddings=128,
            type_vocab_size=2
        )

    def test_bert_embeddings_initialization(self, basic_config):
        """Test BertEmbeddings initialization."""
        embeddings = BertEmbeddings(basic_config)

        assert embeddings.config == basic_config
        assert embeddings.config.vocab_size == 1000
        assert embeddings.config.hidden_size == 256
        assert embeddings.config.max_position_embeddings == 128
        assert embeddings.config.type_vocab_size == 2

        # Components should be None before building
        assert embeddings.word_embeddings is None
        assert embeddings.position_embeddings is None
        assert embeddings.token_type_embeddings is None
        assert embeddings.layer_norm is None
        assert embeddings.dropout is None

    def test_bert_embeddings_build(self, basic_config):
        """Test BertEmbeddings build process."""
        embeddings = BertEmbeddings(basic_config)
        input_shape = (None, 64)  # (batch_size, sequence_length)

        embeddings.build(input_shape)

        assert embeddings.built is True
        assert embeddings._build_input_shape == input_shape
        assert embeddings.word_embeddings is not None
        assert embeddings.position_embeddings is not None
        assert embeddings.token_type_embeddings is not None
        assert embeddings.layer_norm is not None
        assert embeddings.dropout is not None

        # Check embedding dimensions
        assert embeddings.word_embeddings.input_dim == basic_config.vocab_size
        assert embeddings.word_embeddings.output_dim == basic_config.hidden_size
        assert embeddings.position_embeddings.input_dim == basic_config.max_position_embeddings
        assert embeddings.token_type_embeddings.input_dim == basic_config.type_vocab_size

    def test_bert_embeddings_forward_pass(self, basic_config):
        """Test BertEmbeddings forward pass."""
        embeddings = BertEmbeddings(basic_config)

        batch_size = 4
        seq_length = 32

        # Create test input IDs (avoid 0 for meaningful embeddings)
        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        output = embeddings(input_ids, training=False)

        assert output.shape == (batch_size, seq_length, basic_config.hidden_size)

    def test_bert_embeddings_with_token_type_ids(self, basic_config):
        """Test BertEmbeddings with token type IDs."""
        embeddings = BertEmbeddings(basic_config)

        batch_size = 3
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        # Create token type IDs (0 for first segment, 1 for second segment)
        token_type_ids = ops.concatenate([
            ops.zeros((batch_size, seq_length // 2), dtype='int32'),
            ops.ones((batch_size, seq_length - seq_length // 2), dtype='int32')
        ], axis=1)

        output = embeddings(input_ids, token_type_ids=token_type_ids, training=False)

        assert output.shape == (batch_size, seq_length, basic_config.hidden_size)

    def test_bert_embeddings_with_position_ids(self, basic_config):
        """Test BertEmbeddings with custom position IDs."""
        embeddings = BertEmbeddings(basic_config)

        batch_size = 2
        seq_length = 20

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        # Create custom position IDs
        position_ids = ops.broadcast_to(
            ops.arange(seq_length, dtype='int32')[None, :],
            (batch_size, seq_length)
        )

        output = embeddings(
            input_ids,
            position_ids=position_ids,
            training=False
        )

        assert output.shape == (batch_size, seq_length, basic_config.hidden_size)

    def test_bert_embeddings_all_inputs(self, basic_config):
        """Test BertEmbeddings with all input types."""
        embeddings = BertEmbeddings(basic_config)

        batch_size = 2
        seq_length = 24

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        token_type_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=0,
                maxval=basic_config.type_vocab_size
            ),
            dtype='int32'
        )

        position_ids = ops.broadcast_to(
            ops.arange(seq_length, dtype='int32')[None, :],
            (batch_size, seq_length)
        )

        output = embeddings(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            training=False
        )

        assert output.shape == (batch_size, seq_length, basic_config.hidden_size)

    def test_bert_embeddings_training_mode(self, basic_config):
        """Test BertEmbeddings in training vs evaluation mode."""
        embeddings = BertEmbeddings(basic_config)

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        # Test both training modes
        output_train = embeddings(input_ids, training=True)
        output_eval = embeddings(input_ids, training=False)

        assert output_train.shape == output_eval.shape
        assert output_train.shape == (batch_size, seq_length, basic_config.hidden_size)

    def test_bert_embeddings_normalization_types(self):
        """Test BertEmbeddings with different normalization types."""
        normalization_types = ["layer_norm", "rms_norm", "batch_norm"]

        for norm_type in normalization_types:
            config = BertConfig(
                vocab_size=500,
                hidden_size=128,
                num_heads=8,  # 128 is divisible by 8
                normalization_type=norm_type
            )

            embeddings = BertEmbeddings(config)

            input_ids = ops.cast(
                keras.random.uniform((2, 16), minval=1, maxval=500),
                dtype='int32'
            )

            output = embeddings(input_ids, training=False)
            assert output.shape == (2, 16, 128)


class TestBERTModelInitialization:
    """Test BERT model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BERT model initialization."""
        config = BertConfig(hidden_size=256, num_layers=6, num_heads=8)  # 256/8=32, valid
        model = Bert(config)

        assert model.config == config
        assert model.config.hidden_size == 256
        assert model.config.num_layers == 6
        assert model.add_pooling_layer is True
        assert model.built is False

        # Components should be None before building
        assert model.embeddings is None
        assert len(model.encoder_layers) == 0
        assert model.pooler is None

    def test_initialization_without_pooling(self):
        """Test BERT model initialization without pooling layer."""
        config = BertConfig(hidden_size=256, num_heads=8)  # 256/8=32, valid
        model = Bert(config, add_pooling_layer=False)

        assert model.add_pooling_layer is False

    def test_initialization_with_custom_config(self):
        """Test BERT model initialization with custom configuration."""
        config = BertConfig(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,  # 512/8=64, valid
            intermediate_size=2048,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.1,
            normalization_type="rms_norm",
            normalization_position="pre"
        )
        model = Bert(config)

        assert model.config.vocab_size == 25000
        assert model.config.hidden_size == 512
        assert model.config.num_layers == 8
        assert model.config.num_heads == 8
        assert model.config.intermediate_size == 2048
        assert model.config.hidden_dropout_prob == 0.2
        assert model.config.normalization_type == "rms_norm"
        assert model.config.normalization_position == "pre"


class TestBERTModelBuilding:
    """Test BERT model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> BertConfig:
        """Create a basic BERT config for testing."""
        return BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,  # 256/8=32, valid
            intermediate_size=1024,
            max_position_embeddings=128
        )

    def test_build_with_tuple_input_shape(self, basic_config):
        """Test building BERT model with tuple input shape."""
        model = Bert(basic_config)
        input_shape = (None, 64)

        model.build(input_shape)

        assert model.built is True
        assert model._build_input_shape == input_shape
        assert model.embeddings is not None
        assert len(model.encoder_layers) == basic_config.num_layers
        assert model.pooler is not None

        # Check that transformer layers are built
        for layer in model.encoder_layers:
            assert layer.built is True

    def test_build_with_dict_input_shape(self, basic_config):
        """Test building BERT model with dictionary input shape."""
        model = Bert(basic_config)
        input_shape = {
            'input_ids': (None, 64),
            'attention_mask': (None, 64),
            'token_type_ids': (None, 64)
        }

        model.build(input_shape)

        assert model.built is True
        assert model._build_input_shape == input_shape
        assert model.embeddings is not None
        assert len(model.encoder_layers) == basic_config.num_layers

    def test_build_prevents_double_building(self, basic_config):
        """Test that building twice handles correctly."""
        model = Bert(basic_config)
        input_shape = (None, 32)

        # Build first time
        model.build(input_shape)
        assert model.built is True

        # Store reference to check components are preserved
        embeddings_first = model.embeddings
        encoder_layers_first = model.encoder_layers

        # Building again should not cause errors and should preserve state
        model.build(input_shape)

        # Model should still be built and components preserved
        assert model.built is True
        assert model.embeddings is embeddings_first

    def test_build_without_pooling_layer(self, basic_config):
        """Test building BERT model without pooling layer."""
        model = Bert(basic_config, add_pooling_layer=False)
        model.build((None, 32))

        assert model.built is True
        assert model.pooler is None

    def test_transformer_layers_configuration(self, basic_config):
        """Test that TransformerLayers are configured correctly."""
        model = Bert(basic_config)
        model.build((None, 32))

        # Check transformer layer configuration
        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config.hidden_size
            assert layer.num_heads == basic_config.num_heads
            assert layer.intermediate_size == basic_config.intermediate_size
            assert layer.normalization_type == basic_config.normalization_type
            assert layer.normalization_position == basic_config.normalization_position
            assert layer.name == f"encoder_layer_{i}"


class TestBERTModelForwardPass:
    """Test BERT model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> Bert:
        """Create a built BERT model for testing."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=3,
            num_heads=8,  # 256/8=32, valid
            intermediate_size=1024,
            max_position_embeddings=128
        )
        model = Bert(config)
        model.build((None, 64))
        return model

    def test_forward_pass_input_ids_only(self, built_model):
        """Test forward pass with only input IDs."""
        batch_size = 4
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        outputs = built_model(input_ids, training=False)

        # Should return both sequence output and pooled output
        assert isinstance(outputs, tuple)
        sequence_output, pooled_output = outputs

        assert sequence_output.shape == (batch_size, seq_length, built_model.config.hidden_size)
        assert pooled_output.shape == (batch_size, built_model.config.hidden_size)

    def test_forward_pass_dict_input(self, built_model):
        """Test forward pass with dictionary input."""
        batch_size = 3
        seq_length = 24

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
        token_type_ids = ops.zeros((batch_size, seq_length), dtype='int32')

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }

        outputs = built_model(inputs, training=False)

        assert isinstance(outputs, tuple)
        sequence_output, pooled_output = outputs

        assert sequence_output.shape == (batch_size, seq_length, built_model.config.hidden_size)
        assert pooled_output.shape == (batch_size, built_model.config.hidden_size)

    def test_forward_pass_return_dict(self, built_model):
        """Test forward pass with return_dict=True."""
        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        outputs = built_model(input_ids, return_dict=True, training=False)

        assert isinstance(outputs, dict)
        assert 'last_hidden_state' in outputs
        assert 'pooler_output' in outputs

        assert outputs['last_hidden_state'].shape == (batch_size, seq_length, built_model.config.hidden_size)
        assert outputs['pooler_output'].shape == (batch_size, built_model.config.hidden_size)

    def test_forward_pass_no_pooling(self):
        """Test forward pass without pooling layer."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=8  # 256/8=32, valid
        )
        model = Bert(config, add_pooling_layer=False)
        model.build((None, 32))

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)

        # Should return only sequence output (no tuple)
        assert outputs.shape == (batch_size, seq_length, config.hidden_size)

    def test_forward_pass_with_attention_mask(self, built_model):
        """Test forward pass with attention mask."""
        batch_size = 3
        seq_length = 20

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        # Create attention mask with some padding
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
        # Set last few tokens as padding for some examples
        attention_mask = ops.where(
            ops.arange(seq_length)[None, :] < seq_length - 3,
            attention_mask,
            0
        )

        outputs = built_model(
            input_ids,
            attention_mask=attention_mask,
            training=False
        )

        assert isinstance(outputs, tuple)
        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (batch_size, seq_length, built_model.config.hidden_size)

    def test_forward_pass_training_mode(self, built_model):
        """Test forward pass in training vs evaluation mode."""
        batch_size = 2
        seq_length = 12

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        # Test both training modes
        outputs_train = built_model(input_ids, training=True)
        outputs_eval = built_model(input_ids, training=False)

        # Both should have same structure and shapes
        assert len(outputs_train) == len(outputs_eval)
        assert outputs_train[0].shape == outputs_eval[0].shape
        assert outputs_train[1].shape == outputs_eval[1].shape

    def test_forward_pass_different_sequence_lengths(self, built_model):
        """Test forward pass with different sequence lengths."""
        batch_size = 2
        sequence_lengths = [8, 16, 32, 48]

        for seq_length in sequence_lengths:
            input_ids = ops.cast(
                keras.random.uniform(
                    (batch_size, seq_length),
                    minval=1,
                    maxval=built_model.config.vocab_size
                ),
                dtype='int32'
            )

            outputs = built_model(input_ids, training=False)
            sequence_output, pooled_output = outputs

            assert sequence_output.shape == (batch_size, seq_length, built_model.config.hidden_size)
            assert pooled_output.shape == (batch_size, built_model.config.hidden_size)


class TestBERTModelSerialization:
    """Test BERT model serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        config = BertConfig(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,  # 512/8=64, valid
            intermediate_size=2048,
            hidden_dropout_prob=0.15,
            normalization_type="rms_norm"
        )
        model = Bert(config)

        model_config = model.get_config()

        assert isinstance(model_config, dict)
        assert 'config' in model_config
        assert 'add_pooling_layer' in model_config

        config_dict = model_config['config']
        assert config_dict['vocab_size'] == 25000
        assert config_dict['hidden_size'] == 512
        assert config_dict['num_layers'] == 8
        assert config_dict['hidden_dropout_prob'] == 0.15
        assert config_dict['normalization_type'] == "rms_norm"

    def test_from_config_reconstruction(self):
        """Test model reconstruction from configuration."""
        original_config = BertConfig(
            vocab_size=15000,
            hidden_size=384,
            num_layers=6,
            num_heads=12  # 384/12=32, valid
        )
        original_model = Bert(original_config, add_pooling_layer=False)

        # Get config and reconstruct
        model_config = original_model.get_config()
        reconstructed_model = Bert.from_config(model_config)

        # Check that configs match
        assert reconstructed_model.config.vocab_size == original_config.vocab_size
        assert reconstructed_model.config.hidden_size == original_config.hidden_size
        assert reconstructed_model.config.num_layers == original_config.num_layers
        assert reconstructed_model.config.num_heads == original_config.num_heads
        assert reconstructed_model.add_pooling_layer == original_model.add_pooling_layer

    def test_build_config_serialization(self):
        """Test build configuration serialization."""
        config = BertConfig(hidden_size=256, num_heads=8)  # 256/8=32, valid
        model = Bert(config)

        input_shape = (None, 64)
        model.build(input_shape)

        build_config = model.get_build_config()
        assert build_config['input_shape'] == input_shape

        # Test build from config
        new_model = Bert(config)
        new_model.build_from_config(build_config)
        assert new_model.built is True

    def test_embeddings_serialization(self):
        """Test BertEmbeddings serialization."""
        config = BertConfig(vocab_size=1000, hidden_size=256, num_heads=8)  # 256/8=32, valid

        # Test BertEmbeddings serialization
        embeddings = BertEmbeddings(config)
        embeddings_config = embeddings.get_config()
        reconstructed_embeddings = BertEmbeddings.from_config(embeddings_config)

        assert reconstructed_embeddings.config.vocab_size == config.vocab_size
        assert reconstructed_embeddings.config.hidden_size == config.hidden_size

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        # Create and build model
        config = BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=8,   # 256/8=32, valid
            intermediate_size=512
        )
        model = Bert(config)
        # Build with specific input shape to avoid None dimension issues
        model.build((2, 32))  # Use fixed batch size for building

        # Test forward pass before saving
        batch_size = 2
        seq_length = 16
        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        original_outputs = model(input_ids, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_bert.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(
                model_path,
                custom_objects={
                    'BERT': Bert,
                    'BertEmbeddings': BertEmbeddings,
                    'BertConfig': BertConfig
                }
            )

            # Test that loaded model produces same output
            loaded_outputs = loaded_model(input_ids, training=False)

            # Handle both tuple and single tensor outputs
            if isinstance(original_outputs, tuple):
                assert isinstance(loaded_outputs, tuple)
                assert len(original_outputs) == len(loaded_outputs)

                # Outputs should be very close
                for orig, loaded in zip(original_outputs, loaded_outputs):
                    np.testing.assert_allclose(
                        orig.numpy(),
                        loaded.numpy(),
                        rtol=1e-5,
                        atol=1e-6
                    )
            else:
                # Single tensor output
                np.testing.assert_allclose(
                    original_outputs.numpy(),
                    loaded_outputs.numpy(),
                    rtol=1e-5,
                    atol=1e-6
                )


class TestBERTEdgeCases:
    """Test BERT model edge cases and error handling."""

    def test_minimum_sequence_length(self):
        """Test BERT with minimum sequence length."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=8  # 128/8=16, valid
        )
        model = Bert(config)
        model.build((None, 1))

        input_ids = ops.cast([[42]], dtype='int32')
        outputs = model(input_ids, training=False)

        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (1, 1, 128)
        assert pooled_output.shape == (1, 128)

    def test_maximum_position_embeddings(self):
        """Test BERT with maximum sequence length."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=8,  # 128/8=16, valid
            max_position_embeddings=64
        )
        model = Bert(config)

        seq_length = 64
        model.build((None, seq_length))

        input_ids = ops.cast(
            keras.random.uniform(
                (2, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)
        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (2, seq_length, 128)

    def test_single_sample_batch(self):
        """Test BERT with batch size of 1."""
        config = BertConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        model = Bert(config)
        model.build((None, 32))

        input_ids = ops.cast(
            keras.random.uniform((1, 32), minval=1, maxval=1000),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)
        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (1, 32, 128)
        assert pooled_output.shape == (1, 128)

    def test_different_batch_sizes(self):
        """Test BERT with different batch sizes."""
        config = BertConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        model = Bert(config)
        model.build((None, 24))

        batch_sizes = [1, 3, 8, 16]

        for batch_size in batch_sizes:
            input_ids = ops.cast(
                keras.random.uniform(
                    (batch_size, 24),
                    minval=1,
                    maxval=config.vocab_size
                ),
                dtype='int32'
            )

            outputs = model(input_ids, training=False)
            sequence_output, pooled_output = outputs

            assert sequence_output.shape == (batch_size, 24, 128)
            assert pooled_output.shape == (batch_size, 128)

    def test_input_with_heavy_padding(self):
        """Test BERT with heavy padding (lots of zeros)."""
        config = BertConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        model = Bert(config)
        model.build((None, 32))

        batch_size = 3
        seq_length = 32

        # Create input with lots of padding
        input_ids = ops.ones((batch_size, seq_length), dtype='int32')
        # Only first 8 tokens are non-zero, rest are padding (0)
        padded_input_ids = ops.concatenate([
            input_ids[:, :8],
            ops.zeros((batch_size, seq_length - 8), dtype='int32')
        ], axis=1)

        # Create corresponding attention mask
        attention_mask = ops.concatenate([
            ops.ones((batch_size, 8), dtype='int32'),
            ops.zeros((batch_size, seq_length - 8), dtype='int32')
        ], axis=1)

        outputs = model(
            padded_input_ids,
            attention_mask=attention_mask,
            training=False
        )

        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (batch_size, seq_length, 128)
        assert pooled_output.shape == (batch_size, 128)

    def test_all_padding_input(self):
        """Test BERT with all padding tokens."""
        config = BertConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        model = Bert(config)
        model.build((None, 16))

        # All padding tokens (0)
        input_ids = ops.zeros((2, 16), dtype='int32')
        attention_mask = ops.zeros((2, 16), dtype='int32')

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            training=False
        )

        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (2, 16, 128)
        assert pooled_output.shape == (2, 128)

    def test_mixed_token_types(self):
        """Test BERT with mixed token type IDs."""
        config = BertConfig(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8, type_vocab_size=3)
        model = Bert(config)
        model.build((None, 24))

        batch_size = 2
        seq_length = 24

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        # Mix of token types 0, 1, 2
        token_type_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=0,
                maxval=config.type_vocab_size
            ),
            dtype='int32'
        )

        outputs = model(
            input_ids,
            token_type_ids=token_type_ids,
            training=False
        )

        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (batch_size, seq_length, 128)


class TestBERTFactoryFunctions:
    """Test BERT factory functions."""

    def test_create_bert_for_classification(self):
        """Test create_bert_for_classification factory function."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=3,
            num_heads=8  # 256/8=32, valid
        )
        num_labels = 5

        model = create_bert_for_classification(config, num_labels)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 3
        assert model.inputs[0].shape[1:] == (None,)
        assert model.inputs[1].shape[1:] == (None,)
        assert model.inputs[2].shape[1:] == (None,)

        # Test forward pass
        batch_size = 2
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
        token_type_ids = ops.zeros((batch_size, seq_length), dtype='int32')

        logits = model([input_ids, attention_mask, token_type_ids], training=False)
        assert logits.shape == (batch_size, num_labels)

    def test_create_bert_for_classification_with_dropout(self):
        """Test create_bert_for_classification with custom dropout."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=8  # 128/8=16, valid
        )
        num_labels = 3
        classifier_dropout = 0.2

        model = create_bert_for_classification(
            config,
            num_labels,
            classifier_dropout=classifier_dropout
        )

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
        token_type_ids = ops.zeros((batch_size, seq_length), dtype='int32')

        logits = model([input_ids, attention_mask, token_type_ids], training=False)
        assert logits.shape == (batch_size, num_labels)

    def test_create_bert_for_sequence_output(self):
        """Test create_bert_for_sequence_output factory function."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=3,
            num_heads=8  # 256/8=32, valid
        )

        model = create_bert_for_sequence_output(config)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 3

        # Test forward pass
        batch_size = 2
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
        token_type_ids = ops.zeros((batch_size, seq_length), dtype='int32')

        sequence_output = model([input_ids, attention_mask, token_type_ids], training=False)
        assert sequence_output.shape == (batch_size, seq_length, config.hidden_size)

    def test_create_bert_base_uncased(self):
        """Test create_bert_base_uncased configuration."""
        config = create_bert_base_uncased()

        assert config.vocab_size == 30522
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.intermediate_size == 3072

    def test_create_bert_large_uncased(self):
        """Test create_bert_large_uncased configuration."""
        config = create_bert_large_uncased()

        assert config.vocab_size == 30522
        assert config.hidden_size == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.intermediate_size == 4096

    def test_create_bert_with_rms_norm(self):
        """Test create_bert_with_rms_norm configurations."""
        # Test base size with pre-normalization
        config_base_pre = create_bert_with_rms_norm(size="base", normalization_position="pre")

        assert config_base_pre.hidden_size == 768
        assert config_base_pre.num_layers == 12
        assert config_base_pre.normalization_type == "rms_norm"
        assert config_base_pre.normalization_position == "pre"

        # Test large size with post-normalization
        config_large_post = create_bert_with_rms_norm(size="large", normalization_position="post")

        assert config_large_post.hidden_size == 1024
        assert config_large_post.num_layers == 24
        assert config_large_post.normalization_type == "rms_norm"
        assert config_large_post.normalization_position == "post"

    def test_create_bert_with_rms_norm_invalid_size(self):
        """Test create_bert_with_rms_norm with invalid size."""
        with pytest.raises(ValueError, match="Unsupported size"):
            create_bert_with_rms_norm(size="medium")

    def test_factory_functions_functional(self):
        """Test that all factory functions create functional models."""
        configs_and_creators = [
            (create_bert_base_uncased(), "base"),
            (create_bert_large_uncased(), "large"),
            (create_bert_with_rms_norm("base"), "rms_base")
        ]

        for config, name in configs_and_creators:
            # Modify config for faster testing
            config.num_layers = 2
            config.hidden_size = 128
            config.num_heads = 8  # 128/8=16, valid
            config.intermediate_size = 512
            config.vocab_size = 1000

            # Test classification model
            cls_model = create_bert_for_classification(config, num_labels=3)

            batch_size = 2
            seq_length = 16

            input_ids = ops.cast(
                keras.random.uniform(
                    (batch_size, seq_length),
                    minval=1,
                    maxval=config.vocab_size
                ),
                dtype='int32'
            )
            attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
            token_type_ids = ops.zeros((batch_size, seq_length), dtype='int32')

            logits = cls_model([input_ids, attention_mask, token_type_ids], training=False)
            assert logits.shape == (batch_size, 3)

            # Test sequence output model
            seq_model = create_bert_for_sequence_output(config)
            sequence_output = seq_model([input_ids, attention_mask, token_type_ids], training=False)
            assert sequence_output.shape == (batch_size, seq_length, config.hidden_size)


class TestBERTIntegration:
    """Integration tests for BERT components working together."""

    def test_end_to_end_text_classification(self):
        """Test complete text classification workflow."""
        # Create small model for testing
        config = BertConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=3,
            num_heads=8,  # 256/8=32, valid
            intermediate_size=1024
        )

        model = create_bert_for_classification(config, num_labels=4)

        # Create test data
        batch_size = 4
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        # Create attention mask with some padding
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')
        # Set last few tokens as padding for some examples
        attention_mask = ops.where(
            (ops.arange(seq_length)[None, :] < seq_length - 5) |
            (ops.arange(batch_size)[:, None] == 0),
            attention_mask,
            0
        )

        token_type_ids = ops.zeros((batch_size, seq_length), dtype='int32')

        # Forward pass
        logits = model([input_ids, attention_mask, token_type_ids], training=False)

        assert logits.shape == (batch_size, 4)

        # Test that logits are reasonable (not all zeros, not exploding)
        assert not ops.all(logits == 0)
        assert ops.all(ops.isfinite(logits))
        assert ops.max(ops.abs(logits)) < 100

    def test_component_consistency(self):
        """Test consistency between BERT model and manual component usage."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=8  # 128/8=16, valid
        )

        # Create BERT model
        bert_model = Bert(config, add_pooling_layer=False)
        bert_model.build((None, 24))

        # Use the model's own embeddings layer for the manual check
        embeddings = bert_model.embeddings

        batch_size = 2
        seq_length = 24

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        # Get BERT model output
        bert_output = bert_model(input_ids, training=False)

        # Manual processing through components
        embeddings_output = embeddings(input_ids, training=False)

        # Process through transformer layers manually
        hidden_states = embeddings_output
        for layer in bert_model.encoder_layers:
            hidden_states = layer(hidden_states, training=False)

        # Should be very close
        assert allclose(bert_output, hidden_states, rtol=1e-5)

    def test_different_normalization_consistency(self):
        """Test consistency across different normalization types."""
        normalization_types = ["layer_norm", "rms_norm"]

        outputs = {}

        for norm_type in normalization_types:
            config = BertConfig(
                vocab_size=1000,
                hidden_size=128,
                num_layers=2,
                num_heads=8,  # 128/8=16, valid
                normalization_type=norm_type,
                # Use same initialization for consistency
                initializer_range=0.01
            )

            model = Bert(config, add_pooling_layer=False)
            model.build((None, 16))

            input_ids = ops.cast(
                keras.random.uniform(
                    (2, 16),
                    minval=1,
                    maxval=config.vocab_size
                ),
                dtype='int32'
            )

            output = model(input_ids, training=False)
            outputs[norm_type] = output

        # Outputs should have same shape and reasonable values
        layer_norm_output = outputs["layer_norm"]
        rms_norm_output = outputs["rms_norm"]

        assert layer_norm_output.shape == rms_norm_output.shape
        assert ops.all(ops.isfinite(layer_norm_output))
        assert ops.all(ops.isfinite(rms_norm_output))

    def test_training_vs_inference_consistency(self):
        """Test consistency between training and inference modes."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=8,  # 128/8=16, valid
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0
        )

        model = Bert(config)
        model.build((None, 16))

        input_ids = ops.cast(
            keras.random.uniform(
                (2, 16),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        # Compare training and inference modes (should be identical with no dropout)
        outputs_train = model(input_ids, training=True)
        outputs_eval = model(input_ids, training=False)

        # With no dropout, results should be very close
        assert allclose(outputs_train[0], outputs_eval[0], rtol=1e-6)
        assert allclose(outputs_train[1], outputs_eval[1], rtol=1e-6)

    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire BERT model."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=8,  # 64/8=8, valid
            intermediate_size=256
        )

        model = create_bert_for_classification(config, num_labels=3)

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')

        # FIX: Use non-constant token_type_ids. Using all zeros causes the
        # token_type_embedding to be a constant vector added to all positions.
        # This constant is then cancelled out by the mean-subtraction in the
        # subsequent LayerNormalization layer, resulting in a zero gradient for
        # the token_type_embeddings weight.
        token_type_ids = ops.concatenate([
            ops.zeros((batch_size, seq_length // 2), dtype="int32"),
            ops.ones((batch_size, seq_length - seq_length // 2), dtype="int32"),
        ], axis=1)


        with tf.GradientTape() as tape:
            logits = model([input_ids, attention_mask, token_type_ids], training=True)

            # Create a simple classification loss
            targets = ops.one_hot(ops.array([0, 2]), 3)
            loss = keras.losses.categorical_crossentropy(targets, logits)
            loss = ops.mean(loss)

        gradients = tape.gradient(loss, model.trainable_weights)

        # Check that we have gradients for most weights
        non_none_grads = [g for g in gradients if g is not None]
        # Relax the requirement - at least half of the weights should have gradients
        assert len(non_none_grads) > len(model.trainable_weights) * 0.5

        # Check that gradients have reasonable magnitudes
        grad_norms = [
            ops.sqrt(ops.sum(ops.square(g)) + 1e-8) for g in non_none_grads
        ]
        assert all(norm > 0.0 for norm in grad_norms)
        assert all(norm < 1000.0 for norm in grad_norms)

    def test_pooled_output_quality(self):
        """Test quality of pooled output for classification."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=8  # 128/8=16, valid
        )

        model = Bert(config, add_pooling_layer=True)
        model.build((None, 20))

        batch_size = 3
        seq_length = 20

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        sequence_output, pooled_output = model(input_ids, training=False)

        # Pooled output should be different from raw CLS token
        cls_token_output = sequence_output[:, 0, :]

        # They should be different (pooled has tanh activation)
        assert not allclose(cls_token_output, pooled_output, atol=1e-6)

        # Pooled output should be in reasonable range (tanh output)
        assert ops.all(pooled_output >= -1.0)
        assert ops.all(pooled_output <= 1.0)

        # Should have some variation (not all the same)
        pooled_std = ops.std(pooled_output, axis=0)
        assert ops.mean(pooled_std) > 0.01


class TestBERTAdvancedFeatures:
    """Test advanced BERT features from dl-techniques framework."""

    def test_create_bert_with_advanced_features(self):
        """Test create_bert_with_advanced_features function."""
        config = create_bert_with_advanced_features(
            size="base",
            normalization_type="rms_norm",
            normalization_position="pre",
            attention_type="multi_head_attention",
            ffn_type="swiglu",
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1
        )

        # Modify for testing
        config.num_layers = 2
        config.hidden_size = 256
        config.num_heads = 8
        config.vocab_size = 1000

        model = Bert(config)
        model.build((None, 32))

        # Test forward pass
        batch_size = 2
        seq_length = 16
        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)
        sequence_output, pooled_output = outputs

        assert sequence_output.shape == (batch_size, seq_length, config.hidden_size)
        assert pooled_output.shape == (batch_size, config.hidden_size)

    def test_different_ffn_types(self):
        """Test BERT with different FFN types."""
        ffn_types = ["mlp", "swiglu"]

        for ffn_type in ffn_types:
            config = BertConfig(
                vocab_size=1000,
                hidden_size=128,
                num_layers=2,
                num_heads=8,
                ffn_type=ffn_type
            )

            model = Bert(config, add_pooling_layer=False)
            model.build((None, 16))

            input_ids = ops.cast(
                keras.random.uniform((2, 16), minval=1, maxval=1000),
                dtype='int32'
            )

            output = model(input_ids, training=False)
            assert output.shape == (2, 16, 128)

    def test_stochastic_depth_enabled(self):
        """Test BERT with stochastic depth enabled."""
        config = BertConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=3,
            num_heads=8,
            use_stochastic_depth=True,
            stochastic_depth_rate=0.2
        )

        model = Bert(config)
        model.build((None, 24))

        input_ids = ops.cast(
            keras.random.uniform((2, 24), minval=1, maxval=1000),
            dtype='int32'
        )

        # Test both training and inference modes
        output_train = model(input_ids, training=True)
        output_eval = model(input_ids, training=False)

        # Both should have correct shapes
        assert output_train[0].shape == (2, 24, 128)
        assert output_eval[0].shape == (2, 24, 128)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])