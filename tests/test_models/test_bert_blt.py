"""
Comprehensive pytest test suite for BertBlt (BERT with Byte Latent Transformer features).

This module provides extensive testing for the BertBlt implementation including:
- Configuration validation and serialization
- ByteTokenizer functionality and text processing
- HashNGramEmbedding component testing
- BertBltEmbeddings layer functionality with individual parameters
- Model initialization and parameter validation
- Architecture building and shape consistency
- Forward pass functionality with different input formats
- Byte-level processing and text encoding/decoding
- Hash embedding integration
- Serialization and deserialization
- Error handling and edge cases
- Factory function testing for classification and sequence output models
- Integration testing with byte-level features

Key features tested:
- Byte-level tokenization and processing
- Hash n-gram embeddings for enhanced representations
- Text encoding and decoding functionality
- Language-agnostic processing capabilities
- Integration with existing TransformerLayer components
"""

import pytest
import numpy as np
import keras
from keras import ops
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any, List

from dl_techniques.models.bert_blt import (
    BertBltConfig,
    ByteTokenizer,
    HashNGramEmbedding,
    BertBltEmbeddings,
    BertBlt,
    create_bert_blt_base,
    create_bert_blt_large,
    create_bert_blt_for_classification,
    create_bert_blt_for_sequence_output,
    create_robust_bert_blt
)


def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Custom allclose function since keras.ops doesn't have it."""
    return np.allclose(a.numpy() if hasattr(a, 'numpy') else a,
                       b.numpy() if hasattr(b, 'numpy') else b,
                       rtol=rtol, atol=atol)


class TestBertBltConfig:
    """Test BertBlt configuration validation and initialization."""

    def test_basic_initialization(self):
        """Test BertBltConfig initialization with default parameters."""
        config = BertBltConfig()

        # Model architecture defaults (byte-level)
        assert config.vocab_size == 260  # 256 bytes + special tokens
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.intermediate_size == 3072
        assert config.hidden_act == "gelu"

        # Dropout defaults
        assert config.hidden_dropout_prob == 0.1
        assert config.attention_probs_dropout_prob == 0.1

        # Sequence defaults (larger for byte-level)
        assert config.max_position_embeddings == 2048
        assert config.initializer_range == 0.02
        assert config.layer_norm_eps == 1e-12

        # BLT-specific defaults
        assert config.use_hash_embeddings is True
        assert config.hash_vocab_size == 500000
        assert config.ngram_sizes == [3, 4, 5, 6, 7, 8]  # After __post_init__
        assert config.hash_embedding_dim == 768  # After __post_init__

        # Architecture defaults
        assert config.normalization_type == "layer_norm"
        assert config.normalization_position == "post"
        assert config.attention_type == "multi_head_attention"
        assert config.ffn_type == "mlp"
        assert config.use_stochastic_depth is False

        # Test validation passes
        config.validate()

    def test_custom_initialization(self):
        """Test BertBltConfig initialization with custom parameters."""
        config = BertBltConfig(
            vocab_size=300,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,  # 1024 is divisible by 16
            intermediate_size=4096,
            hidden_act="relu",
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.15,
            max_position_embeddings=4096,
            initializer_range=0.01,
            layer_norm_eps=1e-6,
            use_hash_embeddings=False,
            hash_vocab_size=1000000,
            ngram_sizes=[2, 3, 4, 5],
            hash_embedding_dim=512,
            normalization_type="rms_norm",
            normalization_position="pre",
            attention_type="multi_head_attention",
            ffn_type="swiglu"
        )

        assert config.vocab_size == 300
        assert config.hidden_size == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.intermediate_size == 4096
        assert config.hidden_act == "relu"
        assert config.hidden_dropout_prob == 0.2
        assert config.attention_probs_dropout_prob == 0.15
        assert config.max_position_embeddings == 4096
        assert config.initializer_range == 0.01
        assert config.layer_norm_eps == 1e-6
        assert config.use_hash_embeddings is False
        assert config.hash_vocab_size == 1000000
        assert config.ngram_sizes == [2, 3, 4, 5]
        assert config.hash_embedding_dim == 512
        assert config.normalization_type == "rms_norm"
        assert config.normalization_position == "pre"
        assert config.attention_type == "multi_head_attention"
        assert config.ffn_type == "swiglu"

        # Test validation passes
        config.validate()

    def test_post_init_defaults(self):
        """Test __post_init__ method sets correct defaults."""
        config = BertBltConfig(hidden_size=512)

        # Should set default ngram_sizes
        assert config.ngram_sizes == [3, 4, 5, 6, 7, 8]
        # Should set hash_embedding_dim to hidden_size
        assert config.hash_embedding_dim == 512

        # Test with custom ngram_sizes
        config2 = BertBltConfig(ngram_sizes=[2, 3, 4])
        assert config2.ngram_sizes == [2, 3, 4]  # Should not override

        # Test with custom hash_embedding_dim
        config3 = BertBltConfig(hash_embedding_dim=256)
        assert config3.hash_embedding_dim == 256  # Should not override

    def test_config_serialization(self):
        """Test BertBltConfig to_dict and from_dict methods."""
        original_config = BertBltConfig(
            vocab_size=300,
            hidden_size=512,
            num_layers=8,
            num_heads=8,  # 512 is divisible by 8
            intermediate_size=2048,
            hidden_dropout_prob=0.15,
            use_hash_embeddings=True,
            hash_vocab_size=250000,
            ngram_sizes=[3, 4, 5]
        )

        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 300
        assert config_dict['hidden_size'] == 512
        assert config_dict['num_layers'] == 8
        assert config_dict['num_heads'] == 8
        assert config_dict['intermediate_size'] == 2048
        assert config_dict['hidden_dropout_prob'] == 0.15
        assert config_dict['use_hash_embeddings'] is True
        assert config_dict['hash_vocab_size'] == 250000
        assert config_dict['ngram_sizes'] == [3, 4, 5]

        restored_config = BertBltConfig.from_dict(config_dict)
        assert restored_config.vocab_size == original_config.vocab_size
        assert restored_config.hidden_size == original_config.hidden_size
        assert restored_config.num_layers == original_config.num_layers
        assert restored_config.num_heads == original_config.num_heads
        assert restored_config.intermediate_size == original_config.intermediate_size
        assert restored_config.hidden_dropout_prob == original_config.hidden_dropout_prob
        assert restored_config.use_hash_embeddings == original_config.use_hash_embeddings
        assert restored_config.hash_vocab_size == original_config.hash_vocab_size
        assert restored_config.ngram_sizes == original_config.ngram_sizes

    def test_config_validation(self):
        """Test BertBltConfig validation for invalid parameters."""
        # Test invalid hidden_size/num_heads combination
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            config = BertBltConfig(hidden_size=100, num_heads=12)
            config.validate()

        # Test negative vocab_size
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            config = BertBltConfig(vocab_size=-100)
            config.validate()

        # Test negative hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            config = BertBltConfig(hidden_size=-100)
            config.validate()

        # Test invalid dropout rates
        with pytest.raises(ValueError, match="hidden_dropout_prob must be between"):
            config = BertBltConfig(hidden_dropout_prob=1.5)
            config.validate()

    def test_edge_case_parameters(self):
        """Test BertBltConfig with edge case but valid parameters."""
        # Minimum viable configuration
        config1 = BertBltConfig(vocab_size=1, hidden_size=64, num_heads=4)  # 64/4=16, valid
        config1.validate()
        assert config1.vocab_size == 1
        assert config1.hidden_size == 64
        assert config1.num_heads == 4

        # Zero dropout
        config2 = BertBltConfig(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        config2.validate()
        assert config2.hidden_dropout_prob == 0.0
        assert config2.attention_probs_dropout_prob == 0.0

        # Disabled hash embeddings
        config3 = BertBltConfig(use_hash_embeddings=False)
        config3.validate()
        assert config3.use_hash_embeddings is False


class TestByteTokenizer:
    """Test ByteTokenizer component functionality."""

    def test_initialization(self):
        """Test ByteTokenizer initialization."""
        tokenizer = ByteTokenizer()

        assert tokenizer.vocab_size == 260
        assert tokenizer.byte_offset == 4
        assert tokenizer.pad_token_id == 0
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2
        assert tokenizer.unk_token_id == 3

    def test_custom_initialization(self):
        """Test ByteTokenizer with custom parameters."""
        tokenizer = ByteTokenizer(vocab_size=300, byte_offset=8)

        assert tokenizer.vocab_size == 300
        assert tokenizer.byte_offset == 8

    def test_text_to_bytes_basic(self):
        """Test basic text to bytes conversion."""
        tokenizer = ByteTokenizer()

        # Simple ASCII text
        text = "hello"
        tokens = tokenizer.text_to_bytes(text, add_bos=False, add_eos=False)

        # Expected: UTF-8 bytes of "hello" + byte_offset
        expected = [ord(c) + tokenizer.byte_offset for c in text.encode('utf-8')]
        assert tokens == expected

    def test_text_to_bytes_with_special_tokens(self):
        """Test text to bytes conversion with special tokens."""
        tokenizer = ByteTokenizer()

        text = "hi"
        tokens = tokenizer.text_to_bytes(text, add_bos=True, add_eos=True)

        expected = [tokenizer.bos_token_id] + [ord(c) + tokenizer.byte_offset for c in text.encode('utf-8')] + [
            tokenizer.eos_token_id]
        assert tokens == expected

    def test_text_to_bytes_unicode(self):
        """Test text to bytes conversion with Unicode characters."""
        tokenizer = ByteTokenizer()

        text = "café"  # Contains non-ASCII character
        tokens = tokenizer.text_to_bytes(text, add_bos=False, add_eos=False)

        # Should handle UTF-8 encoding properly
        utf8_bytes = text.encode('utf-8')
        expected = [b + tokenizer.byte_offset for b in utf8_bytes]
        assert tokens == expected
        assert len(tokens) > len(text)  # More bytes than characters due to UTF-8

    def test_text_to_bytes_empty(self):
        """Test text to bytes conversion with empty text."""
        tokenizer = ByteTokenizer()

        tokens = tokenizer.text_to_bytes("", add_bos=True, add_eos=True)
        assert tokens == [tokenizer.bos_token_id, tokenizer.eos_token_id]

        tokens_no_special = tokenizer.text_to_bytes("", add_bos=False, add_eos=False)
        assert tokens_no_special == []

    def test_tokens_to_text_basic(self):
        """Test basic token to text conversion."""
        tokenizer = ByteTokenizer()

        text = "hello"
        tokens = tokenizer.text_to_bytes(text, add_bos=False, add_eos=False)
        recovered_text = tokenizer.tokens_to_text(tokens)

        assert recovered_text == text

    def test_tokens_to_text_with_special_tokens(self):
        """Test token to text conversion filtering special tokens."""
        tokenizer = ByteTokenizer()

        text = "test"
        tokens = tokenizer.text_to_bytes(text, add_bos=True, add_eos=True)
        recovered_text = tokenizer.tokens_to_text(tokens)

        assert recovered_text == text  # Special tokens should be filtered out

    def test_tokens_to_text_unicode(self):
        """Test token to text conversion with Unicode."""
        tokenizer = ByteTokenizer()

        text = "Hello 世界"  # Mixed ASCII and Unicode
        tokens = tokenizer.text_to_bytes(text, add_bos=False, add_eos=False)
        recovered_text = tokenizer.tokens_to_text(tokens)

        assert recovered_text == text

    def test_round_trip_conversion(self):
        """Test round-trip conversion for various texts."""
        tokenizer = ByteTokenizer()

        test_texts = [
            "Hello, World!",
            "café résumé naïve",
            "こんにちは",  # Japanese
            "🚀🌟✨",  # Emojis
            "Mixed text with émojis 🎉 and numbers 123",
            ""
        ]

        for text in test_texts:
            tokens = tokenizer.text_to_bytes(text, add_bos=False, add_eos=False)
            recovered = tokenizer.tokens_to_text(tokens)
            assert recovered == text, f"Round-trip failed for: {text}"

    def test_tokens_to_text_invalid_tokens(self):
        """Test token to text conversion with invalid tokens."""
        tokenizer = ByteTokenizer()

        # Include some invalid token IDs
        invalid_tokens = [1, 2, 3, 300, 400, 72, 101, 108, 108, 111]  # "hello" with invalid tokens
        result = tokenizer.tokens_to_text(invalid_tokens)

        # Should handle gracefully and return valid parts
        assert isinstance(result, str)

    def test_serialization(self):
        """Test ByteTokenizer serialization."""
        tokenizer = ByteTokenizer(vocab_size=300, byte_offset=8)

        config = tokenizer.get_config()
        assert config['vocab_size'] == 300
        assert config['byte_offset'] == 8

    def test_as_keras_layer(self):
        """Test ByteTokenizer as Keras layer."""
        tokenizer = ByteTokenizer()

        # Should be able to build and call as layer
        input_shape = (None, 32)
        tokenizer.build(input_shape)
        assert tokenizer.built


class TestHashNGramEmbedding:
    """Test HashNGramEmbedding component functionality."""

    def test_initialization(self):
        """Test HashNGramEmbedding initialization."""
        embed = HashNGramEmbedding(
            hash_vocab_size=1000,
            embed_dim=128,
            ngram_sizes=[3, 4, 5]
        )

        assert embed.hash_vocab_size == 1000
        assert embed.embed_dim == 128
        assert embed.ngram_sizes == [3, 4, 5]

    def test_parameter_validation(self):
        """Test HashNGramEmbedding parameter validation."""
        # Test invalid hash_vocab_size
        with pytest.raises(ValueError, match="hash_vocab_size must be positive"):
            HashNGramEmbedding(hash_vocab_size=-1, embed_dim=128)

        # Test invalid embed_dim
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            HashNGramEmbedding(hash_vocab_size=1000, embed_dim=0)

        # Test invalid ngram_sizes
        with pytest.raises(ValueError, match="ngram_sizes must be non-empty"):
            HashNGramEmbedding(hash_vocab_size=1000, embed_dim=128, ngram_sizes=[])

        with pytest.raises(ValueError, match="ngram_sizes must be non-empty"):
            HashNGramEmbedding(hash_vocab_size=1000, embed_dim=128, ngram_sizes=[-1, 2])

    def test_build(self):
        """Test HashNGramEmbedding build process."""
        embed = HashNGramEmbedding(
            hash_vocab_size=1000,
            embed_dim=64,
            ngram_sizes=[3, 4]
        )

        input_shape = (None, 32)
        embed.build(input_shape)

        assert embed.built
        assert len(embed.hash_embeddings) == 2  # For 3-gram and 4-gram
        assert "3" in embed.hash_embeddings
        assert "4" in embed.hash_embeddings

    def test_forward_pass(self):
        """Test HashNGramEmbedding forward pass."""
        embed = HashNGramEmbedding(
            hash_vocab_size=1000,
            embed_dim=64,
            ngram_sizes=[3, 4, 5]
        )

        batch_size = 2
        seq_len = 16

        # Create byte token inputs (values 0-255 + offset)
        inputs = ops.cast(
            keras.random.uniform(
                (batch_size, seq_len),
                minval=4,  # byte_offset
                maxval=260  # vocab_size
            ),
            dtype='int32'
        )

        output = embed(inputs)

        assert output.shape == (batch_size, seq_len, 64)
        assert ops.all(ops.isfinite(output))

    def test_different_ngram_sizes(self):
        """Test HashNGramEmbedding with different n-gram sizes."""
        ngram_configs = [
            [2, 3],
            [3, 4, 5, 6],
            [8]  # Single large n-gram
        ]

        for ngram_sizes in ngram_configs:
            embed = HashNGramEmbedding(
                hash_vocab_size=1000,
                embed_dim=32,
                ngram_sizes=ngram_sizes
            )

            inputs = ops.cast(
                keras.random.uniform((2, 20), minval=4, maxval=260),
                dtype='int32'
            )

            output = embed(inputs)
            assert output.shape == (2, 20, 32)

    def test_compute_ngram_embeddings_logic(self):
        """Test internal n-gram computation logic."""
        embed = HashNGramEmbedding(
            hash_vocab_size=100,
            embed_dim=16,
            ngram_sizes=[3]
        )

        # Build the layer
        embed.build((None, 10))

        # Test with simple sequence
        inputs = ops.array([[10, 20, 30, 40, 50]], dtype='int32')

        # Should compute hashes for 3-grams at each position
        output = embed(inputs)
        assert output.shape == (1, 5, 16)

        # All positions should have embeddings (even early ones with incomplete n-grams)
        assert ops.all(ops.isfinite(output))

    def test_serialization(self):
        """Test HashNGramEmbedding serialization."""
        embed = HashNGramEmbedding(
            hash_vocab_size=2000,
            embed_dim=96,
            ngram_sizes=[2, 3, 4, 5]
        )

        config = embed.get_config()
        assert config['hash_vocab_size'] == 2000
        assert config['embed_dim'] == 96
        assert config['ngram_sizes'] == [2, 3, 4, 5]

    def test_hash_collision_handling(self):
        """Test that hash collisions are handled reasonably."""
        # Use small hash vocab to force collisions
        embed = HashNGramEmbedding(
            hash_vocab_size=10,  # Very small to force collisions
            embed_dim=8,
            ngram_sizes=[3]
        )

        # Different inputs should still produce embeddings (even with collisions)
        inputs1 = ops.array([[100, 101, 102, 103, 104]], dtype='int32')
        inputs2 = ops.array([[200, 201, 202, 203, 204]], dtype='int32')

        output1 = embed(inputs1)
        output2 = embed(inputs2)

        assert output1.shape == (1, 5, 8)
        assert output2.shape == (1, 5, 8)
        # Should still produce valid embeddings despite small hash space
        assert ops.all(ops.isfinite(output1))
        assert ops.all(ops.isfinite(output2))


class TestBertBltEmbeddingsComponent:
    """Test BertBltEmbeddings component individually with individual parameters."""

    @pytest.fixture
    def basic_params(self) -> Dict[str, Any]:
        """Create basic parameters for BertBltEmbeddings testing."""
        return {
            'vocab_size': 260,
            'hidden_size': 256,
            'max_position_embeddings': 512,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,
            'use_hash_embeddings': True,
            'hash_vocab_size': 10000,
            'ngram_sizes': [3, 4, 5],
            'hash_embedding_dim': 256,
            'normalization_type': 'layer_norm'
        }

    def test_bert_blt_embeddings_initialization(self, basic_params):
        """Test BertBltEmbeddings initialization with individual parameters."""
        embeddings = BertBltEmbeddings(**basic_params)

        assert embeddings.vocab_size == 260
        assert embeddings.hidden_size == 256
        assert embeddings.max_position_embeddings == 512
        assert embeddings.initializer_range == 0.02
        assert embeddings.layer_norm_eps == 1e-12
        assert embeddings.hidden_dropout_prob == 0.1
        assert embeddings.use_hash_embeddings is True
        assert embeddings.hash_vocab_size == 10000
        assert embeddings.ngram_sizes == [3, 4, 5]
        assert embeddings.hash_embedding_dim == 256
        assert embeddings.normalization_type == 'layer_norm'

        # Components should be created in __init__ (modern Keras 3 pattern)
        assert embeddings.tokenizer is not None
        assert embeddings.byte_embeddings is not None
        assert embeddings.position_embeddings is not None
        assert embeddings.hash_embeddings is not None  # Should be created when enabled
        assert embeddings.layer_norm is not None
        assert embeddings.dropout is not None

        # Check that they are unbuilt initially
        assert not embeddings.built
        assert not embeddings.byte_embeddings.built

    def test_bert_blt_embeddings_without_hash(self):
        """Test BertBltEmbeddings without hash embeddings."""
        params = {
            'vocab_size': 260,
            'hidden_size': 128,
            'max_position_embeddings': 256,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,
            'use_hash_embeddings': False,
            'normalization_type': 'layer_norm'
        }

        embeddings = BertBltEmbeddings(**params)

        assert embeddings.use_hash_embeddings is False
        assert embeddings.hash_embeddings is None
        assert embeddings.hash_projection is None

    def test_bert_blt_embeddings_parameter_validation(self):
        """Test BertBltEmbeddings parameter validation."""
        base_params = {
            'vocab_size': 260,
            'hidden_size': 256,
            'max_position_embeddings': 512,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,
            'normalization_type': 'layer_norm'
        }

        # Test invalid vocab_size
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            params = base_params.copy()
            params['vocab_size'] = -1
            BertBltEmbeddings(**params)

        # Test invalid hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            params = base_params.copy()
            params['hidden_size'] = 0
            BertBltEmbeddings(**params)

        # Test invalid max_position_embeddings
        with pytest.raises(ValueError, match="max_position_embeddings must be positive"):
            params = base_params.copy()
            params['max_position_embeddings'] = -10
            BertBltEmbeddings(**params)

    def test_bert_blt_embeddings_build(self, basic_params):
        """Test BertBltEmbeddings build process."""
        embeddings = BertBltEmbeddings(**basic_params)
        input_shape = (None, 128)  # (batch_size, sequence_length)

        embeddings.build(input_shape)

        assert embeddings.built is True
        assert embeddings.tokenizer.built
        assert embeddings.byte_embeddings.built
        assert embeddings.position_embeddings.built
        assert embeddings.hash_embeddings.built  # Should be built when enabled
        assert embeddings.layer_norm.built
        assert embeddings.dropout.built

        # Check embedding dimensions
        assert embeddings.byte_embeddings.input_dim == basic_params['vocab_size']
        assert embeddings.byte_embeddings.output_dim == basic_params['hidden_size']
        assert embeddings.position_embeddings.input_dim == basic_params['max_position_embeddings']

    def test_bert_blt_embeddings_forward_pass(self, basic_params):
        """Test BertBltEmbeddings forward pass."""
        embeddings = BertBltEmbeddings(**basic_params)

        batch_size = 4
        seq_length = 32

        # Create byte token inputs (4 to 259 for valid byte tokens + offset)
        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=basic_params['vocab_size']
            ),
            dtype='int32'
        )

        output = embeddings(input_ids, training=False)

        assert output.shape == (batch_size, seq_length, basic_params['hidden_size'])

    def test_bert_blt_embeddings_with_position_ids(self, basic_params):
        """Test BertBltEmbeddings with custom position IDs."""
        embeddings = BertBltEmbeddings(**basic_params)

        batch_size = 2
        seq_length = 20

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=basic_params['vocab_size']
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

        assert output.shape == (batch_size, seq_length, basic_params['hidden_size'])

    def test_bert_blt_embeddings_training_mode(self, basic_params):
        """Test BertBltEmbeddings in training vs evaluation mode."""
        embeddings = BertBltEmbeddings(**basic_params)

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=basic_params['vocab_size']
            ),
            dtype='int32'
        )

        # Test both training modes
        output_train = embeddings(input_ids, training=True)
        output_eval = embeddings(input_ids, training=False)

        assert output_train.shape == output_eval.shape
        assert output_train.shape == (batch_size, seq_length, basic_params['hidden_size'])

    def test_bert_blt_embeddings_text_encoding(self, basic_params):
        """Test BertBltEmbeddings text encoding functionality."""
        embeddings = BertBltEmbeddings(**basic_params)

        # Test text encoding
        text = "Hello, World!"
        encoded = embeddings.encode_text(text, max_length=32)

        assert encoded.shape == (1, 32)  # Single batch, padded to max_length
        assert encoded.dtype.name == 'int32'

    def test_bert_blt_embeddings_text_decoding(self, basic_params):
        """Test BertBltEmbeddings text decoding functionality."""
        embeddings = BertBltEmbeddings(**basic_params)

        # Test round-trip encoding and decoding
        original_text = "Hello, byte world!"
        encoded = embeddings.encode_text(original_text, max_length=64, add_special_tokens=True)
        decoded = embeddings.decode_tokens(encoded)

        # Should recover original text (special tokens filtered out)
        assert original_text in decoded or decoded in original_text  # Account for padding/special tokens

    def test_bert_blt_embeddings_normalization_types(self):
        """Test BertBltEmbeddings with different normalization types."""
        normalization_types = ["layer_norm", "rms_norm", "batch_norm"]

        for norm_type in normalization_types:
            params = {
                'vocab_size': 260,
                'hidden_size': 128,
                'max_position_embeddings': 256,
                'initializer_range': 0.02,
                'layer_norm_eps': 1e-12,
                'hidden_dropout_prob': 0.1,
                'use_hash_embeddings': False,  # Simpler for this test
                'normalization_type': norm_type
            }

            embeddings = BertBltEmbeddings(**params)

            input_ids = ops.cast(
                keras.random.uniform((2, 16), minval=4, maxval=260),
                dtype='int32'
            )

            output = embeddings(input_ids, training=False)
            assert output.shape == (2, 16, 128)

    def test_bert_blt_embeddings_hash_projection(self):
        """Test BertBltEmbeddings with hash dimension different from hidden size."""
        params = {
            'vocab_size': 260,
            'hidden_size': 256,
            'max_position_embeddings': 512,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,
            'use_hash_embeddings': True,
            'hash_vocab_size': 10000,
            'ngram_sizes': [3, 4],
            'hash_embedding_dim': 128,  # Different from hidden_size
            'normalization_type': 'layer_norm'
        }

        embeddings = BertBltEmbeddings(**params)

        # Should create projection layer
        assert embeddings.hash_projection is not None

        input_ids = ops.cast(
            keras.random.uniform((2, 20), minval=4, maxval=260),
            dtype='int32'
        )

        output = embeddings(input_ids, training=False)
        assert output.shape == (2, 20, 256)  # Should match hidden_size

    def test_bert_blt_embeddings_serialization(self, basic_params):
        """Test BertBltEmbeddings serialization and deserialization."""
        embeddings = BertBltEmbeddings(**basic_params)

        # Test get_config
        config = embeddings.get_config()
        assert isinstance(config, dict)
        assert config['vocab_size'] == basic_params['vocab_size']
        assert config['hidden_size'] == basic_params['hidden_size']
        assert config['max_position_embeddings'] == basic_params['max_position_embeddings']
        assert config['use_hash_embeddings'] == basic_params['use_hash_embeddings']
        assert config['hash_vocab_size'] == basic_params['hash_vocab_size']
        assert config['ngram_sizes'] == basic_params['ngram_sizes']
        assert config['normalization_type'] == basic_params['normalization_type']

    def test_bert_blt_embeddings_compute_output_shape(self, basic_params):
        """Test BertBltEmbeddings compute_output_shape method."""
        embeddings = BertBltEmbeddings(**basic_params)

        input_shape = (4, 64)  # (batch_size, seq_length)
        output_shape = embeddings.compute_output_shape(input_shape)

        expected_shape = (4, 64, basic_params['hidden_size'])
        assert output_shape == expected_shape


class TestBertBltModelInitialization:
    """Test BertBlt model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BertBlt model initialization."""
        config = BertBltConfig(hidden_size=256, num_layers=4, num_heads=8)  # 256/8=32, valid
        model = BertBlt(config)

        assert model.config == config
        assert model.config.hidden_size == 256
        assert model.config.num_layers == 4
        assert model.add_pooling_layer is True
        assert model.built is False

        # Components should be created in __init__ (modern Keras 3 pattern)
        assert model.embeddings is not None
        assert len(model.encoder_layers) == 4
        assert model.pooler is not None

        # But should not be built yet
        assert not model.embeddings.built
        for layer in model.encoder_layers:
            assert not layer.built

        # Check that embeddings has correct parameters
        assert model.embeddings.vocab_size == config.vocab_size
        assert model.embeddings.hidden_size == config.hidden_size
        assert model.embeddings.max_position_embeddings == config.max_position_embeddings
        assert model.embeddings.use_hash_embeddings == config.use_hash_embeddings
        assert model.embeddings.normalization_type == config.normalization_type

    def test_initialization_without_pooling(self):
        """Test BertBlt model initialization without pooling layer."""
        config = BertBltConfig(hidden_size=256, num_heads=8)  # 256/8=32, valid
        model = BertBlt(config, add_pooling_layer=False)

        assert model.add_pooling_layer is False
        assert model.pooler is None

    def test_initialization_with_custom_config(self):
        """Test BertBlt model initialization with custom configuration."""
        config = BertBltConfig(
            vocab_size=300,
            hidden_size=512,
            num_layers=6,
            num_heads=8,  # 512/8=64, valid
            intermediate_size=2048,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.1,
            normalization_type="rms_norm",
            normalization_position="pre",
            use_hash_embeddings=True,
            hash_vocab_size=100000,
            ngram_sizes=[2, 3, 4]
        )
        model = BertBlt(config)

        assert model.config.vocab_size == 300
        assert model.config.hidden_size == 512
        assert model.config.num_layers == 6
        assert model.config.num_heads == 8
        assert model.config.intermediate_size == 2048
        assert model.config.hidden_dropout_prob == 0.2
        assert model.config.normalization_type == "rms_norm"
        assert model.config.normalization_position == "pre"
        assert model.config.use_hash_embeddings is True
        assert model.config.hash_vocab_size == 100000
        assert model.config.ngram_sizes == [2, 3, 4]

        # Check that embeddings received correct individual parameters
        assert model.embeddings.vocab_size == 300
        assert model.embeddings.hidden_size == 512
        assert model.embeddings.hidden_dropout_prob == 0.2
        assert model.embeddings.normalization_type == "rms_norm"
        assert model.embeddings.use_hash_embeddings is True
        assert model.embeddings.hash_vocab_size == 100000
        assert model.embeddings.ngram_sizes == [2, 3, 4]


class TestBertBltModelBuilding:
    """Test BertBlt model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> BertBltConfig:
        """Create a basic BertBlt config for testing."""
        return BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=3,
            num_heads=8,  # 256/8=32, valid
            intermediate_size=1024,
            max_position_embeddings=512,
            use_hash_embeddings=True,
            hash_vocab_size=10000
        )

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality."""
        model = BertBlt(basic_config)

        # Test forward pass (builds automatically)
        batch_size = 2
        seq_length = 32
        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=basic_config.vocab_size
            ),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)

        assert model.built is True
        assert model.embeddings.built
        assert len(model.encoder_layers) == basic_config.num_layers

        # Check that transformer layers are built
        for layer in model.encoder_layers:
            assert layer.built is True

        # Check outputs
        assert isinstance(outputs, tuple)
        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (batch_size, seq_length, basic_config.hidden_size)
        assert pooled_output.shape == (batch_size, basic_config.hidden_size)

    def test_build_without_pooling_layer(self, basic_config):
        """Test building BertBlt model without pooling layer."""
        model = BertBlt(basic_config, add_pooling_layer=False)

        input_ids = ops.cast(
            keras.random.uniform((2, 32), minval=4, maxval=basic_config.vocab_size),
            dtype='int32'
        )

        output = model(input_ids, training=False)

        assert model.built is True
        assert model.pooler is None
        assert output.shape == (2, 32, basic_config.hidden_size)

    def test_transformer_layers_configuration(self, basic_config):
        """Test that TransformerLayers are configured correctly."""
        model = BertBlt(basic_config)

        # Trigger building by calling the model
        input_ids = ops.cast(
            keras.random.uniform((1, 16), minval=4, maxval=basic_config.vocab_size),
            dtype='int32'
        )
        _ = model(input_ids, training=False)

        # Check transformer layer configuration
        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config.hidden_size
            assert layer.num_heads == basic_config.num_heads
            assert layer.intermediate_size == basic_config.intermediate_size
            assert layer.normalization_type == basic_config.normalization_type
            assert layer.normalization_position == basic_config.normalization_position
            assert layer.name == f"encoder_layer_{i}"


class TestBertBltModelForwardPass:
    """Test BertBlt model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> BertBlt:
        """Create a built BertBlt model for testing."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=3,
            num_heads=8,  # 256/8=32, valid
            intermediate_size=1024,
            max_position_embeddings=512,
            use_hash_embeddings=True
        )
        model = BertBlt(config)

        # Build by calling with sample input
        sample_input = ops.cast(
            keras.random.uniform((1, 16), minval=4, maxval=config.vocab_size),
            dtype='int32'
        )
        _ = model(sample_input, training=False)

        return model

    def test_forward_pass_input_ids_only(self, built_model):
        """Test forward pass with only input IDs."""
        batch_size = 4
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
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
                minval=4,
                maxval=built_model.config.vocab_size
            ),
            dtype='int32'
        )

        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')

        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
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
                minval=4,
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
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=2,
            num_heads=8,  # 256/8=32, valid
            use_hash_embeddings=False  # Simpler for this test
        )
        model = BertBlt(config, add_pooling_layer=False)

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
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
                minval=4,
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

    def test_forward_pass_different_sequence_lengths(self, built_model):
        """Test forward pass with different sequence lengths."""
        batch_size = 2
        sequence_lengths = [8, 16, 32, 64]

        for seq_length in sequence_lengths:
            input_ids = ops.cast(
                keras.random.uniform(
                    (batch_size, seq_length),
                    minval=4,
                    maxval=built_model.config.vocab_size
                ),
                dtype='int32'
            )

            outputs = built_model(input_ids, training=False)
            sequence_output, pooled_output = outputs

            assert sequence_output.shape == (batch_size, seq_length, built_model.config.hidden_size)
            assert pooled_output.shape == (batch_size, built_model.config.hidden_size)


class TestBertBltTextProcessing:
    """Test BertBlt text processing capabilities."""

    @pytest.fixture
    def model(self) -> BertBlt:
        """Create a BertBlt model for text processing tests."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=True,
            hash_vocab_size=5000
        )
        return BertBlt(config)

    def test_encode_text(self, model):
        """Test text encoding functionality."""
        text = "Hello, byte world!"
        encoded = model.encode_text(text, max_length=32)

        assert encoded.shape == (1, 32)
        assert encoded.dtype.name == 'int32'

        # Should have some non-zero values (not all padding)
        assert ops.sum(ops.cast(encoded != 0, 'int32')) > 0

    def test_decode_tokens(self, model):
        """Test token decoding functionality."""
        # Create simple byte tokens
        tokens = ops.array([[1, 72, 101, 108, 108, 111, 2, 0, 0, 0]], dtype='int32')  # BOS + "Hello" + EOS + padding
        decoded = model.decode_tokens(tokens)

        assert isinstance(decoded, str)
        # Should contain "Hello" somewhere
        assert "ello" in decoded  # Account for BOS/EOS filtering

    def test_encode_and_predict(self, model):
        """Test encode_and_predict convenience method."""
        text = "Test sentence."
        outputs = model.encode_and_predict(text, max_length=16, return_dict=True)

        assert isinstance(outputs, dict)
        assert 'last_hidden_state' in outputs

        # Check shapes
        last_hidden_state = outputs['last_hidden_state']
        assert last_hidden_state.shape[0] == 1  # Batch size 1
        assert last_hidden_state.shape[1] == 16  # Max length
        assert last_hidden_state.shape[2] == model.config.hidden_size

    def test_multilingual_text_processing(self, model):
        """Test text processing with multilingual input."""
        texts = [
            "Hello World",
            "Bonjour le monde",
            "Hola mundo",
            "こんにちは世界",
            "🚀🌟✨",
            ""
        ]

        for text in texts:
            # Should not crash on any text
            encoded = model.encode_text(text, max_length=32)
            assert encoded.shape == (1, 32)

            # Test round-trip when possible
            if text:  # Skip empty string
                decoded = model.decode_tokens(encoded)
                assert isinstance(decoded, str)

    def test_long_text_processing(self, model):
        """Test processing of long text that needs truncation."""
        # Create text longer than max_position_embeddings
        long_text = "This is a very long sentence. " * 100  # Much longer than typical limits

        encoded = model.encode_text(long_text, max_length=64)
        assert encoded.shape == (1, 64)  # Should be truncated

        # Should still produce valid outputs
        outputs = model.encode_and_predict(long_text, max_length=64, return_dict=False)
        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        else:
            sequence_output = outputs
        assert sequence_output.shape == (1, 64, model.config.hidden_size)

    def test_special_characters_processing(self, model):
        """Test processing of text with special characters."""
        special_texts = [
            "Hello\nWorld\t!",
            "Text with 'quotes' and \"double quotes\"",
            "Numbers: 123456789",
            "Symbols: !@#$%^&*()",
            "Unicode: café résumé naïve",
        ]

        for text in special_texts:
            encoded = model.encode_text(text)
            assert encoded.shape[0] == 1  # Single batch

            # Should produce valid embeddings
            outputs = model.encode_and_predict(text, return_dict=False)
            if isinstance(outputs, tuple):
                sequence_output = outputs[0]
            else:
                sequence_output = outputs
            assert ops.all(ops.isfinite(sequence_output))


class TestBertBltSerialization:
    """Test BertBlt model serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        config = BertBltConfig(
            vocab_size=300,
            hidden_size=512,
            num_layers=6,
            num_heads=8,  # 512/8=64, valid
            intermediate_size=2048,
            hidden_dropout_prob=0.15,
            normalization_type="rms_norm",
            use_hash_embeddings=True,
            hash_vocab_size=50000,
            ngram_sizes=[3, 4, 5, 6]
        )
        model = BertBlt(config)

        model_config = model.get_config()

        assert isinstance(model_config, dict)
        assert 'config' in model_config
        assert 'add_pooling_layer' in model_config

        config_dict = model_config['config']
        assert config_dict['vocab_size'] == 300
        assert config_dict['hidden_size'] == 512
        assert config_dict['num_layers'] == 6
        assert config_dict['hidden_dropout_prob'] == 0.15
        assert config_dict['normalization_type'] == "rms_norm"
        assert config_dict['use_hash_embeddings'] is True
        assert config_dict['hash_vocab_size'] == 50000
        assert config_dict['ngram_sizes'] == [3, 4, 5, 6]

    def test_from_config_reconstruction(self):
        """Test model reconstruction from configuration."""
        original_config = BertBltConfig(
            vocab_size=280,
            hidden_size=384,
            num_layers=4,
            num_heads=12,  # 384/12=32, valid
            use_hash_embeddings=True,
            hash_vocab_size=25000
        )
        original_model = BertBlt(original_config, add_pooling_layer=False)

        # Get config and reconstruct
        model_config = original_model.get_config()
        reconstructed_model = BertBlt.from_config(model_config)

        # Check that configs match
        assert reconstructed_model.config.vocab_size == original_config.vocab_size
        assert reconstructed_model.config.hidden_size == original_config.hidden_size
        assert reconstructed_model.config.num_layers == original_config.num_layers
        assert reconstructed_model.config.num_heads == original_config.num_heads
        assert reconstructed_model.config.use_hash_embeddings == original_config.use_hash_embeddings
        assert reconstructed_model.config.hash_vocab_size == original_config.hash_vocab_size
        assert reconstructed_model.add_pooling_layer == original_model.add_pooling_layer

        # Check that embeddings parameters match
        assert reconstructed_model.embeddings.vocab_size == original_model.embeddings.vocab_size
        assert reconstructed_model.embeddings.hidden_size == original_model.embeddings.hidden_size
        assert reconstructed_model.embeddings.use_hash_embeddings == original_model.embeddings.use_hash_embeddings

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        # Create and use model (this builds it)
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=2,
            num_heads=8,  # 256/8=32, valid
            intermediate_size=512,
            use_hash_embeddings=False  # Simpler for serialization test
        )
        model = BertBlt(config)

        # Test forward pass (builds automatically)
        batch_size = 2
        seq_length = 16
        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        original_outputs = model(input_ids, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_bert_blt.keras')
            model.save(model_path)

            loaded_model = keras.models.load_model(model_path)

            # Test that loaded model produces same output
            loaded_outputs = loaded_model(input_ids, training=False)

            # Handle both tuple and single tensor outputs
            if isinstance(original_outputs, tuple):
                assert isinstance(loaded_outputs, tuple)
                assert len(original_outputs) == len(loaded_outputs)

                # Outputs should be very close
                for orig, loaded in zip(original_outputs, loaded_outputs):
                    np.testing.assert_allclose(
                        keras.ops.convert_to_numpy(orig),
                        keras.ops.convert_to_numpy(loaded),
                        rtol=1e-5,
                        atol=1e-6,
                        err_msg="Outputs should match after serialization"
                    )
            else:
                # Single tensor output
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs),
                    keras.ops.convert_to_numpy(loaded_outputs),
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg="Outputs should match after serialization"
                )


class TestBertBltEdgeCases:
    """Test BertBlt model edge cases and error handling."""

    def test_minimum_sequence_length(self):
        """Test BertBlt with minimum sequence length."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,  # 128/8=16, valid
            use_hash_embeddings=False
        )
        model = BertBlt(config)

        # Single token
        input_ids = ops.cast([[100]], dtype='int32')  # Valid byte token
        outputs = model(input_ids, training=False)

        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (1, 1, 128)
        assert pooled_output.shape == (1, 128)

    def test_maximum_position_embeddings(self):
        """Test BertBlt with maximum sequence length."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,  # 128/8=16, valid
            max_position_embeddings=64,
            use_hash_embeddings=False
        )
        model = BertBlt(config)

        seq_length = 64
        input_ids = ops.cast(
            keras.random.uniform(
                (2, seq_length),
                minval=4,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)
        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (2, seq_length, 128)

    def test_byte_token_range(self):
        """Test BertBlt with different byte token ranges."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=False
        )
        model = BertBlt(config)

        # Test different byte ranges
        byte_ranges = [
            (4, 50),  # Low byte values
            (100, 150),  # Mid byte values
            (200, 259)  # High byte values
        ]

        for min_val, max_val in byte_ranges:
            input_ids = ops.cast(
                keras.random.uniform(
                    (2, 16),
                    minval=min_val,
                    maxval=max_val + 1
                ),
                dtype='int32'
            )

            outputs = model(input_ids, training=False)
            sequence_output, pooled_output = outputs

            assert sequence_output.shape == (2, 16, 128)
            assert pooled_output.shape == (2, 128)
            assert ops.all(ops.isfinite(sequence_output))
            assert ops.all(ops.isfinite(pooled_output))

    def test_hash_embeddings_edge_cases(self):
        """Test BertBlt with hash embeddings edge cases."""
        # Very small hash vocab (forces collisions)
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=64,
            num_layers=1,
            num_heads=8,
            use_hash_embeddings=True,
            hash_vocab_size=100,  # Very small
            ngram_sizes=[2]  # Simple n-gram
        )
        model = BertBlt(config)

        input_ids = ops.cast(
            keras.random.uniform((2, 20), minval=4, maxval=260),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)
        sequence_output, pooled_output = outputs

        assert sequence_output.shape == (2, 20, 64)
        assert ops.all(ops.isfinite(sequence_output))

    def test_all_padding_input(self):
        """Test BertBlt with all padding tokens."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=False
        )
        model = BertBlt(config)

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

    def test_text_encoding_edge_cases(self):
        """Test text encoding edge cases."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=64,
            num_layers=1,
            num_heads=8
        )
        model = BertBlt(config)

        # Empty text
        encoded_empty = model.encode_text("", max_length=16)
        assert encoded_empty.shape == (1, 16)

        # Very long text (should truncate)
        long_text = "A" * 1000
        encoded_long = model.encode_text(long_text, max_length=32)
        assert encoded_long.shape == (1, 32)

        # Text with only special characters
        special_text = "\n\t\r"
        encoded_special = model.encode_text(special_text, max_length=16)
        assert encoded_special.shape == (1, 16)


class TestBertBltFactoryFunctions:
    """Test BertBlt factory functions."""

    def test_create_bert_blt_base(self):
        """Test create_bert_blt_base factory function."""
        config = create_bert_blt_base()

        assert config.vocab_size == 260
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.intermediate_size == 3072
        assert config.use_hash_embeddings is True
        assert config.hash_vocab_size == 500000
        assert config.ngram_sizes == [3, 4, 5, 6, 7, 8]

    def test_create_bert_blt_large(self):
        """Test create_bert_blt_large factory function."""
        config = create_bert_blt_large()

        assert config.vocab_size == 260
        assert config.hidden_size == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.intermediate_size == 4096
        assert config.max_position_embeddings == 4096
        assert config.use_hash_embeddings is True
        assert config.hash_vocab_size == 1000000

    def test_create_bert_blt_for_classification(self):
        """Test create_bert_blt_for_classification factory function."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=3,
            num_heads=8,  # 256/8=32, valid
            use_hash_embeddings=False  # Simpler for test
        )
        num_labels = 5

        model = create_bert_blt_for_classification(config, num_labels)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 2  # input_ids, attention_mask
        assert model.inputs[0].shape[1:] == (None,)
        assert model.inputs[1].shape[1:] == (None,)

        # Test forward pass
        batch_size = 2
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')

        logits = model([input_ids, attention_mask], training=False)
        assert logits.shape == (batch_size, num_labels)

    def test_create_bert_blt_for_classification_with_dropout(self):
        """Test create_bert_blt_for_classification with custom dropout."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,  # 128/8=16, valid
            use_hash_embeddings=False
        )
        num_labels = 3
        classifier_dropout = 0.2

        model = create_bert_blt_for_classification(
            config,
            num_labels,
            classifier_dropout=classifier_dropout
        )

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')

        logits = model([input_ids, attention_mask], training=False)
        assert logits.shape == (batch_size, num_labels)

    def test_create_bert_blt_for_sequence_output(self):
        """Test create_bert_blt_for_sequence_output factory function."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=3,
            num_heads=8,  # 256/8=32, valid
            use_hash_embeddings=False
        )

        model = create_bert_blt_for_sequence_output(config)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 2

        # Test forward pass
        batch_size = 2
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')

        sequence_output = model([input_ids, attention_mask], training=False)
        assert sequence_output.shape == (batch_size, seq_length, config.hidden_size)

    def test_create_robust_bert_blt(self):
        """Test create_robust_bert_blt factory function."""
        config = create_robust_bert_blt()

        # Should be based on base config
        assert config.hidden_size == 768
        assert config.num_layers == 12

        # Should have robustness optimizations
        assert config.use_hash_embeddings is True
        assert config.hash_vocab_size == 1000000  # Larger hash space
        assert 2 in config.ngram_sizes  # Include bigrams
        assert config.hidden_dropout_prob == 0.15  # Higher regularization
        assert config.attention_probs_dropout_prob == 0.15

    def test_factory_functions_functional(self):
        """Test that all factory functions create functional models."""
        configs_and_creators = [
            (create_bert_blt_base(), "base"),
            (create_bert_blt_large(), "large"),
            (create_robust_bert_blt(), "robust")
        ]

        for config, name in configs_and_creators:
            # Modify config for faster testing
            config.num_layers = 2
            config.hidden_size = 128
            config.num_heads = 8  # 128/8=16, valid
            config.intermediate_size = 512
            config.use_hash_embeddings = False  # Simpler for test

            # Test classification model
            cls_model = create_bert_blt_for_classification(config, num_labels=3)

            batch_size = 2
            seq_length = 16

            input_ids = ops.cast(
                keras.random.uniform(
                    (batch_size, seq_length),
                    minval=4,
                    maxval=config.vocab_size
                ),
                dtype='int32'
            )
            attention_mask = ops.ones((batch_size, seq_length), dtype='int32')

            logits = cls_model([input_ids, attention_mask], training=False)
            assert logits.shape == (batch_size, 3)

            # Test sequence output model
            seq_model = create_bert_blt_for_sequence_output(config)
            sequence_output = seq_model([input_ids, attention_mask], training=False)
            assert sequence_output.shape == (batch_size, seq_length, config.hidden_size)


class TestBertBltIntegration:
    """Integration tests for BertBlt components working together."""

    def test_end_to_end_text_classification(self):
        """Test complete text classification workflow."""
        # Create small model for testing
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=3,
            num_heads=8,  # 256/8=32, valid
            intermediate_size=1024,
            use_hash_embeddings=True,
            hash_vocab_size=5000
        )

        model = create_bert_blt_for_classification(config, num_labels=4)

        # Create test data with realistic byte tokens
        batch_size = 4
        seq_length = 32

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
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

        # Forward pass
        logits = model([input_ids, attention_mask], training=False)

        assert logits.shape == (batch_size, 4)

        # Test that logits are reasonable
        assert not ops.all(logits == 0)
        assert ops.all(ops.isfinite(logits))
        assert ops.max(ops.abs(logits)) < 100

    def test_hash_embedding_integration(self):
        """Test integration of hash embeddings with the full model."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=True,
            hash_vocab_size=1000,
            ngram_sizes=[3, 4]
        )

        model = BertBlt(config)

        # Test with and without hash embeddings
        input_ids = ops.cast(
            keras.random.uniform((2, 24), minval=4, maxval=260),
            dtype='int32'
        )

        outputs_with_hash = model(input_ids, training=False)

        # Disable hash embeddings
        config.use_hash_embeddings = False
        model_no_hash = BertBlt(config)
        outputs_no_hash = model_no_hash(input_ids, training=False)

        # Both should work but produce different results
        assert outputs_with_hash[0].shape == outputs_no_hash[0].shape
        # Should be different due to hash embeddings
        assert not allclose(outputs_with_hash[0], outputs_no_hash[0], atol=1e-3)

    def test_byte_level_vs_token_level_comparison(self):
        """Test byte-level processing advantages."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=False  # Focus on byte-level processing
        )

        model = BertBlt(config)

        # Test with texts that would be challenging for traditional tokenizers
        challenging_texts = [
            "Café",  # Accented characters
            "Hello world",  # Simple baseline
            "🚀🌟",  # Emojis
            "test123",  # Mixed alphanumeric
        ]

        for text in challenging_texts:
            # Should handle all texts uniformly at byte level
            encoded = model.encode_text(text, max_length=32)
            outputs = model(encoded, training=False)

            assert outputs[0].shape == (1, 32, 128)
            assert ops.all(ops.isfinite(outputs[0]))

    def test_component_consistency(self):
        """Test consistency between BertBlt model and manual component usage."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=False  # Simpler for consistency test
        )

        # Create BertBlt model
        bert_model = BertBlt(config, add_pooling_layer=False)

        batch_size = 2
        seq_length = 24

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )

        # Get BertBlt model output
        bert_output = bert_model(input_ids, training=False)

        # Manual processing through components
        embeddings_output = bert_model.embeddings(input_ids, training=False)

        # Process through transformer layers manually
        hidden_states = embeddings_output
        for layer in bert_model.encoder_layers:
            hidden_states = layer(hidden_states, training=False)

        # Should be very close
        assert allclose(bert_output, hidden_states, rtol=1e-5)

    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire BertBlt model."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=64,
            num_layers=2,
            num_heads=8,  # 64/8=8, valid
            intermediate_size=256,
            use_hash_embeddings=True,
            hash_vocab_size=1000
        )

        model = create_bert_blt_for_classification(config, num_labels=3)

        batch_size = 2
        seq_length = 16

        input_ids = ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=4,
                maxval=config.vocab_size
            ),
            dtype='int32'
        )
        attention_mask = ops.ones((batch_size, seq_length), dtype='int32')

        with tf.GradientTape() as tape:
            logits = model([input_ids, attention_mask], training=True)

            # Create a simple classification loss
            targets = ops.one_hot(ops.array([0, 2]), 3)
            loss = keras.losses.categorical_crossentropy(targets, logits)
            loss = ops.mean(loss)

        gradients = tape.gradient(loss, model.trainable_weights)

        # Check that we have gradients for most weights
        non_none_grads = [g for g in gradients if g is not None]
        # At least half of the weights should have gradients
        assert len(non_none_grads) > len(model.trainable_weights) * 0.5

        # Check that gradients have reasonable magnitudes
        grad_norms = [
            ops.sqrt(ops.sum(ops.square(g)) + 1e-8) for g in non_none_grads
        ]
        assert all(norm > 0.0 for norm in grad_norms)
        assert all(norm < 1000.0 for norm in grad_norms)

    def test_different_normalization_consistency(self):
        """Test consistency across different normalization types."""
        normalization_types = ["layer_norm", "rms_norm"]

        outputs = {}

        for norm_type in normalization_types:
            config = BertBltConfig(
                vocab_size=260,
                hidden_size=128,
                num_layers=2,
                num_heads=8,  # 128/8=16, valid
                normalization_type=norm_type,
                use_hash_embeddings=False,
                # Use same initialization for consistency
                initializer_range=0.01
            )

            model = BertBlt(config, add_pooling_layer=False)

            input_ids = ops.cast(
                keras.random.uniform(
                    (2, 16),
                    minval=4,
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


class TestBertBltAdvancedFeatures:
    """Test advanced BertBlt features."""

    def test_hash_embedding_sizes(self):
        """Test BertBlt with different hash embedding configurations."""
        hash_configs = [
            {"hash_vocab_size": 1000, "ngram_sizes": [3]},
            {"hash_vocab_size": 10000, "ngram_sizes": [3, 4, 5]},
            {"hash_vocab_size": 50000, "ngram_sizes": [2, 3, 4, 5, 6]}
        ]

        for hash_config in hash_configs:
            config = BertBltConfig(
                vocab_size=260,
                hidden_size=128,
                num_layers=2,
                num_heads=8,
                use_hash_embeddings=True,
                **hash_config
            )

            model = BertBlt(config, add_pooling_layer=False)

            input_ids = ops.cast(
                keras.random.uniform((2, 20), minval=4, maxval=260),
                dtype='int32'
            )

            output = model(input_ids, training=False)
            assert output.shape == (2, 20, 128)
            assert ops.all(ops.isfinite(output))

    def test_different_ffn_types_with_byte_processing(self):
        """Test BertBlt with different FFN types."""
        ffn_types = ["mlp", "swiglu"]

        for ffn_type in ffn_types:
            config = BertBltConfig(
                vocab_size=260,
                hidden_size=128,
                num_layers=2,
                num_heads=8,
                ffn_type=ffn_type,
                use_hash_embeddings=False
            )

            model = BertBlt(config, add_pooling_layer=False)

            input_ids = ops.cast(
                keras.random.uniform((2, 16), minval=4, maxval=260),
                dtype='int32'
            )

            output = model(input_ids, training=False)
            assert output.shape == (2, 16, 128)

    def test_stochastic_depth_with_byte_processing(self):
        """Test BertBlt with stochastic depth enabled."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=3,
            num_heads=8,
            use_stochastic_depth=True,
            stochastic_depth_rate=0.2,
            use_hash_embeddings=False
        )

        model = BertBlt(config)

        input_ids = ops.cast(
            keras.random.uniform((2, 24), minval=4, maxval=260),
            dtype='int32'
        )

        # Test both training and inference modes
        output_train = model(input_ids, training=True)
        output_eval = model(input_ids, training=False)

        # Both should have correct shapes
        assert output_train[0].shape == (2, 24, 128)
        assert output_eval[0].shape == (2, 24, 128)

    def test_hash_embedding_dimension_mismatch(self):
        """Test BertBlt with hash embedding dimension different from hidden size."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=True,
            hash_vocab_size=5000,
            hash_embedding_dim=128,  # Different from hidden_size
            ngram_sizes=[3, 4]
        )

        model = BertBlt(config)

        input_ids = ops.cast(
            keras.random.uniform((2, 20), minval=4, maxval=260),
            dtype='int32'
        )

        output = model(input_ids, training=False)
        sequence_output, pooled_output = output

        # Should work with projection layer
        assert sequence_output.shape == (2, 20, 256)
        assert pooled_output.shape == (2, 256)

    def test_multilingual_robustness(self):
        """Test BertBlt robustness with multilingual text."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=True,
            hash_vocab_size=10000
        )

        model = BertBlt(config)

        # Test various languages and scripts
        multilingual_texts = [
            "Hello world",  # English
            "Bonjour le monde",  # French
            "Hola mundo",  # Spanish
            "Guten Tag Welt",  # German
            "こんにちは世界",  # Japanese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "你好世界",  # Chinese
            "🌍🚀✨"  # Emojis
        ]

        for text in multilingual_texts:
            if text.strip():  # Skip empty texts
                encoded = model.encode_text(text, max_length=32)
                outputs = model(encoded, training=False)

                # Should produce valid embeddings for all languages
                assert outputs[0].shape == (1, 32, 128)
                assert ops.all(ops.isfinite(outputs[0]))

    def test_byte_level_corruption_robustness(self):
        """Test BertBlt robustness to text corruption."""
        config = BertBltConfig(
            vocab_size=260,
            hidden_size=128,
            num_layers=2,
            num_heads=8,
            use_hash_embeddings=True
        )

        model = BertBlt(config)

        # Simulate various types of text corruption
        corrupted_inputs = [
            # Random byte insertions (invalid UTF-8 sequences)
            ops.cast(keras.random.uniform((1, 20), minval=4, maxval=260), dtype='int32'),
            # Mix of valid and invalid bytes
            ops.concatenate([
                ops.cast([[72, 101, 108, 108, 111]], dtype='int32'),  # "Hello"
                ops.cast(keras.random.uniform((1, 15), minval=4, maxval=260), dtype='int32')
            ], axis=1)
        ]

        for corrupt_input in corrupted_inputs:
            # Should handle corrupted input gracefully
            outputs = model(corrupt_input, training=False)
            assert outputs[0].shape == (1, corrupt_input.shape[1], 128)
            assert ops.all(ops.isfinite(outputs[0]))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])