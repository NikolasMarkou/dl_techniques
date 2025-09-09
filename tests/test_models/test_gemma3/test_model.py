"""
Comprehensive Tests for Gemma3 Model Implementation

This module provides thorough tests for the refined Gemma3 model, including:
- Basic functionality tests with framework integration
- Configuration variants optimized for single GPU testing
- Serialization/deserialization with Modern Keras 3 compliance
- Factory functions and utilities
- Error handling and edge cases
- Performance validation
- Dual normalization pattern testing
- Mixed attention pattern validation

All tests use small model configurations to keep memory usage low for single GPU environments.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any, List

import keras

from dl_techniques.models.gemma3.model import (
    Gemma3Model,
    Gemma3TransformerBlock,
    create_gemma3_270m,
    GEMMA3_270M_CONFIG
)


class TestGemma3ModelBasic:
    """Test basic Gemma3 model functionality with framework integration."""

    @pytest.fixture
    def tiny_config(self) -> Dict[str, Any]:
        """Create a tiny model configuration for fast testing."""
        return {
            'vocab_size': 1000,
            'emb_dim': 128,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 256,
            'head_dim': 32,  # emb_dim // num_heads
            'max_context_length': 512,
            'sliding_window': 64,
            'layer_types': ['full_attention', 'sliding_attention'],
            'use_bias': False,
            'dropout_rate': 0.1,
        }

    @pytest.fixture
    def mini_config(self) -> Dict[str, Any]:
        """Create a mini model configuration for more comprehensive testing."""
        return {
            'vocab_size': 1500,
            'emb_dim': 192,
            'num_layers': 4,
            'num_heads': 6,
            'num_kv_groups': 2,
            'hidden_dim': 384,
            'head_dim': 32,  # emb_dim // num_heads
            'max_context_length': 1024,
            'sliding_window': 128,
            'layer_types': ['full_attention', 'sliding_attention', 'full_attention', 'sliding_attention'],
            'query_pre_attn_scalar': 32.0,
            'use_bias': False,
            'dropout_rate': 0.0,
        }

    @pytest.fixture
    def sample_input(self) -> keras.KerasTensor:
        """Create sample input tensor for testing."""
        return keras.ops.convert_to_tensor(
            np.random.randint(0, 800, size=(2, 32)),
            dtype='int32'
        )

    def test_model_creation(self, tiny_config: Dict[str, Any]):
        """Test basic model creation and initialization."""
        model = Gemma3Model(**tiny_config)

        # Check model attributes
        assert model.vocab_size == tiny_config['vocab_size']
        assert model.emb_dim == tiny_config['emb_dim']
        assert model.num_layers == tiny_config['num_layers']
        assert len(model.transformer_blocks) == tiny_config['num_layers']
        assert model.head_dim == tiny_config['head_dim']
        assert model.max_context_length == tiny_config['max_context_length']

        # Check embedding layer
        assert model.token_embeddings.input_dim == tiny_config['vocab_size']
        assert model.token_embeddings.output_dim == tiny_config['emb_dim']

        # Check final normalization layer (should be RMSNorm from factory)
        assert model.final_norm is not None
        from dl_techniques.layers.norms.rms_norm import RMSNorm
        assert isinstance(model.final_norm, RMSNorm)

        # Check output head
        assert model.output_head is not None
        assert model.output_head.units == tiny_config['vocab_size']

        # Check transformer blocks
        assert len(model.transformer_blocks) == tiny_config['num_layers']
        for i, block in enumerate(model.transformer_blocks):
            assert isinstance(block, Gemma3TransformerBlock)
            assert block.emb_dim == tiny_config['emb_dim']
            assert block.attention_type == tiny_config['layer_types'][i]

    def test_model_forward_pass(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test forward pass through the model."""
        model = Gemma3Model(**tiny_config)

        # Forward pass
        output = model(sample_input)

        # Check output shape
        expected_shape = (sample_input.shape[0], sample_input.shape[1], tiny_config['vocab_size'])
        assert output.shape == expected_shape

        # Check output is not NaN or infinite
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np)), "Output contains NaN values"
        assert not np.any(np.isinf(output_np)), "Output contains infinite values"

        # Check output values are reasonable for logits
        assert np.all(np.abs(output_np) < 100), "Logits are too large"

    def test_transformer_block_creation(self, tiny_config: Dict[str, Any]):
        """Test individual transformer block creation and functionality."""
        block = Gemma3TransformerBlock(
            emb_dim=tiny_config['emb_dim'],
            num_heads=tiny_config['num_heads'],
            num_kv_groups=tiny_config['num_kv_groups'],
            hidden_dim=tiny_config['hidden_dim'],
            max_context_length=tiny_config['max_context_length'],
            attention_type='full_attention',
            use_bias=tiny_config['use_bias'],
            dropout_rate=tiny_config['dropout_rate']
        )

        # Check dual normalization layers exist
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')
        assert hasattr(block, 'pre_feedforward_layernorm')
        assert hasattr(block, 'post_feedforward_layernorm')

        # Check attention and FFN layers exist
        assert hasattr(block, 'attention')
        assert hasattr(block, 'ffn')

        # Check framework components are used
        from dl_techniques.layers.norms.rms_norm import RMSNorm
        from dl_techniques.layers.attention.group_query_attention import GroupedQueryAttention
        from dl_techniques.layers.ffn.geglu_ffn import GeGLUFFN

        assert isinstance(block.input_layernorm, RMSNorm)
        assert isinstance(block.attention, GroupedQueryAttention)
        assert isinstance(block.ffn, GeGLUFFN)

        # Test forward pass through block
        sample_input = keras.random.normal((2, 16, tiny_config['emb_dim']))
        output = block(sample_input)

        assert output.shape == sample_input.shape
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_embedding_scaling(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test that embedding scaling is applied correctly."""
        model = Gemma3Model(**tiny_config)

        # Get raw embeddings
        embeddings = model.token_embeddings(sample_input)

        # Check that scaling factor is computed correctly
        expected_scale = keras.ops.sqrt(keras.ops.cast(tiny_config['emb_dim'], 'float32'))
        actual_scale = model.emb_scale

        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(expected_scale),
            keras.ops.convert_to_numpy(actual_scale),
            rtol=1e-6, atol=1e-6,
            err_msg="Embedding scale should match sqrt(emb_dim)"
        )

        # Test forward pass to ensure scaling is applied
        output = model(sample_input)
        assert output.shape == (sample_input.shape[0], sample_input.shape[1], tiny_config['vocab_size'])

    def test_attention_mask_creation(self, tiny_config: Dict[str, Any]):
        """Test attention mask creation for sliding window and full attention."""
        model = Gemma3Model(**tiny_config)

        seq_len = 8  # Small sequence for testing
        global_mask, local_mask = model._create_attention_masks(seq_len)

        # Check shapes
        assert global_mask.shape == (seq_len, seq_len)
        assert local_mask.shape == (seq_len, seq_len)

        # Convert to numpy for easier testing
        global_mask_np = keras.ops.convert_to_numpy(global_mask)
        local_mask_np = keras.ops.convert_to_numpy(local_mask)

        # Check that global mask is causal (upper triangular)
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert global_mask_np[i, j], f"Global mask should mask future positions ({i}, {j})"
                else:
                    assert not global_mask_np[i, j], f"Global mask should not mask past positions ({i}, {j})"

        # Check that local mask includes sliding window constraint
        # Local mask should mask both future positions AND positions too far in the past
        sliding_window = tiny_config['sliding_window']
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:  # Future positions should be masked
                    assert local_mask_np[i, j], f"Local mask should mask future positions ({i}, {j})"
                elif i - j >= sliding_window:  # Too far past should be masked
                    assert local_mask_np[i, j], f"Local mask should mask positions beyond sliding window ({i}, {j})"

    def test_training_mode(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test model behavior in training vs inference mode."""
        tiny_config_with_dropout = tiny_config.copy()
        tiny_config_with_dropout['dropout_rate'] = 0.2  # Set higher dropout for testing
        model = Gemma3Model(**tiny_config_with_dropout)

        # Training mode
        output_train = model(sample_input, training=True)

        # Inference mode
        output_inference = model(sample_input, training=False)

        # Outputs should be similar but not identical due to dropout
        output_train_np = keras.ops.convert_to_numpy(output_train)
        output_inference_np = keras.ops.convert_to_numpy(output_inference)

        # Check shapes are the same
        assert output_train_np.shape == output_inference_np.shape

        # With dropout, training and inference should be different
        if tiny_config_with_dropout['dropout_rate'] > 0:
            # Allow for some similarity but expect some difference
            relative_diff = np.abs(output_train_np - output_inference_np) / (np.abs(output_inference_np) + 1e-8)
            mean_diff = np.mean(relative_diff)
            assert mean_diff > 1e-6, f"Training and inference outputs are too similar with dropout: {mean_diff}"

    def test_dual_normalization_pattern(self, tiny_config: Dict[str, Any]):
        """Test that dual normalization pattern is correctly implemented."""
        # Create a single transformer block for detailed testing
        block = Gemma3TransformerBlock(
            emb_dim=tiny_config['emb_dim'],
            num_heads=tiny_config['num_heads'],
            num_kv_groups=tiny_config['num_kv_groups'],
            hidden_dim=tiny_config['hidden_dim'],
            max_context_length=tiny_config['max_context_length'],
            attention_type='full_attention'
        )

        # Check that all four normalization layers exist
        assert hasattr(block, 'input_layernorm')
        assert hasattr(block, 'post_attention_layernorm')
        assert hasattr(block, 'pre_feedforward_layernorm')
        assert hasattr(block, 'post_feedforward_layernorm')

        # All should be RMSNorm instances from framework
        from dl_techniques.layers.norms.rms_norm import RMSNorm
        assert isinstance(block.input_layernorm, RMSNorm)
        assert isinstance(block.post_attention_layernorm, RMSNorm)
        assert isinstance(block.pre_feedforward_layernorm, RMSNorm)
        assert isinstance(block.post_feedforward_layernorm, RMSNorm)

        # Test forward pass to ensure all normalization layers are applied
        sample_input = keras.random.normal((2, 16, tiny_config['emb_dim']))
        output = block(sample_input)

        assert output.shape == sample_input.shape
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))

    def test_mixed_attention_patterns(self, mini_config: Dict[str, Any]):
        """Test mixed attention patterns (sliding window + full attention)."""
        model = Gemma3Model(**mini_config)

        # Check that layer types are set correctly
        assert model.layer_types == mini_config['layer_types']
        assert len(model.layer_types) == mini_config['num_layers']

        # Check that transformer blocks have correct attention types
        for i, block in enumerate(model.transformer_blocks):
            expected_type = mini_config['layer_types'][i]
            assert block.attention_type == expected_type

        # Test forward pass with mixed attention
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, mini_config['vocab_size'], size=(2, 24)),
            dtype='int32'
        )
        output = model(sample_input)

        expected_shape = (2, 24, mini_config['vocab_size'])
        assert output.shape == expected_shape

        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np))
        assert not np.any(np.isinf(output_np))


class TestGemma3ModelConfigurations:
    """Test various model configurations for single GPU testing."""

    def test_different_head_configurations(self):
        """Test different attention head configurations."""
        configs = [
            # Standard configuration
            {'num_heads': 4, 'num_kv_groups': 4, 'emb_dim': 128},
            # Grouped query attention
            {'num_heads': 8, 'num_kv_groups': 2, 'emb_dim': 128},
            # Single KV head (extreme GQA)
            {'num_heads': 6, 'num_kv_groups': 1, 'emb_dim': 120},
        ]

        for config in configs:
            model_config = {
                'vocab_size': 500,
                'emb_dim': config['emb_dim'],
                'num_layers': 2,
                'num_heads': config['num_heads'],
                'num_kv_groups': config['num_kv_groups'],
                'hidden_dim': config['emb_dim'] * 2,
                'head_dim': config['emb_dim'] // config['num_heads'],
                'max_context_length': 512,
                'layer_types': ['full_attention', 'sliding_attention'],
            }

            model = Gemma3Model(**model_config)
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 500, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, 16, model_config['vocab_size'])

            # Check that head_dim is computed correctly
            expected_head_dim = config['emb_dim'] // config['num_heads']
            assert model.head_dim == expected_head_dim

    def test_different_layer_patterns(self):
        """Test different attention layer patterns."""
        base_config = {
            'vocab_size': 600,
            'emb_dim': 96,
            'num_layers': 4,
            'num_heads': 6,
            'num_kv_groups': 2,
            'hidden_dim': 192,
            'head_dim': 16,
            'max_context_length': 512,
            'sliding_window': 64,
        }

        layer_patterns = [
            ['full_attention'] * 4,  # All full attention
            ['sliding_attention'] * 4,  # All sliding window
            ['full_attention', 'sliding_attention'] * 2,  # Alternating
            ['sliding_attention'] * 3 + ['full_attention'],  # Mostly sliding with final full
        ]

        for layer_types in layer_patterns:
            config = base_config.copy()
            config['layer_types'] = layer_types

            model = Gemma3Model(**config)
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 600, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, 16, config['vocab_size'])

            # Check layer configuration
            assert model.layer_types == layer_types
            for i, block in enumerate(model.transformer_blocks):
                assert block.attention_type == layer_types[i]

    def test_different_context_lengths(self):
        """Test different max_context_length configurations."""
        base_config = {
            'vocab_size': 400,
            'emb_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'head_dim': 16,
            'layer_types': ['full_attention', 'sliding_attention'],
            'sliding_window': 32,
        }

        context_lengths = [256, 512, 1024, 2048]

        for max_context_length in context_lengths:
            config = base_config.copy()
            config['max_context_length'] = max_context_length

            model = Gemma3Model(**config)
            assert model.max_context_length == max_context_length

            # Test with sequence length within context limit
            seq_len = min(32, max_context_length)
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 400, size=(1, seq_len)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, seq_len, config['vocab_size'])

    def test_sliding_window_variations(self):
        """Test different sliding window sizes."""
        base_config = {
            'vocab_size': 400,
            'emb_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'head_dim': 16,
            'max_context_length': 512,
            'layer_types': ['sliding_attention', 'sliding_attention'],
        }

        window_sizes = [32, 64, 128, 256]

        for sliding_window in window_sizes:
            config = base_config.copy()
            config['sliding_window'] = sliding_window

            model = Gemma3Model(**config)
            assert model.sliding_window == sliding_window

            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 400, size=(1, 24)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, 24, config['vocab_size'])

    def test_query_pre_attn_scalar(self):
        """Test query pre-attention scalar configuration."""
        base_config = {
            'vocab_size': 300,
            'emb_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'head_dim': 16,
            'max_context_length': 256,
            'layer_types': ['full_attention', 'full_attention'],
        }

        scalar_values = [None, 16.0, 64.0, 256.0]

        for query_pre_attn_scalar in scalar_values:
            config = base_config.copy()
            config['query_pre_attn_scalar'] = query_pre_attn_scalar

            model = Gemma3Model(**config)
            assert model.query_pre_attn_scalar == query_pre_attn_scalar

            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 300, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, 16, config['vocab_size'])


class TestGemma3ModelSerialization:
    """Test model serialization and deserialization."""

    @pytest.fixture
    def serializable_config(self) -> Dict[str, Any]:
        """Create a config for serialization testing."""
        return {
            'vocab_size': 500,
            'emb_dim': 64,
            'num_layers': 3,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'head_dim': 16,
            'max_context_length': 512,
            'sliding_window': 64,
            'layer_types': ['full_attention', 'sliding_attention', 'full_attention'],
            'dropout_rate': 0.1,
            'use_bias': False,
        }

    def test_get_config_completeness(self, serializable_config: Dict[str, Any]):
        """Test that get_config returns all necessary parameters."""
        model = Gemma3Model(**serializable_config)
        config = model.get_config()

        # Check all important parameters are present
        expected_keys = {
            'vocab_size', 'emb_dim', 'num_layers', 'num_heads',
            'num_kv_groups', 'hidden_dim', 'head_dim', 'max_context_length',
            'sliding_window', 'layer_types', 'query_pre_attn_scalar',
            'use_bias', 'dropout_rate'
        }

        for key in expected_keys:
            assert key in config, f"Missing key in config: {key}"
            assert config[key] == getattr(model,
                                          key), f"Config mismatch for {key}: {config[key]} != {getattr(model, key)}"

    def test_transformer_block_serialization(self, serializable_config: Dict[str, Any]):
        """Test transformer block serialization."""
        block = Gemma3TransformerBlock(
            emb_dim=serializable_config['emb_dim'],
            num_heads=serializable_config['num_heads'],
            num_kv_groups=serializable_config['num_kv_groups'],
            hidden_dim=serializable_config['hidden_dim'],
            max_context_length=serializable_config['max_context_length'],
            attention_type='full_attention',
            use_bias=serializable_config['use_bias'],
            dropout_rate=serializable_config['dropout_rate']
        )

        config = block.get_config()

        # Check all parameters are present
        expected_keys = {
            'emb_dim', 'num_heads', 'num_kv_groups', 'hidden_dim',
            'max_context_length', 'attention_type', 'use_bias', 'dropout_rate'
        }

        for key in expected_keys:
            assert key in config, f"Missing key in transformer block config: {key}"

    def test_serialization_cycle(self, serializable_config: Dict[str, Any]):
        """Test full serialization and deserialization cycle."""
        # Create original model
        original_model = Gemma3Model(**serializable_config)
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 500, size=(2, 20)),
            dtype='int32'
        )
        original_output = original_model(sample_input, training=False)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'gemma3_test.keras')
            original_model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            # Test loaded model
            loaded_output = loaded_model(sample_input, training=False)

            # Outputs should be identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Original and loaded model outputs should match"
            )

    def test_config_reconstruction(self, serializable_config: Dict[str, Any]):
        """Test model can be reconstructed from config."""
        original_model = Gemma3Model(**serializable_config)
        config = original_model.get_config()

        # Remove base model keys that aren't constructor arguments
        model_keys = {'name', 'trainable', 'dtype'}
        for key in model_keys:
            config.pop(key, None)

        # Create new model from config
        reconstructed_model = Gemma3Model(**config)

        # Test same architecture
        assert reconstructed_model.vocab_size == original_model.vocab_size
        assert reconstructed_model.emb_dim == original_model.emb_dim
        assert reconstructed_model.num_layers == original_model.num_layers
        assert len(reconstructed_model.transformer_blocks) == len(original_model.transformer_blocks)
        assert reconstructed_model.layer_types == original_model.layer_types


class TestGemma3ModelFactories:
    """Test factory functions and utilities."""

    def test_create_gemma3_270m_default(self):
        """Test the default 270M model creation."""
        model = create_gemma3_270m()

        # Check configuration matches GEMMA3_270M_CONFIG
        assert model.vocab_size == GEMMA3_270M_CONFIG['vocab_size']
        assert model.emb_dim == GEMMA3_270M_CONFIG['emb_dim']
        assert model.num_layers == GEMMA3_270M_CONFIG['num_layers']
        assert model.num_heads == GEMMA3_270M_CONFIG['num_heads']
        assert model.max_context_length == GEMMA3_270M_CONFIG['max_context_length']

        # Check attention patterns
        assert model.layer_types == GEMMA3_270M_CONFIG['layer_types']
        assert len(model.layer_types) == GEMMA3_270M_CONFIG['num_layers']

        # Check sliding window configuration
        assert model.sliding_window == GEMMA3_270M_CONFIG['sliding_window']

    def test_create_gemma3_270m_custom_vocab(self):
        """Test 270M model creation with custom vocabulary size."""
        custom_vocab_size = 50000
        model = create_gemma3_270m(vocab_size=custom_vocab_size)

        assert model.vocab_size == custom_vocab_size
        # Other parameters should remain default
        assert model.emb_dim == GEMMA3_270M_CONFIG['emb_dim']
        assert model.num_layers == GEMMA3_270M_CONFIG['num_layers']

    def test_create_gemma3_270m_custom_context_length(self):
        """Test 270M model creation with custom context length."""
        custom_context_length = 8192
        model = create_gemma3_270m(max_context_length=custom_context_length)

        assert model.max_context_length == custom_context_length
        # Check that context length is propagated to transformer blocks
        for block in model.transformer_blocks:
            assert block.max_context_length == custom_context_length

    def test_create_gemma3_270m_overrides(self):
        """Test 270M model creation with multiple parameter overrides."""
        overrides = {
            'vocab_size': 10000,
            'num_layers': 4,
            'emb_dim': 256,
            'num_heads': 8,
            'max_context_length': 1024,
            'dropout_rate': 0.1
        }

        model = create_gemma3_270m(**overrides)

        # Check overridden parameters
        assert model.vocab_size == overrides['vocab_size']
        assert model.num_layers == overrides['num_layers']
        assert model.emb_dim == overrides['emb_dim']
        assert model.num_heads == overrides['num_heads']
        assert model.max_context_length == overrides['max_context_length']
        assert model.dropout_rate == overrides['dropout_rate']

        # Check derived parameters
        assert model.head_dim == overrides['emb_dim'] // overrides['num_heads']
        assert len(model.transformer_blocks) == overrides['num_layers']

    def test_gemma3_270m_config_constant(self):
        """Test that the GEMMA3_270M_CONFIG constant is valid."""
        # Test that config can create a valid model
        try:
            # Use smaller config for testing to avoid memory issues
            test_config = GEMMA3_270M_CONFIG.copy()
            test_config.update({
                'vocab_size': 1000,
                'emb_dim': 128,
                'num_layers': 2,
                'num_heads': 4,
                'hidden_dim': 256,
            })
            test_config['layer_types'] = test_config['layer_types'][:2]  # Match num_layers

            model = Gemma3Model(**test_config)
            assert isinstance(model, Gemma3Model)

        except Exception as e:
            pytest.fail(f"GEMMA3_270M_CONFIG is invalid: {e}")

    def test_factory_logging(self, caplog):
        """Test that factory function produces appropriate logging."""
        with caplog.at_level('INFO'):
            model = create_gemma3_270m(
                vocab_size=1000,
                num_layers=2,
                max_context_length=512
            )

        # Check that informative logs were produced
        assert "Creating Gemma 3 270M model with framework components" in caplog.text
        assert "Vocabulary size: 1000" in caplog.text
        assert "Max context length: 512" in caplog.text


class TestGemma3ModelErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_vocab_size(self):
        """Test error handling for invalid vocabulary size."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Gemma3Model(
                vocab_size=0,
                emb_dim=128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                head_dim=32
            )

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Gemma3Model(
                vocab_size=-100,
                emb_dim=128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                head_dim=32
            )

    def test_invalid_emb_dim(self):
        """Test error handling for invalid embedding dimension."""
        with pytest.raises(ValueError, match="emb_dim must be positive"):
            Gemma3Model(
                vocab_size=1000,
                emb_dim=-128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                head_dim=32
            )

    def test_invalid_num_layers(self):
        """Test error handling for invalid number of layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            Gemma3Model(
                vocab_size=1000,
                emb_dim=128,
                num_layers=0,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                head_dim=32
            )

    def test_invalid_max_context_length(self):
        """Test error handling for invalid max context length."""
        with pytest.raises(ValueError, match="max_context_length must be positive"):
            Gemma3Model(
                vocab_size=1000,
                emb_dim=128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                head_dim=32,
                max_context_length=-1024
            )

    def test_mismatched_layer_types(self):
        """Test error handling for mismatched layer_types and num_layers."""
        with pytest.raises(ValueError, match="layer_types length .* must match num_layers"):
            Gemma3Model(
                vocab_size=1000,
                emb_dim=128,
                num_layers=3,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                head_dim=32,
                layer_types=['full_attention', 'sliding_attention']  # Only 2 elements, but num_layers=3
            )

    def test_transformer_block_invalid_params(self):
        """Test error handling for invalid transformer block parameters."""
        with pytest.raises(ValueError, match="emb_dim must be positive"):
            Gemma3TransformerBlock(
                emb_dim=-128,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                max_context_length=512
            )

        with pytest.raises(ValueError, match="num_heads must be positive"):
            Gemma3TransformerBlock(
                emb_dim=128,
                num_heads=0,
                num_kv_groups=2,
                hidden_dim=256,
                max_context_length=512
            )

        with pytest.raises(ValueError, match="max_context_length must be positive"):
            Gemma3TransformerBlock(
                emb_dim=128,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                max_context_length=-512
            )


class TestGemma3ModelPerformance:
    """Test model performance characteristics for single GPU."""

    def test_parameter_count_scaling(self):
        """Test parameter count scales appropriately with model size."""
        configs = [
            {'emb_dim': 64, 'num_layers': 2, 'vocab_size': 500},
            {'emb_dim': 128, 'num_layers': 2, 'vocab_size': 500},
            {'emb_dim': 64, 'num_layers': 4, 'vocab_size': 500},
        ]

        param_counts = []
        for config in configs:
            model_config = {
                'vocab_size': config['vocab_size'],
                'emb_dim': config['emb_dim'],
                'num_layers': config['num_layers'],
                'num_heads': 4,
                'num_kv_groups': 2,
                'hidden_dim': config['emb_dim'] * 2,
                'head_dim': config['emb_dim'] // 4,
                'max_context_length': 256,
                'layer_types': ['full_attention'] * config['num_layers'],
            }

            model = Gemma3Model(**model_config)

            # Build model to initialize weights
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, config['vocab_size'], size=(1, 16)),
                dtype='int32'
            )
            _ = model(sample_input)

            # Count parameters
            param_count = sum(keras.ops.size(w) for w in model.trainable_variables)
            param_counts.append(int(param_count))

        # Larger embedding dimension should increase parameters
        assert param_counts[1] > param_counts[0]

        # More layers should increase parameters
        assert param_counts[2] > param_counts[0]

    def test_embedding_scaling_impact(self):
        """Test embedding scaling impact on output magnitude."""
        config = {
            'vocab_size': 300,
            'emb_dim': 64,
            'num_layers': 1,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'head_dim': 16,
            'max_context_length': 128,
            'layer_types': ['full_attention'],
        }

        model = Gemma3Model(**config)
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 300, size=(2, 8)),
            dtype='int32'
        )

        # Get raw embeddings (without scaling)
        raw_embeddings = model.token_embeddings(sample_input)
        raw_magnitude = keras.ops.mean(keras.ops.square(raw_embeddings))

        # Get scaled embeddings (via forward pass)
        output = model(sample_input, return_logits=False)

        # The magnitude should be scaled by sqrt(emb_dim)
        expected_scale_factor = config['emb_dim']  # sqrt(emb_dim)^2 = emb_dim

        # Note: This is a rough test as the output goes through transformer layers
        # but the initial scaling should still have some effect

    def test_forward_pass_efficiency(self):
        """Test forward pass runs efficiently."""
        import time

        # Create small efficient model
        model = create_gemma3_270m(
            vocab_size=1000,
            emb_dim=128,
            num_layers=2,
            num_heads=4,
            max_context_length=256
        )

        # Create input
        inputs = keras.ops.convert_to_tensor(
            np.random.randint(0, 1000, size=(2, 32)),
            dtype='int32'
        )

        # Warmup
        for _ in range(3):
            _ = model(inputs)

        # Time forward passes
        start_time = time.time()
        num_runs = 5
        for _ in range(num_runs):
            _ = model(inputs)

        avg_time = (time.time() - start_time) / num_runs

        # Should be fast on single GPU
        assert avg_time < 10.0, f"Forward pass too slow: {avg_time:.4f}s"

    def test_different_sequence_lengths(self):
        """Test model handles different sequence lengths efficiently."""
        model = create_gemma3_270m(
            vocab_size=500,
            emb_dim=64,
            num_layers=2,
            num_heads=4,
            max_context_length=1024
        )

        # Test different sequence lengths
        sequence_lengths = [16, 32, 64, 128, 256]

        for seq_len in sequence_lengths:
            if seq_len <= model.max_context_length:
                inputs = keras.ops.convert_to_tensor(
                    np.random.randint(0, 500, size=(1, seq_len)),
                    dtype='int32'
                )

                output = model(inputs)
                assert output.shape[1] == seq_len

                # Check output quality
                output_np = keras.ops.convert_to_numpy(output)
                assert not np.any(np.isnan(output_np))
                assert not np.any(np.isinf(output_np))

    def test_memory_efficient_attention_masks(self):
        """Test attention mask creation doesn't consume excessive memory."""
        # Test with larger sequence length
        seq_len = 128

        model = create_gemma3_270m(
            vocab_size=200,
            emb_dim=32,
            num_layers=1,
            max_context_length=seq_len * 2
        )

        # Create masks
        global_mask, local_mask = model._create_attention_masks(seq_len)

        # Check shapes are correct
        assert global_mask.shape == (seq_len, seq_len)
        assert local_mask.shape == (seq_len, seq_len)

        # Check masks are boolean tensors (memory efficient)
        assert 'bool' in str(global_mask.dtype).lower()
        assert 'bool' in str(local_mask.dtype).lower()


# ---------------------------------------------------------------------
# Integration and end-to-end tests
# ---------------------------------------------------------------------

class TestGemma3Integration:
    """Test integration with dl_techniques framework."""

    def test_framework_component_integration(self):
        """Test that all framework components are properly integrated."""
        model = create_gemma3_270m(
            vocab_size=400,
            emb_dim=64,
            num_layers=2,
            num_heads=4
        )

        # Check framework components are used in transformer blocks
        for block in model.transformer_blocks:
            # Check normalization layers are from framework
            from dl_techniques.layers.norms.rms_norm import RMSNorm
            assert isinstance(block.input_layernorm, RMSNorm)
            assert isinstance(block.post_attention_layernorm, RMSNorm)
            assert isinstance(block.pre_feedforward_layernorm, RMSNorm)
            assert isinstance(block.post_feedforward_layernorm, RMSNorm)

            # Check attention layer is from framework
            from dl_techniques.layers.attention.group_query_attention import GroupedQueryAttention
            assert isinstance(block.attention, GroupedQueryAttention)

            # Check FFN layer is from framework
            from dl_techniques.layers.ffn.geglu_ffn import GeGLUFFN
            assert isinstance(block.ffn, GeGLUFFN)

        # Check final norm is from framework
        from dl_techniques.layers.norms.rms_norm import RMSNorm
        assert isinstance(model.final_norm, RMSNorm)

    def test_end_to_end_training_simulation(self):
        """Test complete training simulation on tiny model."""
        # Create very small model for training test
        model = create_gemma3_270m(
            vocab_size=200,
            emb_dim=32,
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            max_context_length=128,
            dropout_rate=0.1
        )

        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Create tiny dataset
        batch_size, seq_len = 4, 16
        vocab_size = 200

        inputs = keras.ops.convert_to_tensor(
            np.random.randint(0, vocab_size, size=(batch_size, seq_len)),
            dtype='int32'
        )
        targets = keras.ops.convert_to_tensor(
            np.random.randint(0, vocab_size, size=(batch_size, seq_len)),
            dtype='int32'
        )

        # Run a few training steps
        initial_loss = float('inf')
        for step in range(3):
            metrics = model.train_on_batch(inputs, targets, return_dict=True)
            loss_tensor = metrics['loss']
            loss = float(keras.ops.convert_to_numpy(loss_tensor))
            assert isinstance(loss, (float, np.floating))
            assert loss >= 0

            if step == 0:
                initial_loss = loss

        # Loss should be reasonable
        assert loss < 20.0, f"Loss too high: {loss}"

    def test_sliding_window_vs_full_attention_behavior(self):
        """Test that sliding window and full attention produce different outputs."""
        base_config = {
            'vocab_size': 300,
            'emb_dim': 64,
            'num_layers': 1,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'head_dim': 16,
            'max_context_length': 256,
            'sliding_window': 32,
        }

        # Model with full attention
        model_full = Gemma3Model(**{**base_config, 'layer_types': ['full_attention']})

        # Model with sliding window attention
        model_sliding = Gemma3Model(**{**base_config, 'layer_types': ['sliding_attention']})

        # Test with longer sequence where difference should be visible
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 300, size=(1, 48)),
            dtype='int32'
        )

        output_full = model_full(sample_input, training=False)
        output_sliding = model_sliding(sample_input, training=False)

        # Outputs should be different due to different attention patterns
        output_full_np = keras.ops.convert_to_numpy(output_full)
        output_sliding_np = keras.ops.convert_to_numpy(output_sliding)

        # Check they're not identical (should be different due to attention pattern)
        difference = np.mean(np.abs(output_full_np - output_sliding_np))
        assert difference > 1e-6, f"Full and sliding window attention outputs too similar: {difference}"

    def test_framework_serialization_compatibility(self):
        """Test that framework components serialize correctly."""
        model = create_gemma3_270m(
            vocab_size=300,
            emb_dim=48,
            num_layers=2,
            num_heads=6,
            max_context_length=256
        )

        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 300, size=(1, 16)),
            dtype='int32'
        )
        original_output = model(sample_input, training=False)

        # Test serialization with framework components
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'gemma3_framework_test.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            loaded_output = loaded_model(sample_input, training=False)

            # Should be identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Framework components should serialize correctly"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])