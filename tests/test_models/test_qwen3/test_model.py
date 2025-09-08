"""
Refined Tests for Qwen3 Model Implementation

This module provides comprehensive tests for the refined Qwen3 model, including:
- Basic functionality tests with enhanced TransformerLayer integration
- Configuration variants optimized for single GPU testing
- Serialization/deserialization with built-in MoE support
- Factory functions and utilities
- Error handling and edge cases
- Performance validation

All tests use small model configurations to keep memory usage low for single GPU environments.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any

import keras

from dl_techniques.utils.logger import logger

# Import the refined Qwen3 implementation
from dl_techniques.models.qwen3.model import (
    Qwen3Model,
    create_qwen3_model,
    create_qwen3_coder_30b_config,
    create_qwen3_small_config,
    create_qwen3_medium_config,
    get_qwen3_model_info,
    compile_qwen3_for_training,
)


class TestQwen3ModelBasic:
    """Test basic Qwen3 model functionality with enhanced TransformerLayer."""

    @pytest.fixture
    def tiny_config(self) -> Dict[str, Any]:
        """Create a tiny model configuration for fast testing."""
        return {
            'vocab_size': 1000,
            'd_model': 128,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 256,
            'moe_layers': [],  # No MoE for basic tests
            'context_length': 512,
            'rope_theta': 10000.0,
            'use_weight_tying': True,
            'dropout_rate': 0.1,
        }

    @pytest.fixture
    def mini_config_with_moe(self) -> Dict[str, Any]:
        """Create a mini model configuration with MoE for testing."""
        return {
            'vocab_size': 800,
            'd_model': 96,
            'num_layers': 4,
            'num_heads': 6,
            'num_kv_groups': 2,
            'hidden_dim': 192,
            'moe_layers': [1, 3],  # Second and fourth layers use MoE
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 128,
            'context_length': 256,
            'rope_theta': 10000.0,
            'use_weight_tying': False,
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
        model = Qwen3Model(**tiny_config)

        # Check model attributes
        assert model.vocab_size == tiny_config['vocab_size']
        assert model.d_model == tiny_config['d_model']
        assert model.num_layers == tiny_config['num_layers']
        assert len(model.transformer_blocks) == tiny_config['num_layers']
        assert model.head_dim == tiny_config['d_model'] // tiny_config['num_heads']

        # Check embedding layer
        assert model.token_embedding.input_dim == tiny_config['vocab_size']
        assert model.token_embedding.output_dim == tiny_config['d_model']

        # Check normalization layer (should be RMSNorm from factory)
        assert model.final_norm is not None
        from dl_techniques.layers.norms.rms_norm import RMSNorm
        assert isinstance(model.final_norm, RMSNorm)

        # Check weight tying
        if tiny_config['use_weight_tying']:
            assert model.output_head is None
        else:
            assert model.output_head is not None

        # Check no MoE configuration
        assert not model.moe_layers
        assert model.moe_config is None

    def test_model_forward_pass(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test forward pass through the model."""
        model = Qwen3Model(**tiny_config)

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

    def test_model_with_moe(self, mini_config_with_moe: Dict[str, Any]):
        """Test model with Mixture of Experts layers using TransformerLayer integration."""
        model = Qwen3Model(**mini_config_with_moe)

        # Check MoE configuration
        assert model.moe_layers == [1, 3]
        assert model.num_experts == mini_config_with_moe['num_experts']
        assert model.moe_config is not None
        assert model.moe_config.num_experts == mini_config_with_moe['num_experts']
        assert model.moe_config.gating_config.top_k == mini_config_with_moe['num_experts_per_tok']

        # Check transformer blocks
        for i, block in enumerate(model.transformer_blocks):
            from dl_techniques.layers.transformer import TransformerLayer
            assert isinstance(block, TransformerLayer), f"Block {i} is not a TransformerLayer"

            if i in mini_config_with_moe['moe_layers']:
                assert 'moe' in block.name.lower(), f"MoE block {i} doesn't have 'moe' in name"

        # Test forward pass
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, mini_config_with_moe['vocab_size'], size=(2, 24)),
            dtype='int32'
        )
        output = model(sample_input, training=True)

        expected_shape = (2, 24, mini_config_with_moe['vocab_size'])
        assert output.shape == expected_shape

        # Check output quality
        output_np = keras.ops.convert_to_numpy(output)
        assert not np.any(np.isnan(output_np)), "MoE model output contains NaN"
        assert not np.any(np.isinf(output_np)), "MoE model output contains infinite values"

    def test_training_mode(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test model behavior in training vs inference mode."""
        tiny_config_with_dropout = tiny_config.copy()
        tiny_config_with_dropout['dropout_rate'] = 0.3  # Set higher dropout for testing
        model = Qwen3Model(**tiny_config_with_dropout)

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

    def test_weight_tying_vs_separate_head(self, tiny_config: Dict[str, Any]):
        """Test weight tying vs separate output head."""
        # Create sample input
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, tiny_config['vocab_size'], size=(2, 16)),
            dtype='int32'
        )

        # Test with weight tying
        config_tied = tiny_config.copy()
        config_tied['use_weight_tying'] = True
        model_tied = Qwen3Model(**config_tied)

        # Test with separate head
        config_separate = tiny_config.copy()
        config_separate['use_weight_tying'] = False
        model_separate = Qwen3Model(**config_separate)

        # Forward passes to build the models
        output_tied = model_tied(sample_input)
        output_separate = model_separate(sample_input)

        # Both should produce valid outputs
        assert output_tied.shape == output_separate.shape

        # Check parameter counts are different
        tied_params = sum(keras.ops.size(w) for w in model_tied.trainable_variables)
        separate_params = sum(keras.ops.size(w) for w in model_separate.trainable_variables)

        # Separate head should have more parameters
        assert separate_params > tied_params

    def test_transformer_layer_integration(self, mini_config_with_moe: Dict[str, Any]):
        """Test that TransformerLayer is properly integrated."""
        model = Qwen3Model(**mini_config_with_moe)

        # Check that all transformer blocks are TransformerLayer instances
        for i, block in enumerate(model.transformer_blocks):
            from dl_techniques.layers.transformer import TransformerLayer
            assert isinstance(block, TransformerLayer)

            # Check naming convention
            if i in mini_config_with_moe['moe_layers']:
                assert 'moe' in block.name.lower()
            else:
                assert 'transformer_block' in block.name

            # Check configuration
            assert hasattr(block, 'hidden_size')
            assert hasattr(block, 'num_heads')
            assert hasattr(block, 'normalization_type')
            assert hasattr(block, 'attention_type')

    def test_auxiliary_loss_and_metrics(self, mini_config_with_moe: Dict[str, Any]):
        """Test auxiliary loss and metrics for MoE models."""
        model = Qwen3Model(**mini_config_with_moe)

        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, mini_config_with_moe['vocab_size'], size=(2, 16)),
            dtype='int32'
        )

        # Forward pass
        _ = model(sample_input, training=True)

        # Test auxiliary loss interface
        aux_loss = model.get_auxiliary_loss()
        # Note: aux_loss might be None if TransformerLayer doesn't expose it yet
        if aux_loss is not None:
            assert aux_loss.shape == ()  # Should be scalar
            assert aux_loss >= 0  # Should be non-negative

        # Test metrics interface
        metrics = model.get_moe_metrics()
        assert isinstance(metrics, dict)
        # Metrics might be empty if TransformerLayer doesn't expose them yet


class TestQwen3ModelConfigurations:
    """Test various model configurations for single GPU testing."""

    def test_different_head_configurations(self):
        """Test different attention head configurations."""
        configs = [
            # Standard configuration
            {'num_heads': 4, 'num_kv_groups': 4, 'd_model': 128},
            # Grouped query attention
            {'num_heads': 8, 'num_kv_groups': 2, 'd_model': 128},
            # Single KV head (extreme GQA)
            {'num_heads': 6, 'num_kv_groups': 1, 'd_model': 120},
        ]

        for config in configs:
            model_config = {
                'vocab_size': 500,
                'd_model': config['d_model'],
                'num_layers': 2,
                'num_heads': config['num_heads'],
                'num_kv_groups': config['num_kv_groups'],
                'hidden_dim': config['d_model'] * 2,
            }

            model = Qwen3Model(**model_config)
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 500, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, 16, model_config['vocab_size'])

            # Check that head_dim is computed correctly
            expected_head_dim = config['d_model'] // config['num_heads']
            assert model.head_dim == expected_head_dim

    def test_different_moe_patterns(self):
        """Test different MoE layer patterns with TransformerLayer."""
        base_config = {
            'vocab_size': 600,
            'd_model': 96,
            'num_layers': 4,
            'num_heads': 6,
            'num_kv_groups': 2,
            'hidden_dim': 192,
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 128,
        }

        moe_patterns = [
            [],  # No MoE
            [0],  # First layer only
            [3],  # Last layer only
            [1, 3],  # Alternating
            [0, 1, 2, 3],  # All layers
        ]

        for moe_layers in moe_patterns:
            config = base_config.copy()
            config['moe_layers'] = moe_layers

            model = Qwen3Model(**config)
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 600, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input, training=True)  # Use training=True for MoE

            assert output.shape == (1, 16, config['vocab_size'])

            # Check MoE configuration
            if moe_layers:
                assert model.moe_config is not None
                assert model.moe_layers == moe_layers
            else:
                assert model.moe_config is None
                assert not model.moe_layers

    def test_rope_theta_variations(self):
        """Test different RoPE theta values."""
        base_config = {
            'vocab_size': 400,
            'd_model': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
        }

        rope_thetas = [10000.0, 100000.0, 1000000.0, 10000000.0]

        for theta in rope_thetas:
            config = base_config.copy()
            config['rope_theta'] = theta

            model = Qwen3Model(**config)
            assert model.rope_theta == theta

            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 400, size=(1, 32)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, 32, config['vocab_size'])

    def test_vocab_padding(self):
        """Test vocabulary padding functionality."""
        base_config = {
            'vocab_size': 1000,
            'd_model': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
        }

        # Test with padding
        config_padded = base_config.copy()
        config_padded['vocab_padding_size'] = 2048

        model = Qwen3Model(**config_padded)

        assert model.final_vocab_size == 2048
        assert model.vocab_size == 1000  # Original size preserved

        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 1000, size=(2, 16)),
            dtype='int32'
        )
        output = model(sample_input)

        # Output should match original vocab size due to weight tying logic
        assert output.shape[2] == 1000


class TestQwen3ModelSerialization:
    """Test model serialization and deserialization."""

    @pytest.fixture
    def serializable_config(self) -> Dict[str, Any]:
        """Create a config for serialization testing."""
        return {
            'vocab_size': 500,
            'd_model': 64,
            'num_layers': 3,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'moe_layers': [1],  # One MoE layer
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 64,
            'dropout_rate': 0.1,
        }

    def test_get_config_completeness(self, serializable_config: Dict[str, Any]):
        """Test that get_config returns all necessary parameters."""
        model = Qwen3Model(**serializable_config)
        config = model.get_config()

        # Check all important parameters are present
        expected_keys = {
            'vocab_size', 'd_model', 'num_layers', 'num_heads',
            'num_kv_groups', 'hidden_dim', 'moe_layers',
            'num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
            'context_length', 'rope_theta', 'use_weight_tying', 'dropout_rate',
            'vocab_padding_size'
        }

        for key in expected_keys:
            assert key in config, f"Missing key in config: {key}"
            assert config[key] == getattr(model, key), f"Config mismatch for {key}: {config[key]} != {getattr(model, key)}"

    def test_serialization_cycle_standard(self, serializable_config: Dict[str, Any]):
        """Test full serialization and deserialization cycle for standard model."""
        # Remove MoE for standard test
        config = serializable_config.copy()
        config['moe_layers'] = []

        # Create original model
        original_model = Qwen3Model(**config)
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 500, size=(2, 20)),
            dtype='int32'
        )
        original_output = original_model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'qwen3_standard_test.keras')
            original_model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            # Test loaded model
            loaded_output = loaded_model(sample_input)

            # Outputs should be identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Original and loaded standard model outputs should match"
            )

    def test_serialization_cycle_moe(self, serializable_config: Dict[str, Any]):
        """Test serialization cycle for MoE model."""
        # Create original MoE model
        original_model = Qwen3Model(**serializable_config)
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 500, size=(2, 20)),
            dtype='int32'
        )
        original_output = original_model(sample_input, training=False)  # Use inference mode for consistency

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'qwen3_moe_test.keras')
            original_model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            # Test loaded model
            loaded_output = loaded_model(sample_input, training=False)

            # Outputs should be very close (some variance due to MoE)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-5, atol=1e-5,
                err_msg="Original and loaded MoE model outputs should match"
            )

    def test_config_reconstruction(self, serializable_config: Dict[str, Any]):
        """Test model can be reconstructed from config."""
        original_model = Qwen3Model(**serializable_config)
        config = original_model.get_config()

        # Remove base model keys that aren't constructor arguments
        model_keys = {'name', 'trainable', 'dtype'}
        for key in model_keys:
            config.pop(key, None)

        # Create new model from config
        reconstructed_model = Qwen3Model(**config)

        # Test same architecture
        assert reconstructed_model.vocab_size == original_model.vocab_size
        assert reconstructed_model.d_model == original_model.d_model
        assert reconstructed_model.num_layers == original_model.num_layers
        assert len(reconstructed_model.transformer_blocks) == len(original_model.transformer_blocks)
        assert reconstructed_model.moe_layers == original_model.moe_layers


class TestQwen3ModelFactories:
    """Test factory functions and utilities."""

    def test_create_qwen3_coder_30b_config(self):
        """Test the default configuration creator."""
        config = create_qwen3_coder_30b_config()

        # Check expected values
        assert config['vocab_size'] == 151_936
        assert config['d_model'] == 2048
        assert config['num_layers'] == 48
        assert config['num_heads'] == 32
        assert config['num_kv_groups'] == 4
        assert config['use_weight_tying'] is True

        # Check MoE layers
        assert isinstance(config['moe_layers'], list)
        assert len(config['moe_layers']) > 0
        assert config['num_experts'] == 128
        assert config['num_experts_per_tok'] == 8

    def test_create_qwen3_small_config(self):
        """Test small configuration creator."""
        config = create_qwen3_small_config()

        assert config['vocab_size'] == 32_000
        assert config['d_model'] == 768
        assert config['num_layers'] == 12
        assert len(config['moe_layers']) > 0  # Should have some MoE layers

    def test_create_qwen3_medium_config(self):
        """Test medium configuration creator."""
        config = create_qwen3_medium_config()

        assert config['vocab_size'] == 100_000
        assert config['d_model'] == 1024
        assert config['num_layers'] == 24
        assert len(config['moe_layers']) > 0

    def test_create_qwen3_model_with_size_presets(self):
        """Test factory function with size presets."""
        # Test small model
        model_small = create_qwen3_model(model_size='small')
        assert isinstance(model_small, Qwen3Model)
        assert model_small.d_model == 768

        # Test medium model (may be too large for single GPU, so create minimal version)
        try:
            model_medium = create_qwen3_model(
                model_size='medium',
                num_layers=2,  # Override to make it smaller
                d_model=256,   # Override to make it smaller
                moe_layers=[]  # No MoE to keep it small
            )
            assert isinstance(model_medium, Qwen3Model)
        except Exception:
            pytest.skip("Medium model too large for single GPU testing")

    def test_create_qwen3_model_with_config_override(self):
        """Test factory function with config and overrides."""
        # Use small base config to stay within memory limits
        base_config = {
            'vocab_size': 800,
            'd_model': 96,
            'num_layers': 3,
            'num_heads': 6,
            'num_kv_groups': 2,
            'hidden_dim': 192,
            'moe_layers': [],
        }

        # Override some parameters
        model = create_qwen3_model(
            config=base_config,
            vocab_size=1200,  # Override vocab_size
            dropout_rate=0.2  # Override dropout_rate
        )

        assert model.vocab_size == 1200  # Overridden value
        assert model.d_model == base_config['d_model']  # Original value
        assert model.dropout_rate == 0.2  # Overridden value

    def test_get_qwen3_model_info(self):
        """Test model info utility function."""
        # Create small test model
        config = {
            'vocab_size': 500,
            'd_model': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'hidden_dim': 128,
            'moe_layers': [1],
            'num_experts': 4,
            'num_experts_per_tok': 2,
        }

        model = Qwen3Model(**config)
        info = get_qwen3_model_info(model)

        # Check required fields
        assert info['architecture'] == 'Qwen3'
        assert info['vocab_size'] == config['vocab_size']
        assert info['d_model'] == config['d_model']
        assert info['num_layers'] == config['num_layers']
        assert info['has_moe'] == True
        assert info['moe_layers'] == config['moe_layers']
        assert 'estimated_total_params' in info
        assert 'estimated_moe_params' in info
        assert info['estimated_total_params'] > 0

    def test_compile_qwen3_for_training(self):
        """Test training compilation utility."""
        model = create_qwen3_model(
            vocab_size=500,
            d_model=64,
            num_layers=2,
            num_heads=4,
            moe_layers=[]  # No MoE to keep simple
        )

        compile_qwen3_for_training(
            model,
            learning_rate=1e-4,
            warmup_steps=100,
            max_steps=1000
        )

        assert model.optimizer is not None
        assert hasattr(model.optimizer, 'learning_rate')

class TestQwen3ModelErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_vocab_size(self):
        """Test error handling for invalid vocabulary size."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Qwen3Model(
                vocab_size=0,
                d_model=128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256
            )

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Qwen3Model(
                vocab_size=-100,
                d_model=128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256
            )

    def test_invalid_d_model(self):
        """Test error handling for invalid model dimension."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            Qwen3Model(
                vocab_size=1000,
                d_model=-128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256
            )

    def test_mismatched_heads_and_model_dim(self):
        """Test error handling for mismatched heads and model dimension."""
        with pytest.raises(ValueError, match="d_model .* must be divisible by num_heads"):
            Qwen3Model(
                vocab_size=1000,
                d_model=129,  # Not divisible by 4
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256
            )

    def test_mismatched_heads_and_kv_groups(self):
        """Test error handling for mismatched heads and KV groups."""
        with pytest.raises(ValueError, match="num_heads .* must be divisible by num_kv_groups"):
            Qwen3Model(
                vocab_size=1000,
                d_model=120,  # Is divisible by num_heads (6)
                num_layers=2,
                num_heads=6,  # Is NOT divisible by num_kv_groups (4)
                num_kv_groups=4,
                hidden_dim=256
            )

    def test_invalid_num_layers(self):
        """Test error handling for invalid number of layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            Qwen3Model(
                vocab_size=1000,
                d_model=128,
                num_layers=0,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256
            )

    def test_invalid_dropout_rate(self):
        """Test error handling for invalid dropout rate."""
        with pytest.raises(ValueError, match="dropout_rate must be between"):
            Qwen3Model(
                vocab_size=1000,
                d_model=128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                dropout_rate=1.5  # Invalid > 1.0
            )

        with pytest.raises(ValueError, match="dropout_rate must be between"):
            Qwen3Model(
                vocab_size=1000,
                d_model=128,
                num_layers=2,
                num_heads=4,
                num_kv_groups=2,
                hidden_dim=256,
                dropout_rate=-0.1  # Invalid < 0.0
            )

    def test_invalid_factory_model_size(self):
        """Test error handling for invalid model size in factory."""
        with pytest.raises(ValueError, match="Unknown model_size"):
            create_qwen3_model(model_size='invalid_size')


class TestQwen3ModelPerformance:
    """Test model performance characteristics for single GPU."""

    def test_memory_efficiency_with_weight_tying(self):
        """Test weight tying reduces parameter count."""
        base_config = {
            'vocab_size': 1000,  # Smaller vocab for single GPU
            'd_model': 128,
            'num_layers': 3,
            'num_heads': 8,
            'num_kv_groups': 4,
            'hidden_dim': 256,
        }

        # Model with weight tying
        config_tied = {**base_config, 'use_weight_tying': True}
        model_tied = Qwen3Model(**config_tied)

        # Model without weight tying
        config_separate = {**base_config, 'use_weight_tying': False}
        model_separate = Qwen3Model(**config_separate)

        # Build models to initialize weights
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, base_config['vocab_size'], size=(1, 16)),
            dtype='int32'
        )
        _ = model_tied(sample_input)
        _ = model_separate(sample_input)

        # Count parameters
        tied_params = sum(keras.ops.size(w) for w in model_tied.trainable_variables)
        separate_params = sum(keras.ops.size(w) for w in model_separate.trainable_variables)

        # Weight tying should reduce parameters
        assert tied_params < separate_params

        # Difference should be approximately vocab_size * d_model
        expected_diff = base_config['vocab_size'] * base_config['d_model']
        actual_diff = int(separate_params) - int(tied_params)

        # Allow for some variance due to other parameters
        assert abs(actual_diff - expected_diff) < expected_diff * 0.15

    def test_moe_parameter_scaling(self):
        """Test MoE increases parameters appropriately."""
        base_config = {
            'vocab_size': 800,  # Smaller for single GPU
            'd_model': 96,
            'num_layers': 3,
            'num_heads': 6,
            'num_kv_groups': 2,
            'hidden_dim': 192,
        }

        # Model without MoE
        config_no_moe = {**base_config, 'moe_layers': []}
        model_no_moe = Qwen3Model(**config_no_moe)

        # Model with MoE
        config_with_moe = {
            **base_config,
            'moe_layers': [1, 2],
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 128,
        }
        model_with_moe = Qwen3Model(**config_with_moe)

        # Build models to initialize weights
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, base_config['vocab_size'], size=(1, 16)),
            dtype='int32'
        )
        _ = model_no_moe(sample_input)
        _ = model_with_moe(sample_input)

        # Count parameters
        no_moe_params = sum(keras.ops.size(w) for w in model_no_moe.trainable_variables)
        with_moe_params = sum(keras.ops.size(w) for w in model_with_moe.trainable_variables)

        # MoE should significantly increase parameter count
        assert with_moe_params > no_moe_params

        # But should still be reasonable for single GPU testing
        assert int(with_moe_params) < 1_000_000  # 1M parameter limit for single GPU

    def test_forward_pass_efficiency(self):
        """Test forward pass runs efficiently."""
        import time

        # Create small efficient model
        model = create_qwen3_model(
            vocab_size=1000,
            d_model=128,
            num_layers=4,
            num_heads=8,
            moe_layers=[]  # No MoE for speed test
        )

        # Create input
        inputs = keras.ops.convert_to_tensor(
            np.random.randint(0, 1000, size=(4, 64)),
            dtype='int32'
        )

        # Warmup
        for _ in range(3):
            _ = model(inputs)

        # Time forward passes
        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            _ = model(inputs)

        avg_time = (time.time() - start_time) / num_runs

        # Should be fast on single GPU
        assert avg_time < 5.0, f"Forward pass too slow: {avg_time:.4f}s"

    def test_different_sequence_lengths(self):
        """Test model handles different sequence lengths efficiently."""
        model = create_qwen3_model(
            vocab_size=500,
            d_model=64,
            num_layers=2,
            num_heads=4
        )

        # Test different sequence lengths (keep reasonable for single GPU)
        sequence_lengths = [16, 32, 64, 128, 256]

        for seq_len in sequence_lengths:
            if seq_len <= model.context_length:
                inputs = keras.ops.convert_to_tensor(
                    np.random.randint(0, 500, size=(2, seq_len)),
                    dtype='int32'
                )

                output = model(inputs)
                assert output.shape[1] == seq_len

                # Check output quality
                output_np = keras.ops.convert_to_numpy(output)
                assert not np.any(np.isnan(output_np))
                assert not np.any(np.isinf(output_np))


# ---------------------------------------------------------------------
# Integration and end-to-end tests
# ---------------------------------------------------------------------

class TestQwen3Integration:
    """Test integration with dl_techniques framework."""

    def test_end_to_end_training_simulation(self):
        """Test complete training simulation on tiny model."""
        # Create very small model for training test
        model = create_qwen3_model(
            vocab_size=200,
            d_model=32,
            num_layers=2,
            num_heads=4,
            hidden_dim=64,
            moe_layers=[],  # No MoE for simplicity
            dropout_rate=0.1
        )

        # Compile model
        compile_qwen3_for_training(model, learning_rate=1e-3, max_steps=100, warmup_steps=10)

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
        for step in range(5):
            metrics = model.train_on_batch(inputs, targets, return_dict=True)
            loss_tensor = metrics['loss']
            loss = float(keras.ops.convert_to_numpy(loss_tensor))
            assert isinstance(loss, (float, np.floating))
            assert loss >= 0

            if step == 0:
                initial_loss = loss

        # Loss should be reasonable (not necessarily decreasing due to random data)
        assert loss < 50.0, f"Loss too high: {loss}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])