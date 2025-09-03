"""
Tests for Qwen3 Model Implementation

This module provides comprehensive tests for the Qwen3 model, including:
- Basic functionality tests
- Configuration variants
- Serialization/deserialization
- Factory functions
- Error handling
- Performance validation

All tests use small model configurations to keep test runtime reasonable.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any

import keras

from dl_techniques.models.qwen3.model import (
    Qwen3Model,
    create_qwen3_model,
    create_qwen3_coder_30b_config
)


class TestQwen3ModelBasic:
    """Test basic Qwen3 model functionality."""

    @pytest.fixture
    def tiny_config(self) -> Dict[str, Any]:
        """Create a tiny model configuration for fast testing."""
        return {
            'vocab_size': 1000,
            'd_model': 128,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'head_dim': 32,
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
            'num_layers': 3,
            'num_heads': 6,
            'num_kv_groups': 2,
            'head_dim': 16,
            'hidden_dim': 192,
            'moe_layers': [1],  # Second layer uses MoE
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
        return keras.ops.ones((2, 64), dtype='int32')

    def test_model_creation(self, tiny_config: Dict[str, Any]):
        """Test basic model creation and initialization."""
        model = Qwen3Model(**tiny_config)

        # Check model attributes
        assert model.vocab_size == tiny_config['vocab_size']
        assert model.d_model == tiny_config['d_model']
        assert model.num_layers == tiny_config['num_layers']
        assert len(model.transformer_blocks) == tiny_config['num_layers']

        # Check embedding layer
        assert model.token_embedding.input_dim == tiny_config['vocab_size']
        assert model.token_embedding.output_dim == tiny_config['d_model']

        # Check normalization layer
        assert model.final_norm is not None

        # Check weight tying
        if tiny_config['use_weight_tying']:
            assert model.output_head is None
        else:
            assert model.output_head is not None

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

    def test_model_with_moe(self, mini_config_with_moe: Dict[str, Any]):
        """Test model with Mixture of Experts layers."""
        model = Qwen3Model(**mini_config_with_moe)

        # Check MoE configuration
        assert model.moe_layers == [1]
        assert hasattr(model, 'moe_layer_1')

        # Test forward pass
        sample_input = keras.ops.ones((2, 32), dtype='int32')
        output = model(sample_input)

        expected_shape = (2, 32, mini_config_with_moe['vocab_size'])
        assert output.shape == expected_shape

    def test_training_mode(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test model behavior in training vs inference mode."""
        tiny_config['dropout_rate'] = 0.3  # Set higher dropout for testing
        model = Qwen3Model(**tiny_config)

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
        if tiny_config['dropout_rate'] > 0:
            # Allow for some similarity but expect some difference
            relative_diff = np.abs(output_train_np - output_inference_np) / (np.abs(output_inference_np) + 1e-8)
            assert np.mean(relative_diff) > 1e-6, "Training and inference outputs are too similar with dropout"

    def test_weight_tying_vs_separate_head(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test weight tying vs separate output head."""
        # Test with weight tying
        config_tied = tiny_config.copy()
        config_tied['use_weight_tying'] = True
        model_tied = Qwen3Model(**config_tied)

        # Test with separate head
        config_separate = tiny_config.copy()
        config_separate['use_weight_tying'] = False
        model_separate = Qwen3Model(**config_separate)

        # Forward passes
        output_tied = model_tied(sample_input)
        output_separate = model_separate(sample_input)

        # Both should produce valid outputs
        assert output_tied.shape == output_separate.shape

        # Check parameter counts are different
        tied_params = sum(np.prod(w.shape) for w in model_tied.trainable_variables)
        separate_params = sum(np.prod(w.shape) for w in model_separate.trainable_variables)

        # Separate head should have more parameters
        vocab_size = tiny_config['vocab_size']
        d_model = tiny_config['d_model']
        expected_diff = vocab_size * d_model  # Output head parameters

        assert separate_params > tied_params
        assert abs((separate_params - tied_params) - expected_diff) < 1000  # Allow some tolerance


class TestQwen3ModelConfigurations:
    """Test various model configurations."""

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
                'head_dim': config['d_model'] // config['num_heads'],
                'hidden_dim': config['d_model'] * 2,
            }

            model = Qwen3Model(**model_config)
            sample_input = keras.ops.ones((1, 16), dtype='int32')
            output = model(sample_input)

            assert output.shape == (1, 16, model_config['vocab_size'])

    def test_different_moe_patterns(self):
        """Test different MoE layer patterns."""
        base_config = {
            'vocab_size': 600,
            'd_model': 96,
            'num_layers': 4,
            'num_heads': 6,
            'num_kv_groups': 2,
            'head_dim': 16,
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
            sample_input = keras.ops.ones((1, 16), dtype='int32')
            output = model(sample_input)

            assert output.shape == (1, 16, config['vocab_size'])

    def test_rope_theta_variations(self):
        """Test different RoPE theta values."""
        base_config = {
            'vocab_size': 400,
            'd_model': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'head_dim': 16,
            'hidden_dim': 128,
        }

        rope_thetas = [10000.0, 100000.0, 1000000.0, 10000000.0]

        for theta in rope_thetas:
            config = base_config.copy()
            config['rope_theta'] = theta

            model = Qwen3Model(**config)
            sample_input = keras.ops.ones((1, 32), dtype='int32')
            output = model(sample_input)

            assert output.shape == (1, 32, config['vocab_size'])


class TestQwen3ModelSerialization:
    """Test model serialization and deserialization."""

    @pytest.fixture
    def serializable_config(self) -> Dict[str, Any]:
        """Create a config for serialization testing."""
        return {
            'vocab_size': 500,
            'd_model': 64,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'head_dim': 16,
            'hidden_dim': 128,
            'moe_layers': [1],
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 64,
        }

    def test_get_config_completeness(self, serializable_config: Dict[str, Any]):
        """Test that get_config returns all necessary parameters."""
        model = Qwen3Model(**serializable_config)
        config = model.get_config()

        # Check all important parameters are present
        expected_keys = {
            'vocab_size', 'd_model', 'num_layers', 'num_heads',
            'num_kv_groups', 'head_dim', 'hidden_dim', 'moe_layers',
            'num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
            'context_length', 'rope_theta', 'use_weight_tying', 'dropout_rate'
        }

        for key in expected_keys:
            assert key in config, f"Missing key in config: {key}"
            assert config[key] == getattr(model, key), f"Config mismatch for {key}"

    def test_serialization_cycle(self, serializable_config: Dict[str, Any]):
        """Test full serialization and deserialization cycle."""
        # Create original model
        original_model = Qwen3Model(**serializable_config)
        sample_input = keras.ops.ones((2, 24), dtype='int32')
        original_output = original_model(sample_input)

        # Save and load model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'qwen3_test.keras')
            original_model.save(model_path)
            loaded_model = keras.models.load_model(model_path)

            # Test loaded model
            loaded_output = loaded_model(sample_input)

            # Outputs should be identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-6, atol=1e-6,
                err_msg="Original and loaded model outputs should match"
            )

    def test_config_reconstruction(self, serializable_config: Dict[str, Any]):
        """Test model can be reconstructed from config."""
        original_model = Qwen3Model(**serializable_config)
        config = original_model.get_config()

        # Remove base model keys that aren't constructor arguments
        model_keys = {'name', 'trainable'}
        for key in model_keys:
            config.pop(key, None)

        # Create new model from config
        reconstructed_model = Qwen3Model(**config)

        # Test same architecture
        assert reconstructed_model.vocab_size == original_model.vocab_size
        assert reconstructed_model.d_model == original_model.d_model
        assert reconstructed_model.num_layers == original_model.num_layers
        assert len(reconstructed_model.transformer_blocks) == len(original_model.transformer_blocks)


class TestQwen3ModelFactories:
    """Test factory functions."""

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

    def test_create_qwen3_model_default(self):
        """Test factory function with default config."""
        # This would create a huge model, so we override with small config
        small_config = {
            'vocab_size': 1000,
            'd_model': 128,
            'num_layers': 2,
            'num_heads': 4,
            'num_kv_groups': 2,
            'head_dim': 32,
            'hidden_dim': 256,
        }

        model = create_qwen3_model(**small_config)

        assert isinstance(model, Qwen3Model)
        assert model.vocab_size == small_config['vocab_size']
        assert model.d_model == small_config['d_model']

    def test_create_qwen3_model_with_config_override(self):
        """Test factory function with config and overrides."""
        base_config = {
            'vocab_size': 800,
            'd_model': 96,
            'num_layers': 3,
            'num_heads': 6,
            'num_kv_groups': 2,
            'head_dim': 16,
            'hidden_dim': 192,
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
                head_dim=32,
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
                head_dim=32,
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
                head_dim=32,
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
                head_dim=20,  # d_model / num_heads = 120 / 6 = 20
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
                head_dim=32,
                hidden_dim=256
            )


class TestQwen3ModelPerformance:
    """Test model performance characteristics."""

    def test_memory_efficiency_with_weight_tying(self):
        """Test weight tying reduces parameter count."""
        base_config = {
            'vocab_size': 2000,
            'd_model': 128,
            'num_layers': 3,
            'num_heads': 8,
            'num_kv_groups': 4,
            'head_dim': 16,
            'hidden_dim': 256,
        }

        # Model with weight tying
        config_tied = {**base_config, 'use_weight_tying': True}
        model_tied = Qwen3Model(**config_tied)

        # Model without weight tying
        config_separate = {**base_config, 'use_weight_tying': False}
        model_separate = Qwen3Model(**config_separate)

        # Count parameters
        tied_params = sum(np.prod(w.shape) for w in model_tied.trainable_variables)
        separate_params = sum(np.prod(w.shape) for w in model_separate.trainable_variables)

        # Weight tying should reduce parameters
        assert tied_params < separate_params

        # Difference should be approximately vocab_size * d_model
        expected_diff = base_config['vocab_size'] * base_config['d_model']
        actual_diff = separate_params - tied_params

        # Allow for some variance due to other parameters
        assert abs(actual_diff - expected_diff) < expected_diff * 0.1

    def test_moe_parameter_scaling(self):
        """Test MoE increases parameters appropriately."""
        base_config = {
            'vocab_size': 1000,
            'd_model': 96,
            'num_layers': 3,
            'num_heads': 6,
            'num_kv_groups': 2,
            'head_dim': 16,
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

        # Count parameters
        no_moe_params = sum(np.prod(w.shape) for w in model_no_moe.trainable_variables)
        with_moe_params = sum(np.prod(w.shape) for w in model_with_moe.trainable_variables)

        # MoE should significantly increase parameter count
        assert with_moe_params > no_moe_params

        # But should still be reasonable for testing
        assert with_moe_params < 3_000_000  # 3M parameter limit


if __name__ == "__main__":
    pytest.main([__file__])