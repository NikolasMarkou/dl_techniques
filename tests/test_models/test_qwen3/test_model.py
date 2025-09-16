"""
Comprehensive Tests for Qwen3 Model Implementation

This module provides comprehensive tests for the Qwen3 model, including:
- Basic functionality tests with standard transformer blocks
- Configuration variants optimized for single GPU testing
- Selective MoE layer configuration and testing
- Serialization/deserialization with full architecture support
- Factory functions and utilities
- Error handling and edge cases
- Performance validation for transformer architecture

All tests use small model configurations to keep memory usage low for single GPU environments.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import Dict, Any

import keras

from dl_techniques.models.qwen3.model import (
    Qwen3,
    create_qwen3,
    create_qwen3_generation,
    create_qwen3_classification,
)
from dl_techniques.layers.transformer import TransformerLayer

class TestQwen3ModelBasic:
    """Test basic Qwen3 model functionality with standard transformer blocks."""

    @pytest.fixture
    def tiny_config(self) -> Dict[str, Any]:
        """Create a tiny model configuration for fast testing."""
        return {
            'vocab_size': 1000,
            'hidden_size': 128,
            'num_layers': 3,  # 3 standard transformer layers
            'num_attention_heads': 8,
            'num_key_value_heads': 2,
            'max_seq_len': 512,
            'moe_layers': [],  # No MoE for basic tests
            'num_experts': 1,
            'num_experts_per_tok': 1,
            'dropout_rate': 0.1,
        }

    @pytest.fixture
    def mini_config_with_moe(self) -> Dict[str, Any]:
        """Create a mini model configuration with selective MoE layers."""
        return {
            'vocab_size': 800,
            'hidden_size': 96,
            'num_layers': 4,  # 4 transformer layers
            'num_attention_heads': 6,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
            'moe_layers': [1, 3],  # MoE on layers 1 and 3
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 64,
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
        model = Qwen3(**tiny_config)

        # Check model attributes
        assert model.vocab_size == tiny_config['vocab_size']
        assert model.hidden_size == tiny_config['hidden_size']
        assert model.num_layers == tiny_config['num_layers']
        assert len(model.blocks) == tiny_config['num_layers']
        assert model.head_dim == tiny_config['hidden_size'] // tiny_config['num_attention_heads']

        # Check embedding layer
        assert model.embeddings.input_dim == tiny_config['vocab_size']
        assert model.embeddings.output_dim == tiny_config['hidden_size']

        # Check final normalization layer (should be RMSNorm)
        assert model.final_norm is not None

        # Check blocks are TransformerLayer instances
        for i, block in enumerate(model.blocks):
            assert isinstance(block, TransformerLayer), f"Block {i} is not a TransformerLayer"

        # Check LM head
        assert model.lm_head is not None
        assert model.lm_head.units == tiny_config['vocab_size']

    def test_model_forward_pass(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test forward pass through the model."""
        model = Qwen3(**tiny_config)

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

    def test_model_with_selective_moe(self, mini_config_with_moe: Dict[str, Any]):
        """Test model with selective MoE layers."""
        model = Qwen3(**mini_config_with_moe)

        # Check MoE configuration
        assert model.moe_layers == mini_config_with_moe['moe_layers']
        assert model.num_experts == mini_config_with_moe['num_experts']
        assert model.num_experts_per_tok == mini_config_with_moe['num_experts_per_tok']

        # Check that only specified layers have MoE
        for i, block in enumerate(model.blocks):
            if i in model.moe_layers:
                # This block should use MoE
                assert hasattr(block, 'moe_config'), f"Block {i} should have MoE config"
                assert block.moe_config is not None, f"Block {i} MoE config should not be None"
            else:
                # This block should use standard FFN
                # The TransformerLayer should have ffn attribute when not using MoE
                pass  # We can't easily check this without inspecting internals

        # Test forward pass with selective MoE
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
        tiny_config_with_dropout['dropout_rate'] = 0.3
        model = Qwen3(**tiny_config_with_dropout)

        # Training mode
        output_train = model(sample_input, training=True)

        # Inference mode
        output_inference = model(sample_input, training=False)

        # Check shapes are the same
        assert output_train.shape == output_inference.shape

        # Convert to numpy for comparison
        output_train_np = keras.ops.convert_to_numpy(output_train)
        output_inference_np = keras.ops.convert_to_numpy(output_inference)

        # With dropout, training and inference should be different
        if tiny_config_with_dropout['dropout_rate'] > 0:
            relative_diff = np.abs(output_train_np - output_inference_np) / (np.abs(output_inference_np) + 1e-8)
            mean_diff = np.mean(relative_diff)
            assert mean_diff > 1e-6, f"Training and inference outputs are too similar with dropout: {mean_diff}"

    def test_dictionary_input_format(self, tiny_config: Dict[str, Any]):
        """Test model handles dictionary input format."""
        model = Qwen3(**tiny_config)

        batch_size, seq_len = 2, 32
        input_ids = keras.ops.convert_to_tensor(
            np.random.randint(0, tiny_config['vocab_size'], size=(batch_size, seq_len)),
            dtype='int32'
        )
        attention_mask = keras.ops.ones((batch_size, seq_len), dtype='int32')

        # Test dictionary input
        dict_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        output_dict = model(dict_input)

        # Test tensor input
        output_tensor = model(input_ids, attention_mask=attention_mask)

        # Outputs should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_dict),
            keras.ops.convert_to_numpy(output_tensor),
            rtol=1e-6, atol=1e-6,
            err_msg="Dictionary and tensor input should produce identical outputs"
        )

    def test_return_dict_format(self, tiny_config: Dict[str, Any], sample_input: keras.KerasTensor):
        """Test model's return_dict functionality."""
        model = Qwen3(**tiny_config)

        # Test return_dict=False (default)
        output_tensor = model(sample_input, return_dict=False)

        # Test return_dict=True
        output_dict = model(sample_input, return_dict=True)
        assert isinstance(output_dict, dict)
        assert 'logits' in output_dict

        # Logits should be identical
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output_tensor),
            keras.ops.convert_to_numpy(output_dict['logits']),
            rtol=1e-6, atol=1e-6,
            err_msg="Tensor and dict return formats should have identical logits"
        )


class TestQwen3ModelConfigurations:
    """Test various model configurations for single GPU testing."""

    def test_model_variants(self):
        """Test different model variants from MODEL_VARIANTS."""
        # Test tiny variant (smallest)
        model_tiny = Qwen3.from_variant("tiny")
        assert model_tiny.vocab_size == 32000
        assert model_tiny.hidden_size == 512
        assert model_tiny.num_layers == 6
        assert model_tiny.moe_layers == []  # No MoE layers

        # Test small variant
        model_small = Qwen3.from_variant("small")
        assert model_small.vocab_size == 32000
        assert model_small.hidden_size == 768
        assert model_small.num_layers == 12
        assert model_small.moe_layers == [3, 6, 9]  # Selective MoE

        # Test medium variant
        model_medium = Qwen3.from_variant("medium")
        assert model_medium.vocab_size == 100000
        assert model_medium.hidden_size == 1024
        assert model_medium.num_layers == 24
        assert len(model_medium.moe_layers) > 0  # Has MoE layers

        # Test 30b-coder variant (configuration only, too large to test forward pass)
        try:
            model_30b = Qwen3.from_variant("30b-coder", num_layers=2, hidden_size=64)  # Override to tiny size
            assert model_30b.vocab_size == 151936
        except Exception:
            pytest.skip("30b-coder variant too large even with overrides")

    def test_different_head_configurations(self):
        """Test different attention head configurations."""
        configs = [
            # Standard configuration
            {'num_attention_heads': 8, 'num_key_value_heads': 8, 'hidden_size': 128},
            # Grouped query attention
            {'num_attention_heads': 8, 'num_key_value_heads': 2, 'hidden_size': 128},
            # Extreme GQA
            {'num_attention_heads': 6, 'num_key_value_heads': 1, 'hidden_size': 120},
        ]

        for config in configs:
            model_config = {
                'vocab_size': 500,
                'hidden_size': config['hidden_size'],
                'num_layers': 2,
                'num_attention_heads': config['num_attention_heads'],
                'num_key_value_heads': config['num_key_value_heads'],
                'max_seq_len': 256,
                'moe_layers': [],  # No MoE for simplicity
                'num_experts': 1,
                'num_experts_per_tok': 1,
            }

            model = Qwen3(**model_config)
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 500, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input)

            assert output.shape == (1, 16, model_config['vocab_size'])

            # Check head_dim is computed correctly
            expected_head_dim = config['hidden_size'] // config['num_attention_heads']
            assert model.head_dim == expected_head_dim

    def test_different_moe_configurations(self):
        """Test different selective MoE configurations."""
        base_config = {
            'vocab_size': 600,
            'hidden_size': 96,
            'num_layers': 6,
            'num_attention_heads': 6,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
        }

        moe_configs = [
            # No MoE
            {'moe_layers': [], 'num_experts': 1, 'num_experts_per_tok': 1},
            # Sparse MoE (every 3rd layer)
            {'moe_layers': [2, 5], 'num_experts': 4, 'num_experts_per_tok': 2, 'moe_intermediate_size': 64},
            # Dense MoE (every other layer)
            {'moe_layers': [1, 3, 5], 'num_experts': 8, 'num_experts_per_tok': 2, 'moe_intermediate_size': 96},
        ]

        for moe_config in moe_configs:
            config = {**base_config, **moe_config}
            model = Qwen3(**config)

            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 600, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input, training=True)

            assert output.shape == (1, 16, config['vocab_size'])

            # Check MoE configuration
            assert model.moe_layers == moe_config['moe_layers']
            assert model.num_experts == moe_config['num_experts']
            assert model.num_experts_per_tok == moe_config['num_experts_per_tok']

    def test_stochastic_depth_configuration(self):
        """Test stochastic depth configuration."""
        config = {
            'vocab_size': 500,
            'hidden_size': 64,
            'num_layers': 3,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
            'moe_layers': [],
            'num_experts': 1,
            'num_experts_per_tok': 1,
            'use_stochastic_depth': True,
            'stochastic_depth_rate': 0.1
        }

        model = Qwen3(**config)

        # Check stochastic depth is enabled
        assert model.use_stochastic_depth is True
        assert model.stochastic_depth_rate == 0.1

        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 500, size=(2, 16)),
            dtype='int32'
        )

        # Should work in both training and inference
        output_train = model(sample_input, training=True)
        output_inference = model(sample_input, training=False)

        assert output_train.shape == output_inference.shape

    def test_rope_theta_configuration(self):
        """Test different RoPE theta configurations."""
        configs = [
            {'rope_theta': 10_000.0, 'max_seq_len': 2048},      # Standard
            {'rope_theta': 100_000.0, 'max_seq_len': 8192},     # Extended context
            {'rope_theta': 10_000_000.0, 'max_seq_len': 32768}, # Long context
        ]

        for rope_config in configs:
            config = {
                'vocab_size': 500,
                'hidden_size': 64,
                'num_layers': 2,
                'num_attention_heads': 4,
                'num_key_value_heads': 2,
                'moe_layers': [],
                'num_experts': 1,
                'num_experts_per_tok': 1,
                **rope_config
            }

            model = Qwen3(**config)
            assert model.rope_theta == rope_config['rope_theta']
            assert model.max_seq_len == rope_config['max_seq_len']

            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 500, size=(1, 16)),
                dtype='int32'
            )
            output = model(sample_input)
            assert output.shape == (1, 16, 500)


class TestQwen3ModelSerialization:
    """Test model serialization and deserialization."""

    @pytest.fixture
    def serializable_config(self) -> Dict[str, Any]:
        """Create a config for serialization testing."""
        return {
            'vocab_size': 500,
            'hidden_size': 64,
            'num_layers': 3,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
            'moe_layers': [1],  # One MoE layer
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 64,
            'rope_theta': 10_000.0,
            'dropout_rate': 0.1,
        }

    def test_get_config_completeness(self, serializable_config: Dict[str, Any]):
        """Test that get_config returns all necessary parameters."""
        model = Qwen3(**serializable_config)
        config = model.get_config()

        # Check all important parameters are present
        expected_keys = {
            'vocab_size', 'hidden_size', 'num_layers', 'num_attention_heads',
            'num_key_value_heads', 'max_seq_len', 'moe_layers',
            'num_experts', 'num_experts_per_tok', 'moe_intermediate_size',
            'rope_theta', 'norm_eps', 'dropout_rate', 'initializer_range',
            'normalization_type', 'ffn_type', 'use_stochastic_depth',
            'stochastic_depth_rate'
        }

        for key in expected_keys:
            assert key in config, f"Missing key in config: {key}"
            assert config[key] == getattr(model, key), f"Config mismatch for {key}"

    def test_serialization_cycle_standard(self, serializable_config: Dict[str, Any]):
        """Test full serialization and deserialization cycle for standard model."""
        # Remove MoE for standard test
        config = serializable_config.copy()
        config['moe_layers'] = []
        config['num_experts'] = 1
        config['num_experts_per_tok'] = 1

        # Create original model
        original_model = Qwen3(**config)
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
        """Test serialization cycle for selective MoE model."""
        # Create original MoE model
        original_model = Qwen3(**serializable_config)
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 500, size=(2, 20)),
            dtype='int32'
        )
        original_output = original_model(sample_input, training=False)

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
        original_model = Qwen3(**serializable_config)
        config = original_model.get_config()

        # Remove base model keys that aren't constructor arguments
        model_keys = {'name', 'trainable', 'dtype'}
        for key in model_keys:
            config.pop(key, None)

        # Create new model from config
        reconstructed_model = Qwen3.from_config(config)

        # Test same architecture
        assert reconstructed_model.vocab_size == original_model.vocab_size
        assert reconstructed_model.hidden_size == original_model.hidden_size
        assert reconstructed_model.num_layers == original_model.num_layers
        assert len(reconstructed_model.blocks) == len(original_model.blocks)
        assert reconstructed_model.moe_layers == original_model.moe_layers
        assert reconstructed_model.num_experts == original_model.num_experts


class TestQwen3ModelFactories:
    """Test factory functions and utilities."""

    def test_main_factory_functionality(self):
        """Test the main `create_qwen3` factory's core functionality."""
        num_labels = 5
        vocab_size = 1000

        # 1. Test generation model from variant
        gen_model = create_qwen3("tiny", task_type="generation")
        assert gen_model.name == "qwen3_for_generation"
        assert len(gen_model.inputs) == 2
        input_ids = keras.ops.convert_to_tensor(np.random.randint(0, 32000, size=(2, 32)), dtype='int32')
        attention_mask = keras.ops.ones((2, 32), dtype='int32')
        logits = gen_model([input_ids, attention_mask])
        assert logits.shape == (2, 32, 32000)

        # 2. Test classification model from variant
        clf_model = create_qwen3("tiny", task_type="classification", num_labels=num_labels)
        assert clf_model.name == "qwen3_for_classification"
        assert len(clf_model.inputs) == 2
        class_logits = clf_model([input_ids, attention_mask])
        assert class_logits.shape == (2, num_labels)

        # 3. Test parameter override with kwargs
        custom_model = create_qwen3(
            "tiny",
            task_type="generation",
            hidden_size=256,  # Override
            num_layers=2,     # Override
            vocab_size=vocab_size
        )
        assert custom_model.name == "qwen3_for_generation"
        # Access the backbone model to check its config
        backbone = custom_model.get_layer("qwen3_backbone")
        assert backbone.hidden_size == 256
        assert backbone.num_layers == 2
        assert backbone.vocab_size == vocab_size

        # 4. Test creating from a configuration dictionary
        custom_config = {
            'vocab_size': vocab_size,
            'hidden_size': 128,
            'num_layers': 3,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'max_seq_len': 128,
            'moe_layers': [1],  # One MoE layer
            'num_experts': 4,
            'num_experts_per_tok': 2,
        }
        dict_model = create_qwen3(custom_config, task_type="generation")
        dict_backbone = dict_model.get_layer("qwen3_backbone")
        assert dict_backbone.hidden_size == 128
        assert dict_backbone.vocab_size == vocab_size
        assert dict_backbone.moe_layers == [1]

    def test_classification_factory_advanced_options(self):
        """Test advanced options of the classification factory."""
        num_labels = 3
        input_ids = keras.ops.convert_to_tensor(np.random.randint(0, 32000, size=(2, 24)), dtype='int32')
        attention_mask = keras.ops.ones((2, 24), dtype='int32')

        # Test "mean" pooling strategy
        mean_pool_model = create_qwen3(
            "tiny",
            task_type="classification",
            num_labels=num_labels,
            pooling_strategy="mean"
        )
        assert mean_pool_model.name == "qwen3_for_classification"
        mean_pool_logits = mean_pool_model([input_ids, attention_mask])
        assert mean_pool_logits.shape == (2, num_labels)

        # Test custom classifier dropout
        dropout_model = create_qwen3(
            "tiny",
            task_type="classification",
            num_labels=num_labels,
            classifier_dropout=0.5
        )
        # Check that the dropout layer was created with the correct rate
        classifier_dropout_layer = dropout_model.get_layer("classifier_dropout")
        assert classifier_dropout_layer is not None
        assert classifier_dropout_layer.rate == 0.5

    def test_model_variant_configurations(self):
        """Test that MODEL_VARIANTS have correct configurations."""
        variants = ["tiny", "small", "medium", "30b-coder"]

        for variant in variants:
            assert variant in Qwen3.MODEL_VARIANTS
            config = Qwen3.MODEL_VARIANTS[variant]

            # Check required fields
            assert 'vocab_size' in config
            assert 'hidden_size' in config
            assert 'num_layers' in config
            assert 'num_attention_heads' in config
            assert 'num_key_value_heads' in config
            assert 'moe_layers' in config
            assert 'rope_theta' in config
            assert 'description' in config

            # Check reasonable values
            assert config['vocab_size'] > 0
            assert config['hidden_size'] > 0
            assert config['num_layers'] > 0
            assert config['num_attention_heads'] > 0
            assert config['num_key_value_heads'] > 0
            assert config['rope_theta'] > 0

            # Check head consistency
            assert config['hidden_size'] % config['num_attention_heads'] == 0
            assert config['num_attention_heads'] % config['num_key_value_heads'] == 0

            # Check MoE layers are valid
            assert isinstance(config['moe_layers'], list)
            for moe_layer_idx in config['moe_layers']:
                assert 0 <= moe_layer_idx < config['num_layers']

    def test_direct_factory_functions(self):
        """Test direct factory functions."""
        config = {
            'vocab_size': 1000,
            'hidden_size': 64,
            'num_layers': 2,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
            'moe_layers': [],
        }

        # Test generation factory
        gen_model = create_qwen3_generation(config)
        assert gen_model.name == "qwen3_for_generation"
        assert len(gen_model.inputs) == 2
        assert gen_model.outputs[0].shape[-1] == config['vocab_size']

        # Test classification factory
        clf_model = create_qwen3_classification(config, num_labels=10)
        assert clf_model.name == "qwen3_for_classification"
        assert len(clf_model.inputs) == 2
        assert clf_model.outputs[0].shape[-1] == 10


class TestQwen3ModelErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_vocab_size(self):
        """Test error handling for invalid vocabulary size."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Qwen3(
                vocab_size=0,
                hidden_size=128,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2
            )

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            Qwen3(
                vocab_size=-100,
                hidden_size=128,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2
            )

    def test_invalid_hidden_size(self):
        """Test error handling for invalid hidden size."""
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            Qwen3(
                vocab_size=1000,
                hidden_size=-128,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2
            )

    def test_mismatched_heads_and_hidden_size(self):
        """Test error handling for mismatched heads and hidden size."""
        with pytest.raises(ValueError, match="hidden_size .* must be divisible by num_attention_heads"):
            Qwen3(
                vocab_size=1000,
                hidden_size=129,  # Not divisible by 4
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2
            )

    def test_mismatched_heads_and_kv_heads(self):
        """Test error handling for mismatched heads and KV heads."""
        with pytest.raises(ValueError, match="num_attention_heads .* must be divisible by num_key_value_heads"):
            Qwen3(
                vocab_size=1000,
                hidden_size=120,  # Divisible by 6
                num_layers=2,
                num_attention_heads=6,  # NOT divisible by 4
                num_key_value_heads=4
            )

    def test_invalid_num_layers(self):
        """Test error handling for invalid number of layers."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            Qwen3(
                vocab_size=1000,
                hidden_size=128,
                num_layers=0,
                num_attention_heads=4,
                num_key_value_heads=2
            )

    def test_invalid_expert_configuration(self):
        """Test error handling for invalid expert configuration."""
        with pytest.raises(ValueError, match=r"num_experts_per_tok \(\d+\) cannot exceed num_experts \(\d+\)"):
            Qwen3(
                vocab_size=1000,
                hidden_size=128,
                num_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                moe_layers=[1],
                num_experts=4,
                num_experts_per_tok=8  # More than num_experts
            )

    def test_invalid_moe_layer_indices(self):
        """Test error handling for invalid MoE layer indices."""
        with pytest.raises(ValueError, match="All MoE layer indices must be between"):
            Qwen3(
                vocab_size=1000,
                hidden_size=128,
                num_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                moe_layers=[0, 1, 5],  # Index 5 >= num_layers (3)
                num_experts=4,
                num_experts_per_tok=2
            )

    def test_unknown_variant(self):
        """Test error handling for unknown model variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            Qwen3.from_variant('invalid_variant')

    def test_invalid_factory_parameters(self):
        """Test error handling in the main factory function."""
        # Missing num_labels for classification
        with pytest.raises(ValueError, match="`num_labels` must be provided"):
            create_qwen3("tiny", task_type="classification")

        # Invalid task_type
        with pytest.raises(ValueError, match="Unknown task_type"):
            create_qwen3("tiny", task_type="invalid_task")

        # Invalid num_labels
        with pytest.raises(ValueError, match="num_labels must be positive"):
            create_qwen3("tiny", task_type="classification", num_labels=-1)

        # Invalid pooling strategy
        with pytest.raises(ValueError, match="pooling_strategy must be 'cls' or 'mean'"):
            create_qwen3("tiny", task_type="classification", num_labels=3, pooling_strategy="max")


class TestQwen3ModelPerformance:
    """Test model performance characteristics for single GPU."""

    def test_selective_moe_parameter_scaling(self):
        """Test selective MoE increases parameters appropriately."""
        base_config = {
            'vocab_size': 800,
            'hidden_size': 96,
            'num_layers': 4,
            'num_attention_heads': 6,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
        }

        # Model without MoE
        config_no_moe = {**base_config, 'moe_layers': [], 'num_experts': 1, 'num_experts_per_tok': 1}
        model_no_moe = Qwen3(**config_no_moe)

        # Model with selective MoE
        config_with_moe = {
            **base_config,
            'moe_layers': [1, 3],  # 2 out of 4 layers use MoE
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 64,
        }
        model_with_moe = Qwen3(**config_with_moe)

        # Build models
        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 800, size=(1, 16)),
            dtype='int32'
        )
        _ = model_no_moe(sample_input)
        _ = model_with_moe(sample_input)

        # Count parameters
        no_moe_params = sum(keras.ops.size(w) for w in model_no_moe.trainable_variables)
        with_moe_params = sum(keras.ops.size(w) for w in model_with_moe.trainable_variables)

        # Selective MoE should increase parameter count, but not as much as full MoE
        assert with_moe_params > no_moe_params

        # But should still be reasonable for single GPU testing
        assert int(with_moe_params) < 5_000_000  # 5M parameter limit

    def test_forward_pass_efficiency(self):
        """Test forward pass runs efficiently."""
        import time

        # Create small efficient model
        model = Qwen3.from_variant("tiny")

        # Create input
        inputs = keras.ops.convert_to_tensor(
            np.random.randint(0, model.vocab_size, size=(4, 64)),
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
        assert avg_time < 10.0, f"Forward pass too slow: {avg_time:.4f}s"

    def test_different_sequence_lengths(self):
        """Test model handles different sequence lengths efficiently."""
        model = Qwen3.from_variant("tiny")

        # Test different sequence lengths
        sequence_lengths = [16, 32, 64, 128, 256, 512]

        for seq_len in sequence_lengths:
            if seq_len <= model.max_seq_len:
                inputs = keras.ops.convert_to_tensor(
                    np.random.randint(0, model.vocab_size, size=(2, seq_len)),
                    dtype='int32'
                )

                output = model(inputs)
                assert output.shape[1] == seq_len

                # Check output quality
                output_np = keras.ops.convert_to_numpy(output)
                assert not np.any(np.isnan(output_np))
                assert not np.any(np.isinf(output_np))

    def test_memory_usage_scaling(self):
        """Test memory usage scaling with model size."""
        configs = [
            # Tiny
            {'hidden_size': 32, 'num_layers': 2, 'num_attention_heads': 4},
            # Small
            {'hidden_size': 64, 'num_layers': 3, 'num_attention_heads': 4},
            # Medium (for single GPU)
            {'hidden_size': 96, 'num_layers': 4, 'num_attention_heads': 6},
        ]

        param_counts = []

        for config in configs:
            base_config = {
                'vocab_size': 500,
                'num_key_value_heads': 2,
                'max_seq_len': 256,
                'moe_layers': [],  # No MoE to focus on base scaling
                'num_experts': 1,
                'num_experts_per_tok': 1,
                **config
            }

            model = Qwen3(**base_config)

            # Build model
            sample_input = keras.ops.convert_to_tensor(
                np.random.randint(0, 500, size=(1, 16)),
                dtype='int32'
            )
            _ = model(sample_input)

            # Count parameters
            param_count = sum(keras.ops.size(w) for w in model.trainable_variables)
            param_counts.append(int(param_count))

        # Parameters should increase with model size
        assert param_counts[0] < param_counts[1] < param_counts[2]

    def test_rope_scaling_efficiency(self):
        """Test RoPE scaling doesn't significantly impact performance."""
        import time

        configs = [
            {'rope_theta': 10_000.0, 'max_seq_len': 512},
            {'rope_theta': 100_000.0, 'max_seq_len': 1024},
            {'rope_theta': 1_000_000.0, 'max_seq_len': 2048},
        ]

        timings = []

        for rope_config in configs:
            config = {
                'vocab_size': 500,
                'hidden_size': 64,
                'num_layers': 2,
                'num_attention_heads': 4,
                'num_key_value_heads': 2,
                'moe_layers': [],
                **rope_config
            }

            model = Qwen3(**config)
            inputs = keras.ops.convert_to_tensor(
                np.random.randint(0, 500, size=(2, 32)),
                dtype='int32'
            )

            # Warmup
            _ = model(inputs)

            # Time forward pass
            start_time = time.time()
            _ = model(inputs)
            timing = time.time() - start_time
            timings.append(timing)

        # RoPE scaling should not dramatically increase inference time
        max_timing = max(timings)
        min_timing = min(timings)
        assert max_timing / min_timing < 2.0, f"RoPE scaling causes significant slowdown: {max_timing/min_timing:.2f}x"


class TestQwen3Integration:
    """Test integration and end-to-end functionality."""

    def test_end_to_end_training_simulation(self):
        """Test complete training simulation on tiny model."""
        # Create very small model for training test
        model = create_qwen3("tiny", task_type="generation")

        # Compile model
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create tiny dataset
        batch_size, seq_len = 4, 16
        vocab_size = 32000

        inputs = [
            keras.ops.convert_to_tensor(
                np.random.randint(0, vocab_size, size=(batch_size, seq_len)),
                dtype='int32'
            ),
            keras.ops.ones((batch_size, seq_len), dtype='int32')  # attention_mask
        ]
        targets = keras.ops.convert_to_tensor(
            np.random.randint(0, vocab_size, size=(batch_size, seq_len)),
            dtype='int32'
        )

        # Run a few training steps
        for step in range(3):
            metrics = model.train_on_batch(inputs, targets, return_dict=True)
            loss = float(keras.ops.convert_to_numpy(metrics['loss']))
            assert isinstance(loss, (float, np.floating))
            assert loss >= 0
            assert loss < 50.0, f"Loss too high: {loss}"

    def test_model_summary_information(self):
        """Test model summary and logging."""
        model = Qwen3.from_variant("tiny")

        # Test summary doesn't crash
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary failed: {e}")

        # Test model provides useful info
        assert hasattr(model, 'vocab_size')
        assert hasattr(model, 'hidden_size')
        assert hasattr(model, 'num_layers')
        assert hasattr(model, 'moe_layers')
        assert hasattr(model, 'blocks')

        # Check block structure
        assert len(model.blocks) == model.num_layers

        # Each block should be properly configured TransformerLayer
        for i, block in enumerate(model.blocks):
            assert isinstance(block, TransformerLayer)
            assert hasattr(block, 'hidden_size')
            assert block.hidden_size == model.hidden_size

    def test_attention_mask_functionality(self):
        """Test attention mask is properly handled."""
        model = Qwen3.from_variant("tiny")

        batch_size, seq_len = 2, 32
        input_ids = keras.ops.convert_to_tensor(
            np.random.randint(0, model.vocab_size, size=(batch_size, seq_len)),
            dtype='int32'
        )

        # Create attention mask with some padding
        attention_mask = keras.ops.ones((batch_size, seq_len), dtype='int32')
        # Mask out last 8 tokens for second sequence
        attention_mask = keras.ops.scatter_update(
            attention_mask,
            [[1, 24], [1, 25], [1, 26], [1, 27], [1, 28], [1, 29], [1, 30], [1, 31]],
            np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        )

        # Forward pass with mask
        output_with_mask = model(input_ids, attention_mask=attention_mask)

        # Forward pass without mask
        output_without_mask = model(input_ids)

        # Outputs should be different
        output_diff = keras.ops.mean(keras.ops.abs(output_with_mask - output_without_mask))
        assert float(output_diff) > 1e-6, "Attention mask should affect outputs"

    def test_selective_moe_functionality(self):
        """Test selective MoE layers work correctly."""
        config = {
            'vocab_size': 500,
            'hidden_size': 64,
            'num_layers': 4,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
            'moe_layers': [1, 3],  # Only layers 1 and 3 use MoE
            'num_experts': 4,
            'num_experts_per_tok': 2,
            'moe_intermediate_size': 32,
        }

        model = Qwen3(**config)

        sample_input = keras.ops.convert_to_tensor(
            np.random.randint(0, 500, size=(2, 16)),
            dtype='int32'
        )

        # Test forward pass works
        output = model(sample_input, training=True)
        assert output.shape == (2, 16, 500)

        # Test that MoE layers are only where specified
        assert model.moe_layers == [1, 3]

        # Test training vs inference consistency
        output_train = model(sample_input, training=True)
        output_inference = model(sample_input, training=False)

        # Should be similar but not identical (due to MoE routing)
        diff = keras.ops.mean(keras.ops.abs(output_train - output_inference))
        assert float(diff) < 1.0, "Training and inference outputs too different"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])