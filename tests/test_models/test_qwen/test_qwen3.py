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
from typing import Any, Callable, Dict, Optional

import keras

from dl_techniques.models.language.qwen.qwen3 import Qwen3, create_qwen3
from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.heads.nlp.model_head import TASK_HEAD_NAME
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint

from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Our own advisory at `layers/transformers/transformer.py:620`: `moe_config`
# supersedes `ffn_type` / `ffn_args`. This module drives exactly that
# combination on purpose. Pinned live by
# `tests/test_the_deliberate_advisories_still_fire.py`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:moe_config is provided:UserWarning"),
]

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
        """The head layout must reach the attention projections.

        The logits shape is ``(1, 16, vocab_size)`` under every layout, and
        ``model.head_dim == hidden_size // num_attention_heads`` is a knob ECHO
        of a value the constructor computed and stored. The head layout is
        structural -- ``num_key_value_heads`` sets the K/V projection width --
        so the weight-shape signature is the discriminating fact.
        """
        configs = [
            # Standard configuration
            {'num_attention_heads': 8, 'num_key_value_heads': 8, 'hidden_size': 128},
            # Grouped query attention
            {'num_attention_heads': 8, 'num_key_value_heads': 2, 'hidden_size': 128},
            # Extreme GQA
            {'num_attention_heads': 6, 'num_key_value_heads': 1, 'hidden_size': 120},
        ]

        def _build(config):
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
            model(keras.ops.zeros((1, 16), dtype='int32'))
            return model

        builders = {i: (lambda c=c: _build(c)) for i, c in enumerate(configs)}
        sigs = assert_structural_knob_changes_weights(
            builders, knob="head configuration")
        # Stronger than "different": GQA (config 1) shares K/V across query
        # heads, so at the SAME hidden_size it must hold strictly FEWER
        # parameters than full multi-head attention (config 0).
        params = [sum(int(np.prod(w)) for w in sigs[i]) for i in (0, 1)]
        assert params[1] < params[0], (
            "num_key_value_heads=2 did not shrink the K/V projections relative "
            f"to num_key_value_heads=8: {params[1]} vs {params[0]} parameters"
        )

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
        """The MoE layout must reach the parameterisation.

        ``model.num_experts == ...`` and its two siblings are knob ECHOES of
        stored kwargs. The expert count is structural: each expert is its own
        set of FFN weights, so the signature and the parameter count carry the
        claim.
        """
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

        def _build(moe_config):
            model = Qwen3(**{**base_config, **moe_config})
            model(keras.ops.zeros((1, 16), dtype='int32'), training=False)
            return model

        builders = {i: (lambda m=m: _build(m)) for i, m in enumerate(moe_configs)}
        sigs = assert_structural_knob_changes_weights(
            builders, knob="MoE configuration")
        # Stronger than "different": more experts means strictly more
        # parameters. A dropped MoE config would leave all three identical.
        params = [sum(int(np.prod(w)) for w in sigs[i]) for i in range(len(moe_configs))]
        assert params == sorted(params) and params[0] < params[-1], (
            f"num_experts did not grow the parameter count: {params}"
        )

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
        """`create_qwen3` returns the class itself, headless or headed."""
        num_labels = 5
        vocab_size = 1000
        batch = {
            "input_ids": keras.ops.convert_to_tensor(
                np.random.randint(0, 32000, size=(2, 32)), dtype='int32'),
            "attention_mask": keras.ops.ones((2, 32), dtype='int32'),
        }

        # 1. A language model from a variant: the class, returning logits
        gen_model = create_qwen3("tiny")
        assert type(gen_model) is Qwen3
        assert gen_model.include_top is False
        assert gen_model(batch).shape == (2, 32, 32000)

        # 2. A sequence classifier from the same variant: the class, headed
        clf_model = create_qwen3(
            "tiny",
            include_top=True,
            task_config=NLPTaskConfig(
                name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=num_labels),
        )
        assert type(clf_model) is Qwen3
        assert clf_model.lm_head is None
        assert clf_model(batch).shape == (2, num_labels)

        # 3. Overrides of variant keys reach the model itself
        custom_model = create_qwen3(
            "tiny",
            hidden_size=256,  # Override
            num_layers=2,     # Override
            vocab_size=vocab_size
        )
        assert custom_model.hidden_size == 256
        assert custom_model.num_layers == 2
        assert custom_model.vocab_size == vocab_size

    def test_head_knobs_reach_the_task_head(self):
        """Pooling and dropout are the head's own knobs, set at construction."""
        num_labels = 3
        batch = {
            "input_ids": keras.ops.convert_to_tensor(
                np.random.randint(0, 32000, size=(2, 24)), dtype='int32'),
            "attention_mask": keras.ops.ones((2, 24), dtype='int32'),
        }

        # "mean" pooling through `head_config_overrides`
        mean_pool_model = create_qwen3(
            "tiny",
            include_top=True,
            task_config=_CLF,
            head_config_overrides={"pooling_type": "mean"},
        )
        assert mean_pool_model.task_head.pooling_type == "mean"
        assert mean_pool_model(batch).shape == (2, num_labels)

        # The head's dropout is the task config's `dropout_rate`
        dropout_model = create_qwen3(
            "tiny",
            include_top=True,
            task_config=NLPTaskConfig(
                name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=num_labels, dropout_rate=0.5),
        )
        assert dropout_model.task_head.dropout.rate == 0.5

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

    def test_the_class_takes_a_configuration_dictionary(self):
        """A custom configuration goes to the class: `Qwen3(**config)`."""
        config = {
            'vocab_size': 1000,
            'hidden_size': 64,
            'num_layers': 2,
            'num_attention_heads': 4,
            'num_key_value_heads': 2,
            'max_seq_len': 256,
            'moe_layers': [1],  # One MoE layer
            'num_experts': 4,
            'num_experts_per_tok': 2,
        }
        ids = keras.Input(shape=(None,), dtype="int32")

        model = Qwen3(**config)
        assert model.moe_layers == [1]
        assert model(ids).shape[-1] == config['vocab_size']

        clf_model = Qwen3(**config, include_top=True, task_config=NLPTaskConfig(
            name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION,
            num_classes=10))
        assert clf_model(ids).shape[-1] == 10


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
        # A head without a task
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            create_qwen3("tiny", include_top=True)

        batch = {"input_ids": np.ones((1, 8), dtype="int32")}
        tiny = dict(vocab_size=64, hidden_size=32, num_layers=1,
                    num_attention_heads=4, num_key_value_heads=2,
                    max_seq_len=16, moe_layers=[])

        # A negative class count cannot build its head
        model = create_qwen3("tiny", **tiny, include_top=True, task_config=NLPTaskConfig(
            name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=-1))
        with pytest.raises(ValueError, match="Negative dimensions"):
            model(batch)

        # An unknown pooling type is refused by the head
        model = create_qwen3("tiny", **tiny, include_top=True, task_config=_CLF,
                             head_config_overrides={"pooling_type": "bogus"})
        with pytest.raises(ValueError, match="Unknown pooling type"):
            model(batch)


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
        model = create_qwen3("tiny")

        # Compile model
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Create tiny dataset
        batch_size, seq_len = 4, 16
        vocab_size = 32000

        inputs = {
            "input_ids": keras.ops.convert_to_tensor(
                np.random.randint(0, vocab_size, size=(batch_size, seq_len)),
                dtype='int32'
            ),
            "attention_mask": keras.ops.ones((batch_size, seq_len), dtype='int32'),
        }
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

# ---------------------------------------------------------------------


class TestQwen3CausalMasking:
    """Qwen3 is a decoder-only causal LM and must not attend to the future.

    Modelled on the GPT-2 pair in tests/test_models/test_gpt2/test_gpt2.py:186
    and :639 -- the isolation test AND its negative control. The control is what
    makes the isolation attributable to the MASK rather than to some incidental
    property of the architecture.
    """

    CFG = {
        'vocab_size': 512,
        'hidden_size': 64,
        'num_layers': 2,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'max_seq_len': 64,
        'moe_layers': [],
        'num_experts': 1,
        'num_experts_per_tok': 1,
        'dropout_rate': 0.0,
    }

    @staticmethod
    def _pair():
        """Two sequences differing ONLY in the final position."""
        a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        b = np.array([[1, 2, 3, 4, 5, 6, 7, 99]], dtype=np.int32)
        return a, b

    def test_future_does_not_affect_past(self):
        """Changing the LAST token must not change any earlier logits.

        RED-proof: Qwen3.call forwarded only the padding mask (None unless a
        caller supplied one), TransformerLayer defaults attention_mask=None, and
        GroupQueryAttention masks only when given one -- so no causal mask
        existed anywhere in the package and every position saw the future.
        """
        keras.utils.set_random_seed(1234)
        model = Qwen3(**self.CFG)
        a, b = self._pair()

        la = keras.ops.convert_to_numpy(model(a, training=False))
        lb = keras.ops.convert_to_numpy(model(b, training=False))

        for pos in range(7):
            np.testing.assert_allclose(
                la[0, pos], lb[0, pos], atol=1e-5,
                err_msg=(f"position {pos} changed when only position 7 changed "
                         f"-- the future is leaking into the past"))

    def test_the_probe_is_not_vacuous(self):
        """The perturbation must actually change SOMETHING.

        Without this, a model that ignored its input entirely would satisfy the
        isolation test above.
        """
        keras.utils.set_random_seed(1234)
        model = Qwen3(**self.CFG)
        a, b = self._pair()

        la = keras.ops.convert_to_numpy(model(a, training=False))
        lb = keras.ops.convert_to_numpy(model(b, training=False))

        delta = float(np.abs(la[0, 7] - lb[0, 7]).max())
        assert delta > 1e-3, (
            f"changing position 7 moved its own logits by only {delta:.4e}; "
            f"the causality assertion would be vacuous")

    def test_without_the_mask_the_same_tokens_do_leak(self):
        """Negative control: the isolation is due to the MASK.

        Feeding an all-ones ATTEND mask (i.e. no causal structure) to the same
        blocks must let the identical perturbation reach the earlier positions.
        If it does not, the 0.0 above would be explained by something other than
        masking and would prove nothing.
        """
        keras.utils.set_random_seed(1234)
        model = Qwen3(**self.CFG)
        a, b = self._pair()
        model(a, training=False)  # build

        def _forward_unmasked(ids):
            h = model.embeddings(ids)
            seq = keras.ops.shape(h)[1]
            allow = keras.ops.ones((keras.ops.shape(h)[0], seq, seq), dtype="bool")
            for block in model.blocks:
                h = block(h, attention_mask=allow, training=False)
            return keras.ops.convert_to_numpy(model.lm_head(model.final_norm(h)))

        la = _forward_unmasked(a)
        lb = _forward_unmasked(b)

        leak = float(np.abs(la[0, :7] - lb[0, :7]).max())
        assert leak > 1e-3, (
            f"even with an all-attend mask the prefix did not move "
            f"(max |delta| = {leak:.4e}); the masked run's agreement would then "
            f"not be attributable to the causal mask")


# ---------------------------------------------------------------------
# include_top: the model's own task head
# ---------------------------------------------------------------------

_SEQ = 8
_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
)
_NER = NLPTaskConfig(
    name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9
)
#: Same head class and weight shapes as ``_CLF``, a different task.
_SPAM = NLPTaskConfig(
    name="spam", task_type=NLPTaskType.SPAM_DETECTION, num_classes=3
)


def _np(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _head_batch() -> Dict[str, np.ndarray]:
    """Row 1 is right-padded: 5 real tokens, then 3 pad ids under a 0 mask."""
    ids = np.random.default_rng(0).integers(1, 64, (2, _SEQ)).astype("int32")
    mask = np.ones_like(ids)
    ids[1, 5:] = 0
    mask[1, 5:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _relative_paths(model: keras.Model) -> list:
    """Weight paths without the model-root segment (guide 8.3)."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


def _randomized(model: keras.Model, seed: int) -> keras.Model:
    """Overwrite every weight, so a restored value cannot match by init luck."""
    assert model.weights, "build the model before randomizing it"
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    return model


def _backbone(model: keras.Model) -> Dict[str, list]:
    """Weights of every layer but the two tops (``lm_head``, ``task_head``)."""
    return {
        layer.name: [_np(w) for w in layer.weights]
        for layer in model.layers
        if layer.name not in ("lm_head", TASK_HEAD_NAME) and layer.weights
    }


class IncludeTopCases:
    """The ``include_top`` knobs of a causal decoder that ends in ``lm_head``.

    Not collected itself (no ``Test`` prefix). ``TestQwen3IncludeTop`` below,
    ``TestQwen3NextIncludeTop`` in ``test_qwen3_next.py`` and
    ``TestGemma3IncludeTop`` in ``test_gemma/test_gemma3.py`` bind ``CLS``,
    ``KW``, the ``create_<model>`` factory and the measured backbone weight
    count. A backbone moves between tops through
    ``load_weights_from_checkpoint`` (D-007 route R2 of
    plan-2026-10-11T004932-b83935e1) and through the factory's
    ``pretrained=<path>``.
    """

    CLS: Any = None
    KW: Dict[str, Any] = {}
    #: ``create_<model>(variant, pretrained=False, **kwargs)``.
    FACTORY: Optional[Callable[..., keras.Model]] = None
    #: Name the retired functional wrappers gave the class inside them.
    WRAPPED_NAME: str = ""
    #: Backbone weights at ``KW``; ``lm_head`` adds 1, a TEXT_CLASSIFICATION
    #: head 8.
    N_BACKBONE: int = 0

    def model(self, **kw: Any) -> keras.Model:
        return self.CLS(vocab_size=64, **self.KW, **kw)

    def from_factory(self, **kw: Any) -> keras.Model:
        """``create_<model>("tiny", ...)`` shrunk to ``KW`` and vocabulary 64."""
        return self.FACTORY("tiny", **dict(self.KW, vocab_size=64, **kw))

    def test_head_output_shapes(self):
        """Bare logits for single-output heads, a dict for QA; nothing else."""
        batch = _head_batch()
        out = self.model(include_top=True, task_config=_CLF)(batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, 3)

        out = self.model(include_top=True, task_config=_NER)(batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, _SEQ, 9)

        qa = self.model(include_top=True, task_config=NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2))
        out = qa(batch, training=False)
        assert set(out) == {"start_logits", "end_logits"}
        assert tuple(out["start_logits"].shape) == (2, _SEQ)

    def test_a_task_head_replaces_the_lm_head(self):
        """``build()`` must not create the LM head ``call()`` no longer runs."""
        headed = self.model(include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert headed.lm_head is None
        assert not [w.path for w in headed.weights if "lm_head" in w.path]

        headless = self.model()
        headless(_head_batch(), training=False)
        assert [w.path.split("/", 1)[-1] for w in headless.weights
                if "lm_head" in w.path] == ["lm_head/kernel"]

    def test_symbolic_output_matches_the_eager_one(self):
        """A symbolic call reports what ``call`` returns, headless or headed."""
        headed = self.model(include_top=True, task_config=_NER)
        symbolic = headed(keras.Input(shape=(None,), dtype="int32"))
        assert not isinstance(symbolic, dict)
        assert tuple(symbolic.shape) == (None, None, 9)
        symbolic = self.model()(keras.Input(shape=(None,), dtype="int32"))
        assert tuple(symbolic.shape) == (None, None, 64)

    def test_the_default_pooling_is_last_and_an_override_wins(self):
        """Causal decoder: pool the last real token, not position 0 (D-012)."""
        assert self.model(include_top=True, task_config=_CLF
                          ).task_head.pooling_type == "last"
        assert self.model(include_top=True, task_config=_CLF,
                          head_config_overrides={"pooling_type": "mean"}
                          ).task_head.pooling_type == "mean"

    def test_the_head_reads_the_last_real_token(self):
        """Changing the last real token changes the head output.

        Under ``'cls'`` pooling the head reads position 0, which in a causal
        decoder has seen only itself, so this change would be invisible.
        """
        model = self.model(include_top=True, task_config=_CLF)
        batch = _head_batch()
        base = _np(model(batch, training=False))
        edited = {k: v.copy() for k, v in batch.items()}
        edited["input_ids"][1, 4] = (edited["input_ids"][1, 4] % 63) + 1
        moved = _np(model(edited, training=False))
        np.testing.assert_array_equal(moved[0], base[0])
        assert not np.allclose(moved[1], base[1], atol=1e-6, rtol=0)

    @pytest.mark.parametrize("with_mask", [True, False],
                             ids=["attention_mask", "pad_derived"])
    def test_appended_padding_does_not_change_the_head_output(self, with_mask):
        """Padding after the last real token is invisible to the head.

        The decoder is causal, so the real positions' states do not depend on
        what follows them; the head must not pool the padding either. Without
        an ``attention_mask`` the head's mask is ``input_ids != pad_token_id``
        (here 0, passed explicitly: the default is None, which pools every
        position and is covered by ``test_a_decoder_assumes_no_padding_token``).
        """
        model = self.model(pad_token_id=0, include_top=True, task_config=_CLF)
        ids = np.random.default_rng(3).integers(1, 64, (2, 5)).astype("int32")
        padded = np.concatenate([ids, np.zeros((2, 4), "int32")], axis=1)

        def run(x):
            if with_mask:
                return _np(model({"input_ids": x,
                                  "attention_mask": (x != 0).astype("int32")},
                                 training=False))
            return _np(model(x, training=False))

        np.testing.assert_allclose(run(padded), run(ids), atol=1e-5, rtol=0)

    def test_a_custom_pad_token_id_drives_the_derived_mask(self):
        """``pad_token_id`` is the padding id when no mask is given."""
        model = self.model(pad_token_id=7, include_top=True, task_config=_CLF)
        ids = np.random.default_rng(4).integers(8, 64, (2, 5)).astype("int32")
        padded = np.concatenate([ids, np.full((2, 3), 7, "int32")], axis=1)
        np.testing.assert_allclose(_np(model(padded, training=False)),
                                   _np(model(ids, training=False)),
                                   atol=1e-5, rtol=0)

    def test_config_round_trip_with_a_head(self):
        """Every knob is serialized; ``task_config`` as a plain JSON dict."""
        import json

        overrides = {"pooling_type": "mean"}
        model = self.model(pad_token_id=5, include_top=True, task_config=_CLF,
                           head_config_overrides=overrides)
        config = model.get_config()
        assert config["include_top"] is True
        assert config["pad_token_id"] == 5
        assert config["head_config_overrides"] == overrides
        assert isinstance(config["task_config"], dict)
        assert config["task_config"]["task_type"] == "text_classification"
        json.dumps(config)  # plain data, no enum or dataclass inside

        from_dict = self.model(include_top=True,
                               task_config=dict(config["task_config"]),
                               head_config_overrides=overrides)
        assert from_dict.get_config()["task_config"] == config["task_config"]

        rebuilt = self.CLS.from_config(config)
        assert rebuilt.get_config() == config
        assert rebuilt.task_head.pooling_type == "mean"

        batch = _head_batch()
        model(batch, training=False)
        rebuilt(batch, training=False)
        assert _relative_paths(rebuilt) == _relative_paths(model)
        rebuilt.set_weights(model.get_weights())
        np.testing.assert_array_equal(
            _np(rebuilt(batch, training=False)), _np(model(batch, training=False)))

    def test_a_config_without_the_new_keys_still_loads(self):
        """A config written before the knobs existed rebuilds the headless model."""
        config = self.model().get_config()
        for key in ("include_top", "task_config", "head_config_overrides",
                    "pad_token_id"):
            config.pop(key)
        model = self.CLS.from_config(config)
        assert model.include_top is False and model.task_head is None
        assert model.pad_token_id is None and model.lm_head is not None

    def test_keras_round_trip_restores_the_head_by_value(self, tmp_path):
        """Weights compared BEFORE the first call on the loaded model (guide 8.4)."""
        model = self.model(include_top=True, task_config=_CLF)
        batch = _head_batch()
        model(batch, training=False)
        _randomized(model, 5)
        before = _np(model(batch, training=False))
        saved = [_np(w).copy() for w in model.weights]
        head_saved = [_np(w).copy() for w in model.task_head.weights]
        assert head_saved, "the head has no weights to compare"

        path = str(tmp_path / "headed.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert len(loaded.weights) == len(saved)
        for s, w in zip(saved, loaded.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0,
                                       err_msg=f"{w.path} not restored")
        for s, w in zip(head_saved, loaded.task_head.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(_np(loaded(batch, training=False)), before)

    @pytest.mark.parametrize("include_top", [False, True])
    def test_explicit_build_matches_lazy_build(self, include_top):
        """``build()`` materializes exactly the tree ``call()`` runs (guide 8.1)."""
        kw = dict(include_top=include_top,
                  task_config=_CLF if include_top else None)
        explicit = self.model(**kw)
        explicit.build((None, _SEQ))
        lazy = self.model(**kw)
        lazy(_head_batch(), training=False)
        assert _relative_paths(explicit) == _relative_paths(lazy)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_a_disabled_head_builds_no_head_weights(self):
        """Anti-vacuity sibling of the parity arm (guide 8.3)."""
        headless = self.model()
        headless(_head_batch(), training=False)
        assert headless.task_head is None
        assert not [w.path for w in headless.weights if "task_head" in w.path]

        headed = self.model(include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert [w.path for w in headed.weights if "/task_head/" in w.path]

    def test_the_backbone_weight_paths_do_not_depend_on_the_head(self):
        """The backbone's paths are identical across tops (D-005 of the plan).

        The backbone is every weight but ``lm_head/`` and ``task_head/``; the
        headless model adds ``lm_head/kernel``, a head 8 weights after it.
        """
        def paths(**kw):
            m = self.model(name="m", **kw)
            m(_head_batch(), training=False)
            return [w.path for w in m.weights]

        headless = paths()
        backbone = [p for p in headless if not p.startswith("m/lm_head/")]
        assert (len(backbone), len(headless)) == (
            self.N_BACKBONE, self.N_BACKBONE + 1)
        for task in (_CLF, _NER):
            headed = paths(include_top=True, task_config=task)
            assert headed[:len(backbone)] == backbone, task.name
            assert all(p.startswith("m/task_head/")
                       for p in headed[len(backbone):]), task.name
            assert len(headed) > len(backbone)

    def test_backbone_weights_transfer_across_heads(self, tmp_path):
        """A backbone trained under one head is reused, same weights, under another.

        ``load_weights_from_checkpoint(..., skip_prefixes=(TASK_HEAD_NAME,))``
        copies the backbone by layer name into a TOKEN_CLASSIFICATION model
        and into a headless one, and never the checkpoint's ``task_head``: a
        SPAM head of the same shapes keeps its own initialization.
        """
        batch = _head_batch()
        source = self.model(include_top=True, task_config=_CLF)
        source(batch, training=False)
        _randomized(source, 1)
        path = str(tmp_path / "trained_under_clf.keras")
        source.save(path)
        reference = _backbone(source)

        for target in (self.model(include_top=True, task_config=_NER),
                       self.model(include_top=True, task_config=_SPAM),
                       self.model()):
            target(batch, training=False)
            fresh_head = [_np(w).copy() for w in target.task_head.weights] \
                if target.task_head is not None else []
            load_weights_from_checkpoint(
                target, path, skip_prefixes=(TASK_HEAD_NAME,), strict=True)
            got = _backbone(target)
            assert set(got) == set(reference)
            for name, values in reference.items():
                for a, b in zip(values, got[name]):
                    np.testing.assert_array_equal(a, b, err_msg=name)
            if target.task_head is not None:
                for a, w in zip(fresh_head, target.task_head.weights):
                    np.testing.assert_array_equal(a, _np(w))

        # Anti-vacuity: a fresh backbone does NOT hold the reference values.
        fresh = self.model()
        fresh(batch, training=False)
        assert not all(np.array_equal(a[0], b[0]) for a, b in
                       zip(reference.values(), _backbone(fresh).values()))

    def test_the_factory_returns_the_class_with_its_head(self):
        """``create_<model>`` adds no layer of its own: it returns the class.

        The head's knobs reach the in-model head: ``head_config_overrides``
        as the head's ``pooling_type`` and the task's ``dropout_rate`` as the
        head's dropout.
        """
        model = self.from_factory(
            include_top=True,
            task_config=NLPTaskConfig(
                name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION,
                num_classes=4, dropout_rate=0.25),
            head_config_overrides={"pooling_type": "mean"})
        assert type(model) is self.CLS
        assert model.include_top is True and model.lm_head is None
        assert model.task_config["num_classes"] == 4
        assert model.task_head.pooling_type == "mean"
        assert model.task_head.dropout.rate == 0.25
        assert tuple(model(_head_batch(), training=False).shape) == (2, 4)

        headless = self.from_factory()
        assert type(headless) is self.CLS
        assert headless.include_top is False and headless.task_head is None
        assert tuple(headless(_head_batch(), training=False).shape) == (
            2, _SEQ, 64)

    def test_an_old_wrapper_backbone_moves_through_the_factory(self, tmp_path):
        """A backbone saved inside an old two-input wrapper fills any top.

        The retired factories returned ``keras.Model([input_ids,
        attention_mask], core(...))`` around the class, named
        ``WRAPPED_NAME``; such archives still exist. The shared load rule
        reads through that one-model wrapper (D-015 and D-027 of the earlier
        plans), so ``create_<model>(..., pretrained=<path>)`` restores the
        backbone into a classifier, a token classifier and a headless class;
        the checkpoint's equal-shape ``task_head`` is never copied. The
        headless result's own checkpoint then fills a fresh classifier
        exactly.
        """
        batch = _head_batch()
        core = self.model(include_top=True, task_config=_CLF,
                          name=self.WRAPPED_NAME)
        ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
        mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")
        wrapper = keras.Model(
            [ids, mask], core({"input_ids": ids, "attention_mask": mask}))
        _randomized(wrapper, 1)
        path = str(tmp_path / "old_wrapper_classifier.keras")
        wrapper.save(path)
        reference = _backbone(core)

        def check(target):
            got = _backbone(target)
            assert set(got) == set(reference)
            for name, values in reference.items():
                for a, b in zip(values, got[name]):
                    np.testing.assert_array_equal(a, b, err_msg=name)

        def head_is_fresh(target):
            pairs = [(a, b) for a, b in zip(target.task_head.weights,
                                             core.task_head.weights)
                     if len(a.shape) > 1]
            assert pairs and all(a.shape == b.shape for a, b in pairs)
            assert not any(np.array_equal(_np(a), _np(b)) for a, b in pairs)

        clf = self.from_factory(pretrained=path, include_top=True,
                                task_config=_CLF)
        assert type(clf) is self.CLS
        check(clf)
        head_is_fresh(clf)  # an equal-shape checkpoint head is never copied
        ner = self.from_factory(pretrained=path, include_top=True,
                                task_config=_NER)
        check(ner)
        headless = self.from_factory(pretrained=path)
        check(headless)
        # The classifier archive holds no lm_head; the headless one is its own.
        assert headless.lm_head is not None and core.lm_head is None

        path = str(tmp_path / "headless.keras")
        headless.save(path)
        target = self.from_factory(pretrained=path, include_top=True,
                                   task_config=_CLF)
        check(target)
        head_is_fresh(target)

        # Anti-vacuity: a fresh backbone does NOT hold the reference values.
        fresh = self.model()
        fresh(batch, training=False)
        assert not all(np.array_equal(a[0], b[0]) for a, b in
                       zip(reference.values(), _backbone(fresh).values()))

    @pytest.mark.parametrize("retired", [
        {"task_type": "classification"},
        {"num_labels": 3},
        {"pooling_strategy": "mean"},
        {"classifier_dropout_rate": 0.5},
    ], ids=["task_type", "num_labels", "pooling_strategy",
            "classifier_dropout_rate"])
    def test_the_factory_refuses_the_retired_task_knobs(self, retired):
        """The per-task dispatch is gone, and its knobs are refused, not ignored.

        A head comes only from ``include_top=True`` with a ``task_config``;
        ``task_type`` and the classification knobs reach the constructor and
        Keras refuses them, so a caller of the old API gets an error, never a
        silently headless model.
        """
        with pytest.raises(ValueError, match="Unrecognized keyword arguments"):
            self.from_factory(**retired)

    def test_include_top_without_task_config_raises(self):
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            self.model(include_top=True)

    @pytest.mark.parametrize("dead", [
        {"task_config": _CLF},
        {"head_config_overrides": {"pooling_type": "mean"}},
    ], ids=["task_config", "head_config_overrides"])
    def test_head_knobs_without_include_top_raise(self, dead):
        with pytest.raises(ValueError, match="only take effect with include_top=True"):
            self.model(**dead)


class QwenIncludeTopCases(IncludeTopCases):
    """``IncludeTopCases`` plus the Qwen-only ``return_dict`` call argument."""

    def test_return_dict_wraps_a_bare_head_output(self):
        """``return_dict=True`` keeps its meaning under a head.

        A bare head output comes back under ``logits``; a multi-output head's
        dict comes back as is.
        """
        batch = _head_batch()
        model = self.model(include_top=True, task_config=_CLF)
        out = model(batch, training=False, return_dict=True)
        assert set(out) == {"logits"}
        np.testing.assert_array_equal(_np(out["logits"]),
                                      _np(model(batch, training=False)))
        qa = self.model(include_top=True, task_config=NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2))
        assert set(qa(batch, training=False, return_dict=True)) == {
            "start_logits", "end_logits"}


class TestQwen3IncludeTop(QwenIncludeTopCases):
    """``Qwen3`` with ``include_top`` (2 blocks: 24 backbone weights, measured).

    ``test_explicit_build_matches_lazy_build[False]`` also guards the new
    ``Qwen3.build``: before it, an explicit build created no weight at all.
    """

    CLS = Qwen3
    KW = dict(hidden_size=32, num_layers=2, num_attention_heads=4,
              num_key_value_heads=2, max_seq_len=16, moe_layers=[],
              num_experts=1, num_experts_per_tok=1, dropout_rate=0.0)
    FACTORY = staticmethod(create_qwen3)
    WRAPPED_NAME = "qwen3_backbone"
    N_BACKBONE = 24
