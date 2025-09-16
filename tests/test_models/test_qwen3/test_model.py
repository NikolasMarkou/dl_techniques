"""
Refined Tests for the Refactored Qwen3 Model Implementation

This module provides comprehensive tests for the refactored Qwen3 model, which
follows a modern backbone-and-head design pattern.

Tests cover:
- The Qwen3Model backbone's core functionality (hidden state output).
- The end-to-end models created by task-specific factories (generation, classification).
- Serialization/deserialization of both backbone and full models.
- Factory function logic, including variants, overrides, and error handling.
- Integration tests simulating a basic training loop.
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
)

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def tiny_config() -> Dict[str, Any]:
    """Create a tiny model configuration for fast testing (non-MoE)."""
    return {
        'vocab_size': 1000,
        'hidden_size': 128,
        'num_layers': 2,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'intermediate_size': 256,
        'max_seq_len': 512,
        'moe_layers': [],
        'rope_theta': 10000.0,
        'dropout_rate': 0.1,
    }

@pytest.fixture
def mini_moe_config() -> Dict[str, Any]:
    """Create a mini model configuration with MoE for testing."""
    return {
        'vocab_size': 800,
        'hidden_size': 96,
        'num_layers': 4,
        'num_attention_heads': 6,
        'num_key_value_heads': 2,
        'intermediate_size': 192,
        'max_seq_len': 256,
        'moe_layers': [1, 3],
        'num_experts': 4,
        'num_experts_per_tok': 2,
        'moe_intermediate_size': 128,
        'rope_theta': 10000.0,
        'dropout_rate': 0.0,
    }

@pytest.fixture
def sample_input_ids() -> keras.KerasTensor:
    """Create sample input tensor for testing."""
    return keras.ops.convert_to_tensor(
        np.random.randint(0, 800, size=(2, 32)), dtype='int32'
    )

@pytest.fixture
def sample_inputs(sample_input_ids: keras.KerasTensor) -> Dict[str, keras.KerasTensor]:
    """Create a sample dictionary of inputs."""
    attention_mask = keras.ops.ones_like(sample_input_ids)
    return {"input_ids": sample_input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------

class TestQwen3Backbone:
    """Test the core Qwen3Model backbone functionality."""

    def test_backbone_creation(self, tiny_config: Dict[str, Any]):
        """Test backbone creation and attribute initialization."""
        backbone = Qwen3(**tiny_config)
        assert backbone.hidden_size == tiny_config['hidden_size']
        assert backbone.num_layers == tiny_config['num_layers']
        assert len(backbone.transformer_blocks) == tiny_config['num_layers']
        assert backbone.head_dim == tiny_config['hidden_size'] // tiny_config['num_attention_heads']
        assert not backbone.moe_layers
        assert backbone.moe_config is None

    def test_backbone_forward_pass(self, tiny_config: Dict[str, Any], sample_input_ids: keras.KerasTensor):
        """Test the forward pass of the backbone returns hidden states."""
        backbone = Qwen3(**tiny_config)
        hidden_states = backbone(sample_input_ids)

        expected_shape = (
            sample_input_ids.shape[0], sample_input_ids.shape[1], tiny_config['hidden_size']
        )
        assert hidden_states.shape == expected_shape
        output_np = keras.ops.convert_to_numpy(hidden_states)
        assert not np.any(np.isnan(output_np))

    def test_backbone_with_moe(self, mini_moe_config: Dict[str, Any], sample_input_ids: keras.KerasTensor):
        """Test MoE configuration and forward pass in the backbone."""
        backbone = Qwen3(**mini_moe_config)
        assert backbone.moe_layers == [1, 3]
        assert backbone.moe_config is not None
        assert backbone.moe_config.num_experts == mini_moe_config['num_experts']

        hidden_states = backbone(sample_input_ids, training=True)
        expected_shape = (
            sample_input_ids.shape[0], sample_input_ids.shape[1], mini_moe_config['hidden_size']
        )
        assert hidden_states.shape == expected_shape

    def test_auxiliary_loss(self, mini_moe_config: Dict[str, Any], sample_input_ids: keras.KerasTensor):
        """Test that the backbone can expose the auxiliary loss from MoE layers."""
        backbone = Qwen3(**mini_moe_config)
        _ = backbone(sample_input_ids, training=True)
        aux_loss = backbone.get_auxiliary_loss()
        assert aux_loss is not None
        assert aux_loss.shape == ()
        assert aux_loss >= 0


class TestQwen3FactoriesAndFullModels:
    """Test the factory functions and the complete, task-specific models."""

    def test_create_with_variant(self):
        """Test creating a model from a variant string."""
        model = create_qwen3("small", task_type="generation")
        assert isinstance(model, keras.Model)
        assert model.name == "qwen3_for_generation"
        assert model.output_shape[-1] == Qwen3.MODEL_VARIANTS["small"]["vocab_size"]

    def test_create_with_override(self):
        """Test overriding variant parameters with kwargs."""
        model = create_qwen3("small", num_layers=4, hidden_size=64)
        backbone = model.get_layer("qwen3_backbone")
        assert backbone.num_layers == 4
        assert backbone.hidden_size == 64
        assert backbone.vocab_size == Qwen3.MODEL_VARIANTS["small"]["vocab_size"]

    def test_generation_model(self, tiny_config: Dict[str, Any], sample_inputs: Dict[str, Any]):
        """Test the generation model factory and its output."""
        model = create_qwen3(tiny_config, task_type="generation")
        logits = model(sample_inputs)
        expected_shape = (
            sample_inputs["input_ids"].shape[0],
            sample_inputs["input_ids"].shape[1],
            tiny_config['vocab_size'],
        )
        assert logits.shape == expected_shape

    def test_classification_model(self, tiny_config: Dict[str, Any], sample_inputs: Dict[str, Any]):
        """Test the classification model factory and pooling strategies."""
        num_labels = 10
        # Test with 'cls' pooling
        cls_model = create_qwen3(tiny_config, task_type="classification", num_labels=num_labels, pooling_strategy="cls")
        cls_logits = cls_model(sample_inputs)
        assert cls_logits.shape == (sample_inputs["input_ids"].shape[0], num_labels)

        # Test with 'mean' pooling
        mean_model = create_qwen3(tiny_config, task_type="classification", num_labels=num_labels, pooling_strategy="mean")
        mean_logits = mean_model(sample_inputs)
        assert mean_logits.shape == (sample_inputs["input_ids"].shape[0], num_labels)

    def test_factory_error_handling(self):
        """Test error conditions for the main factory."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_qwen3("non_existent_variant")
        with pytest.raises(ValueError, match="Unknown task_type"):
            create_qwen3("small", task_type="regression")
        with pytest.raises(ValueError, match="`num_labels` must be provided"):
            create_qwen3("small", task_type="classification")

    def test_weight_tying_factory(self, tiny_config: Dict[str, Any]):
        """Test that the `use_weight_tying` flag works correctly in the factory."""
        # Model with weight tying (default)
        model_tied = create_qwen3(tiny_config, use_weight_tying=True)
        tied_params = model_tied.count_params()

        # Model without weight tying
        model_separate = create_qwen3(tiny_config, use_weight_tying=False)
        separate_params = model_separate.count_params()

        assert separate_params > tied_params
        expected_diff = tiny_config['vocab_size'] * tiny_config['hidden_size']
        actual_diff = separate_params - tied_params
        assert abs(actual_diff - expected_diff) < 10  # Should be exact

    def test_vocab_padding_with_tying(self, tiny_config: Dict[str, Any]):
        """Test that vocab padding works correctly with weight tying in the factory."""
        config_padded = tiny_config.copy()
        config_padded['vocab_padding_size'] = 2048

        model = create_qwen3(config_padded, task_type="generation", use_weight_tying=True)
        backbone = model.get_layer("qwen3_backbone")

        assert backbone.final_vocab_size == 2048
        assert backbone.token_embedding.input_dim == 2048
        # The final output logits shape should match the *original* vocab size
        assert model.output_shape[-1] == tiny_config['vocab_size']


class TestQwen3Serialization:
    """Test serialization for both backbone and full models."""

    def test_backbone_serialization(self, mini_moe_config: Dict[str, Any], sample_input_ids: keras.KerasTensor):
        """Test serialization cycle for the Qwen3Model backbone."""
        original_backbone = Qwen3(**mini_moe_config)
        original_output = original_backbone(sample_input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'qwen3_backbone.keras')
            original_backbone.save(model_path)
            loaded_backbone = keras.models.load_model(model_path)
            loaded_output = loaded_backbone(sample_input_ids, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-5, atol=1e-5
            )
            assert isinstance(loaded_backbone, Qwen3)

    def test_full_model_serialization(self, tiny_config: Dict[str, Any], sample_inputs: Dict[str, Any]):
        """Test serialization cycle for a full generation model."""
        original_model = create_qwen3(tiny_config, task_type="generation")
        original_output = original_model(sample_inputs, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'qwen3_generation.keras')
            original_model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_output = loaded_model(sample_inputs, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_output),
                keras.ops.convert_to_numpy(loaded_output),
                rtol=1e-5, atol=1e-5
            )
            assert loaded_model.name == "qwen3_for_generation"


class TestQwen3ErrorHandling:
    """Test error handling and edge cases for the backbone."""

    def test_invalid_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            Qwen3(vocab_size=100, hidden_size=0, num_layers=1, num_attention_heads=1)

    def test_mismatched_heads_and_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size .* must be divisible by num_attention_heads"):
            Qwen3(vocab_size=100, hidden_size=129, num_layers=1, num_attention_heads=4)

    def test_mismatched_heads_and_kv_heads(self):
        with pytest.raises(ValueError, match="num_attention_heads .* must be divisible by num_key_value_heads"):
            Qwen3(vocab_size=100, hidden_size=120, num_layers=1, num_attention_heads=6, num_key_value_heads=4)

    def test_invalid_dropout_rate(self):
        with pytest.raises(ValueError, match="dropout_rate must be between"):
            Qwen3(vocab_size=100, hidden_size=128, num_layers=1, num_attention_heads=4, dropout_rate=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])