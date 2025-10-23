"""
Comprehensive test suite for the FNet model implementation.

This test suite validates the FNet foundation model and its integration
with task-specific heads, using small memory footprint configurations
suitable for CPU testing.
"""

import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

# Imports from the project structure, assuming it's available in the path
from dl_techniques.models.fnet.model import (
    FNet,
    create_fnet_with_head,
)
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType


class TestFNetModel:
    """Test suite for the main FNet foundation model class."""

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Small configuration for CPU testing."""
        return {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 2,
            'intermediate_size': 64,
            'max_position_embeddings': 16,
            'hidden_dropout_prob': 0.1,
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample token inputs for testing."""
        batch_size = 2
        seq_len = 16
        return {
            'input_ids': keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len)),
            'attention_mask': keras.ops.ones((batch_size, seq_len), dtype="int32"),
            'token_type_ids': keras.ops.zeros((batch_size, seq_len), dtype="int32"),
            'position_ids': keras.ops.repeat(keras.ops.arange(seq_len)[None, :], repeats=batch_size, axis=0)
        }

    def test_basic_model_creation(self, small_config):
        """Test basic FNet model creation."""
        model = FNet(**small_config)
        assert isinstance(model, keras.Model)
        assert model.vocab_size == 100
        assert model.hidden_size == 32
        assert model.num_layers == 2

    def test_model_forward_pass_dict_input(self, small_config, sample_inputs):
        """Test model forward pass with dictionary input."""
        model = FNet(**small_config)
        output = model(sample_inputs)

        # Output should be a dictionary with specific keys
        assert isinstance(output, dict)
        assert 'last_hidden_state' in output
        assert 'attention_mask' in output

        # Check shapes
        assert output['last_hidden_state'].shape == (2, 16, 32)
        assert output['attention_mask'].shape == (2, 16)

    def test_model_forward_pass_tensor_input(self, small_config):
        """Test model forward pass with a single tensor input."""
        model = FNet(**small_config)
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(2, 16))
        output = model(input_ids)

        assert isinstance(output, dict)
        assert 'last_hidden_state' in output
        assert output['last_hidden_state'].shape == (2, 16, 32)

    def test_from_variant_method(self):
        """Test creating model from predefined variants."""
        variants = ['base', 'large', 'small', 'tiny']
        for variant in variants:
            model = FNet.from_variant(variant)
            assert isinstance(model, FNet)

            # Check variant-specific configurations
            expected_config = FNet.MODEL_VARIANTS[variant]
            assert model.hidden_size == expected_config['hidden_size']
            assert model.num_layers == expected_config['num_layers']

    def test_from_variant_with_overrides(self):
        """Test from_variant with configuration overrides."""
        model = FNet.from_variant(
            'tiny',
            hidden_dropout_prob=0.2,
            normalization_type='rms_norm'
        )

        assert model.hidden_dropout_prob == 0.2
        assert model.normalization_type == 'rms_norm'

    def test_model_serialization(self, small_config, sample_inputs):
        """Test model serialization and loading."""
        model = FNet(**small_config)
        original_pred = model(sample_inputs)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_fnet.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(sample_inputs)

            # Compare the main output tensor
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred["last_hidden_state"]),
                keras.ops.convert_to_numpy(loaded_pred["last_hidden_state"]),
                rtol=1e-6, atol=1e-6,
                err_msg="Loaded model outputs should match original"
            )

    def test_get_config_and_from_config(self, small_config):
        """Test configuration serialization."""
        model = FNet(**small_config)
        config = model.get_config()

        # Verify all expected keys are present
        expected_keys = {
            'vocab_size', 'hidden_size', 'num_layers', 'intermediate_size',
            'hidden_dropout_prob', 'max_position_embeddings'
        }
        assert expected_keys.issubset(set(config.keys()))

        # Test reconstruction
        new_model = FNet.from_config(config)
        assert new_model.vocab_size == model.vocab_size
        assert new_model.hidden_size == model.hidden_size
        assert new_model.num_layers == model.num_layers

    def test_invalid_variant(self):
        """Test error handling for invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant"):
            FNet.from_variant('invalid_variant')

    def test_invalid_input_dict(self, small_config):
        """Test error handling for invalid input dictionary."""
        model = FNet(**small_config)
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            model({'attention_mask': keras.ops.ones((2, 16))})

    def test_gradients_flow(self, small_config, sample_inputs):
        """Test that gradients flow through the model."""
        model = FNet(**small_config)

        with tf.GradientTape() as tape:
            output = model(sample_inputs)
            # Calculate loss based on the main output tensor
            loss = keras.ops.mean(keras.ops.square(output['last_hidden_state']))

        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that all gradients are not None
        assert all(g is not None for g in gradients)

        # Check that at least some gradients have meaningful magnitudes
        grad_norms = [keras.ops.convert_to_numpy(keras.ops.norm(g)) for g in gradients if g is not None]
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_model_with_different_input_lengths(self):
        """Test model with different sequence lengths by creating new instances."""
        base_config = {
            'vocab_size': 50,
            'hidden_size': 16,
            'num_layers': 1,
            'intermediate_size': 32,
        }

        for seq_len in [8, 16, 32]:
            # Create a new model instance for each sequence length
            config = base_config.copy()
            config['max_position_embeddings'] = seq_len
            model = FNet(**config)

            input_ids = keras.random.randint(minval=0, maxval=50, shape=(1, seq_len))
            output = model(input_ids)
            assert output['last_hidden_state'].shape == (1, seq_len, 16)

    def test_model_summary(self, small_config):
        """Test model summary functionality."""
        model = FNet(**small_config)
        # Build the model by calling it on some data
        input_ids = keras.random.randint(minval=0, maxval=100, shape=(1, 16))
        model(input_ids)
        # This should not raise any errors
        model.summary()


class TestFNetWithHeadFactory:
    """Test suite for the `create_fnet_with_head` factory function."""

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Sample full inputs for an end-to-end model."""
        batch_size = 2
        seq_len = 16
        return {
            'input_ids': keras.random.randint(minval=0, maxval=100, shape=(batch_size, seq_len)),
            'attention_mask': keras.ops.ones((batch_size, seq_len), dtype="int32"),
            'token_type_ids': keras.ops.zeros((batch_size, seq_len), dtype="int32"),
        }

    def test_create_with_sentiment_head(self, sample_inputs):
        """Test creating a model for sequence classification."""
        task_config = NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3
        )
        model = create_fnet_with_head(
            fnet_variant="tiny",
            task_config=task_config,
            fnet_config_overrides={"max_position_embeddings": 16, "vocab_size": 100},
            sequence_length=16
        )
        assert isinstance(model, keras.Model)

        output = model(sample_inputs)
        # For sentiment analysis, output should be (batch_size, num_classes)
        assert isinstance(output, dict)
        assert "logits" in output
        assert output["logits"].shape == (2, 3)

    def test_create_with_ner_head(self, sample_inputs):
        """Test creating a model for token classification."""
        task_config = NLPTaskConfig(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=9
        )
        model = create_fnet_with_head(
            fnet_variant="tiny",
            task_config=task_config,
            fnet_config_overrides={"max_position_embeddings": 16, "vocab_size": 100},
            sequence_length=16
        )
        assert isinstance(model, keras.Model)

        output = model(sample_inputs)
        # For NER (token classification), output should be (batch_size, seq_len, num_classes)
        assert isinstance(output, dict)
        assert "logits" in output
        assert output["logits"].shape == (2, 16, 9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])