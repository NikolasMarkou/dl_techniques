import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf


from dl_techniques.models.modern_bert.model import (
    ModernBERT,
    create_modern_bert_base,
    create_modern_bert_large,
    create_modern_bert_for_classification,
    create_modern_bert_for_sequence_output,
)


class TestModernBERT:
    """
    Comprehensive test suite for the ModernBERT model, following modern Keras testing practices.
    """

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Provides a standard, small configuration for ModernBERT to speed up tests."""
        return {
            'vocab_size': 1000,
            'hidden_size': 128,
            'num_layers': 4,
            'num_heads': 4,
            'intermediate_size': 256,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'type_vocab_size': 2,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'use_bias': False,
            'rope_theta_local': 10000.0,
            'rope_theta_global': 160000.0,
            'max_seq_len': 512,
            'global_attention_interval': 2,
            'local_attention_window_size': 32,
            'add_pooling_layer': True,
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Provides a sample dictionary of input tensors for testing."""
        batch_size, seq_len = 2, 64
        return {
            "input_ids": keras.ops.cast(
                keras.random.uniform((batch_size, seq_len), 0, 1000), 'int32'
            ),
            "token_type_ids": keras.ops.cast(
                keras.random.uniform((batch_size, seq_len), 0, 2), 'int32'
            ),
            "attention_mask": keras.ops.cast(
                keras.random.uniform((batch_size, seq_len), 0, 2), 'int32'
            )
        }

    def test_initialization_and_architecture(self, model_config):
        """
        Tests if the model initializes correctly and sets up its architecture as expected,
        including the correct number of layers and the global/local attention pattern.
        """
        model = ModernBERT(**model_config)

        assert model.built is False
        assert isinstance(model.final_norm, keras.layers.LayerNormalization)
        assert len(model.encoder_layers) == model_config['num_layers']
        assert isinstance(model.pooler, keras.layers.Dense)

        # Verify the global attention interval logic
        for i, layer in enumerate(model.encoder_layers):
            is_global_expected = (i + 1) % model_config['global_attention_interval'] == 0
            assert layer.is_global == is_global_expected

    def test_forward_pass_with_pooling(self, model_config, sample_inputs):
        """Tests the forward pass when the pooling layer is enabled."""
        config = model_config.copy()
        config['add_pooling_layer'] = True
        model = ModernBERT(**config)
        batch_size, seq_len, hidden_size = 2, 64, config['hidden_size']

        # Test with return_dict=False (default) -> returns a tuple
        sequence_output, pooled_output = model(sample_inputs)
        assert sequence_output.shape == (batch_size, seq_len, hidden_size)
        assert pooled_output.shape == (batch_size, hidden_size)

        # Test with return_dict=True -> returns a dictionary
        outputs_dict = model(sample_inputs, return_dict=True)
        assert isinstance(outputs_dict, dict)
        assert "last_hidden_state" in outputs_dict
        assert "pooler_output" in outputs_dict
        assert outputs_dict["last_hidden_state"].shape == (batch_size, seq_len, hidden_size)
        assert outputs_dict["pooler_output"].shape == (batch_size, hidden_size)

    def test_forward_pass_without_pooling(self, model_config, sample_inputs):
        """Tests the forward pass when the pooling layer is disabled."""
        config = model_config.copy()
        config['add_pooling_layer'] = False
        model = ModernBERT(**config)
        batch_size, seq_len, hidden_size = 2, 64, config['hidden_size']

        # Test with return_dict=False (default) -> returns a single tensor
        sequence_output = model(sample_inputs)
        assert not isinstance(sequence_output, (tuple, dict))
        assert sequence_output.shape == (batch_size, seq_len, hidden_size)

        # Test with return_dict=True -> returns a dictionary
        outputs_dict = model(sample_inputs, return_dict=True)
        assert isinstance(outputs_dict, dict)
        assert "last_hidden_state" in outputs_dict
        assert "pooler_output" not in outputs_dict
        assert outputs_dict["last_hidden_state"].shape == (batch_size, seq_len, hidden_size)

    def test_forward_pass_input_types(self, model_config, sample_inputs):
        """Tests that the model handles both dictionary and positional tensor inputs."""
        model = ModernBERT(**model_config)
        # Dictionary input is tested in other forward pass tests.
        # Here we test positional tensor input.
        output_seq, output_pooled = model(
            sample_inputs["input_ids"],
            attention_mask=sample_inputs["attention_mask"],
            token_type_ids=sample_inputs["token_type_ids"]
        )
        assert output_seq.shape == (2, 64, model_config['hidden_size'])
        assert output_pooled.shape == (2, model_config['hidden_size'])

    def test_serialization_cycle(self, model_config, sample_inputs):
        """CRITICAL TEST: Ensures the model can be saved and reloaded correctly."""
        model = ModernBERT(**model_config)

        # Get original prediction in inference mode for reproducibility
        original_pred_seq, original_pred_pooled = model(sample_inputs, training=False)

        # Save and load the model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred_seq, loaded_pred_pooled = loaded_model(sample_inputs, training=False)

            # Verify predictions are identical
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred_seq),
                keras.ops.convert_to_numpy(loaded_pred_seq),
                rtol=1e-6, atol=1e-6,
                err_msg="Sequence output differs after serialization"
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred_pooled),
                keras.ops.convert_to_numpy(loaded_pred_pooled),
                rtol=1e-6, atol=1e-6,
                err_msg="Pooled output differs after serialization"
            )

    def test_config_completeness(self, model_config):
        """Tests that get_config returns all __init__ parameters for reconstruction."""
        model = ModernBERT(**model_config)
        config = model.get_config()

        # Check all original config parameters are present in the retrieved config
        for key, value in model_config.items():
            assert key in config, f"Missing key '{key}' in get_config()"
            assert config[key] == value, f"Config mismatch for key '{key}'"

    def test_gradients_flow(self, model_config, sample_inputs):
        """Tests that gradients can be computed through the model."""
        model = ModernBERT(**model_config)

        with tf.GradientTape() as tape:
            sequence_output, pooled_output = model(sample_inputs)
            # Define a simple loss based on both outputs
            loss = keras.ops.mean(sequence_output) + keras.ops.mean(pooled_output)

        gradients = tape.gradient(loss, model.trainable_variables)

        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "Some gradients are None."

    @pytest.mark.parametrize("training", [True, False])
    def test_training_modes(self, model_config, sample_inputs, training):
        """Tests model behavior in both training and inference modes."""
        model = ModernBERT(**model_config)

        # The model should run without errors in both modes
        sequence_output, _ = model(sample_inputs, training=training)
        assert sequence_output.shape[0] == sample_inputs["input_ids"].shape[0]

    def test_missing_input_ids_error(self, model_config):
        """Tests that a ValueError is raised if input_ids are not provided."""
        model = ModernBERT(**model_config)
        with pytest.raises(ValueError, match="input_ids must be provided"):
            model({})  # Pass an empty dictionary


class TestModernBERTFactoryFunctions:
    """Test suite for the factory and builder functions for ModernBERT."""

    def test_create_modern_bert_base(self):
        """Tests the configuration factory for ModernBERT-base."""
        config = create_modern_bert_base()
        assert isinstance(config, dict)
        assert config['hidden_size'] == 768
        assert config['num_layers'] == 22
        assert config['num_heads'] == 12
        # Ensure the config can create a model instance
        model = ModernBERT(**config)
        assert model.hidden_size == 768

    def test_create_modern_bert_large(self):
        """Tests the configuration factory for ModernBERT-large."""
        config = create_modern_bert_large()
        assert isinstance(config, dict)
        assert config['hidden_size'] == 1024
        assert config['num_layers'] == 28
        assert config['num_heads'] == 16
        # Ensure the config can create a model instance
        model = ModernBERT(**config)
        assert model.hidden_size == 1024

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Provides a minimal config for functional model tests."""
        return {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 2,
            'num_heads': 2,
            'intermediate_size': 64,
        }

    def test_create_modern_bert_for_classification(self, small_config):
        """Tests the builder for a classification model, including serialization."""
        num_labels = 5
        model = create_modern_bert_for_classification(small_config, num_labels=num_labels)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 3

        # Test forward pass and output shape
        batch_size, seq_len = 2, 16
        inputs = [
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            keras.ops.zeros((batch_size, seq_len), dtype='int32')
        ]
        logits = model(inputs)
        assert logits.shape == (batch_size, num_labels)

        # Test serialization cycle for the functional model
        original_pred = model(inputs, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'classification_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(inputs, training=False)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6
            )

    def test_create_modern_bert_for_sequence_output(self, small_config):
        """Tests the builder for a sequence output model, including serialization."""
        model = create_modern_bert_for_sequence_output(small_config)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 3

        # Test forward pass and output shape
        batch_size, seq_len = 2, 16
        inputs = [
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            keras.ops.zeros((batch_size, seq_len), dtype='int32')
        ]
        sequence_output = model(inputs)
        assert sequence_output.shape == (batch_size, seq_len, small_config['hidden_size'])

        # Test serialization cycle for the functional model
        original_pred = model(inputs, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'sequence_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(inputs, training=False)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])