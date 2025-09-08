import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

# Adapted imports for the new model
from dl_techniques.models.modern_bert_blt.model import (
    ModernBertBLT,
    create_modern_bert_blt_base_config,
    create_modern_bert_blt_for_classification,
)

# Helper for sequence output factory if not in the main file
def create_modern_bert_blt_for_sequence_output(config: Dict[str, Any]) -> keras.Model:
    model_instance = ModernBertBLT(add_pooling_layer=False, name="modern_bert_blt", **config)
    input_ids = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    attention_mask = keras.Input(shape=(None,), dtype="int32", name="attention_mask")
    sequence_output = model_instance(inputs={"input_ids": input_ids, "attention_mask": attention_mask})
    model = keras.Model(inputs=[input_ids, attention_mask], outputs=sequence_output)
    return model


class TestModernBertBLT:
    """
    Comprehensive test suite for the ModernBertBLT model.
    """

    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        """Provides a standard, small configuration for ModernBertBLT to speed up tests."""
        return {
            'vocab_size': 260,
            'hidden_size': 128,
            'num_layers': 4,
            'num_heads': 4,
            'intermediate_size': 256,
            'hidden_act': 'gelu',
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'initializer_range': 0.02,
            'layer_norm_eps': 1e-12,
            'use_bias': False,
            'rope_theta_local': 10000.0,
            'rope_theta_global': 160000.0,
            'max_seq_len': 512,
            'global_attention_interval': 2,
            'local_attention_window_size': 32,
            'add_pooling_layer': True,
            # BLT specific params
            'use_hash_embeddings': True,
            'hash_vocab_size': 1000,
            'ngram_sizes': [2, 3],
            'hash_embedding_dim': 128,
        }

    @pytest.fixture
    def sample_inputs(self) -> Dict[str, keras.KerasTensor]:
        """Provides a sample dictionary of input tensors for testing."""
        batch_size, seq_len = 2, 64
        return {
            "input_ids": keras.ops.cast(
                keras.random.uniform((batch_size, seq_len), 0, 260), 'int32'
            ),
            # Note: token_type_ids is removed for BLT
            "attention_mask": keras.ops.cast(
                keras.random.uniform((batch_size, seq_len), 0, 2), 'int32'
            )
        }

    def test_initialization_and_architecture(self, model_config):
        """
        Tests if the model initializes correctly and sets up its architecture as expected.
        """
        model = ModernBertBLT(**model_config)

        assert model.built is False
        assert isinstance(model.final_norm, keras.layers.LayerNormalization)
        assert len(model.encoder_layers) == model_config['num_layers']
        assert isinstance(model.pooler, keras.layers.Dense)

        # Verify the global attention interval logic (reused from ModernBERT)
        for i, layer in enumerate(model.encoder_layers):
            is_global_expected = (i + 1) % model_config['global_attention_interval'] == 0
            assert layer.is_global == is_global_expected

    def test_forward_pass_with_pooling(self, model_config, sample_inputs):
        """Tests the forward pass when the pooling layer is enabled."""
        config = model_config.copy()
        config['add_pooling_layer'] = True
        model = ModernBertBLT(**config)
        batch_size, seq_len, hidden_size = 2, 64, config['hidden_size']

        sequence_output, pooled_output = model(sample_inputs)
        assert sequence_output.shape == (batch_size, seq_len, hidden_size)
        assert pooled_output.shape == (batch_size, hidden_size)

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
        model = ModernBertBLT(**config)
        batch_size, seq_len, hidden_size = 2, 64, config['hidden_size']

        sequence_output = model(sample_inputs)
        assert not isinstance(sequence_output, (tuple, dict))
        assert sequence_output.shape == (batch_size, seq_len, hidden_size)

        outputs_dict = model(sample_inputs, return_dict=True)
        assert isinstance(outputs_dict, dict)
        assert "last_hidden_state" in outputs_dict
        assert "pooler_output" not in outputs_dict
        assert outputs_dict["last_hidden_state"].shape == (batch_size, seq_len, hidden_size)

    def test_forward_pass_input_types(self, model_config, sample_inputs):
        """Tests that the model handles both dictionary and positional tensor inputs."""
        model = ModernBertBLT(**model_config)
        output_seq, output_pooled = model(
            sample_inputs["input_ids"],
            attention_mask=sample_inputs["attention_mask"]
        )
        assert output_seq.shape == (2, 64, model_config['hidden_size'])
        assert output_pooled.shape == (2, model_config['hidden_size'])

    def test_serialization_cycle(self, model_config, sample_inputs):
        """CRITICAL TEST: Ensures the model can be saved and reloaded correctly."""
        model = ModernBertBLT(**model_config)
        original_pred_seq, original_pred_pooled = model(sample_inputs, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_pred_seq, loaded_pred_pooled = loaded_model(sample_inputs, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred_seq),
                keras.ops.convert_to_numpy(loaded_pred_seq),
                rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred_pooled),
                keras.ops.convert_to_numpy(loaded_pred_pooled),
                rtol=1e-6, atol=1e-6
            )

    def test_config_completeness(self, model_config):
        """Tests that get_config returns all __init__ parameters for reconstruction."""
        model = ModernBertBLT(**model_config)
        config = model.get_config()

        for key, value in model_config.items():
            assert key in config, f"Missing key '{key}' in get_config()"
            assert config[key] == value, f"Config mismatch for key '{key}'"

    def test_gradients_flow(self, model_config, sample_inputs):
        """Tests that gradients can be computed through the model."""
        model = ModernBertBLT(**model_config)
        with tf.GradientTape() as tape:
            sequence_output, pooled_output = model(sample_inputs)
            loss = keras.ops.mean(sequence_output) + keras.ops.mean(pooled_output)
        gradients = tape.gradient(loss, model.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "Some gradients are None."

    def test_encode_decode_cycle(self, model_config):
        """Tests the model's text encoding and decoding convenience methods."""
        model = ModernBertBLT(**model_config)
        text = "Hello, world! This is a test."

        # Test encoding
        encoded = model.encode_text(text, max_length=32, add_special_tokens=True)
        assert encoded.shape == (1, 32)
        assert encoded.dtype == "int32"

        # Test decoding
        # Note: BOS/EOS tokens are added, so we decode the relevant part
        token_list = encoded.numpy().flatten().tolist()
        # Find the start and end of the actual content
        try:
            start_idx = token_list.index(1) + 1  # 1 is BOS
            end_idx = token_list.index(2)       # 2 is EOS
            decoded_text = model.decode_tokens(np.array(token_list[start_idx:end_idx]))
            assert decoded_text == text
        except ValueError:
            pytest.fail("BOS or EOS token not found in encoded output")


class TestModernBertBLTFactoryFunctions:
    """Test suite for the factory and builder functions for ModernBertBLT."""

    def test_create_modern_bert_blt_base_config(self):
        """Tests the configuration factory for ModernBertBLT-base."""
        config = create_modern_bert_blt_base_config()
        assert isinstance(config, dict)
        assert config['hidden_size'] == 768
        assert config['num_layers'] == 12
        assert config['num_heads'] == 12
        assert config['vocab_size'] == 260
        assert config['use_hash_embeddings'] is True
        model = ModernBertBLT(**config)
        assert model.hidden_size == 768

    @pytest.fixture
    def small_config(self) -> Dict[str, Any]:
        """Provides a minimal config for functional model tests."""
        return {
            'vocab_size': 260,
            'hidden_size': 32,
            'num_layers': 2,
            'num_heads': 2,
            'intermediate_size': 64,
            'use_hash_embeddings': False, # Disable for simplicity
        }

    def test_create_modern_bert_blt_for_classification(self, small_config):
        """Tests the builder for a classification model, including serialization."""
        num_labels = 5
        model = create_modern_bert_blt_for_classification(small_config, num_labels=num_labels)

        assert isinstance(model, keras.Model)
        # BLT models expect 2 inputs: input_ids and attention_mask
        assert len(model.inputs) == 2

        batch_size, seq_len = 2, 16
        inputs = [
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
        ]
        logits = model(inputs)
        assert logits.shape == (batch_size, num_labels)

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

    def test_create_modern_bert_blt_for_sequence_output(self, small_config):
        """Tests the builder for a sequence output model, including serialization."""
        model = create_modern_bert_blt_for_sequence_output(small_config)

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 2

        batch_size, seq_len = 2, 16
        inputs = [
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            keras.ops.zeros((batch_size, seq_len), dtype='int32'),
        ]
        sequence_output = model(inputs)
        assert sequence_output.shape == (batch_size, seq_len, small_config['hidden_size'])

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
    pytest.main([__file__, "-v", "--tb=short"])