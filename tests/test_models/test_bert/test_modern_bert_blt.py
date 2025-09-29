import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

# System imports
from dl_techniques.models.bert.modern_bert_blt import (
    ModernBertBLT,
    create_modern_bert_blt_with_head,
)
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType


class TestModernBertBLT:
    """
    Comprehensive test suite for the ModernBertBLT foundation model.
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
            'rope_theta': 10000.0,
            'max_seq_len': 512,
            'global_attention_interval': 2,
            # ## FIX: WindowAttention requires seq_len == window_size**2
            # ## Since seq_len is 64, window_size must be 8.
            'local_attention_window_size': 8,
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
        assert not hasattr(model, 'pooler'), "Foundation model should not have a pooler."

        # Verify the global/local attention configuration logic
        for i, layer in enumerate(model.encoder_layers):
            is_global_expected = (i + 1) % model_config['global_attention_interval'] == 0
            if is_global_expected:
                assert layer.attention_type == "multi_head"
            else:
                assert layer.attention_type == "window"
                assert layer.attention_args['window_size'] == model_config['local_attention_window_size']


    def test_forward_pass_output_shape_and_type(self, model_config, sample_inputs):
        """Tests the forward pass, ensuring correct output shape and dictionary format."""
        model = ModernBertBLT(**model_config)
        batch_size, seq_len, hidden_size = 2, 64, model_config['hidden_size']

        outputs = model(sample_inputs)
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert "attention_mask" in outputs
        assert outputs["last_hidden_state"].shape == (batch_size, seq_len, hidden_size)
        assert outputs["attention_mask"].shape == (batch_size, seq_len)

    def test_forward_pass_with_tensor_input(self, model_config, sample_inputs):
        """Tests that the model handles positional tensor inputs correctly."""
        model = ModernBertBLT(**model_config)
        outputs = model(
            sample_inputs["input_ids"],
            attention_mask=sample_inputs["attention_mask"]
        )
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (2, 64, model_config['hidden_size'])

    def test_serialization_cycle(self, model_config, sample_inputs):
        """CRITICAL TEST: Ensures the model can be saved and reloaded correctly."""
        model = ModernBertBLT(**model_config)
        original_outputs = model(sample_inputs, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)

            loaded_model = keras.models.load_model(filepath)
            loaded_outputs = loaded_model(sample_inputs, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs["last_hidden_state"]),
                keras.ops.convert_to_numpy(loaded_outputs["last_hidden_state"]),
                rtol=1e-6, atol=1e-6
            )

    def test_config_completeness(self, model_config):
        """Tests that get_config returns all __init__ parameters for reconstruction."""
        model = ModernBertBLT(**model_config)
        config = model.get_config()

        for key, value in model_config.items():
            assert key in config, f"Missing key '{key}' in get_config()"
            assert config[key] == value, f"Config mismatch for key '{key}'"
        assert "add_pooling_layer" not in config

    def test_gradients_flow(self, model_config, sample_inputs):
        """Tests that gradients can be computed through the model."""
        model = ModernBertBLT(**model_config)
        with tf.GradientTape() as tape:
            outputs = model(sample_inputs)
            loss = keras.ops.mean(outputs["last_hidden_state"])
        gradients = tape.gradient(loss, model.trainable_variables)
        assert len(gradients) > 0, "No gradients were computed."
        assert all(g is not None for g in gradients), "Some gradients are None."

    def test_encode_decode_cycle(self, model_config):
        """Tests the model's text encoding and decoding convenience methods."""
        model = ModernBertBLT(**model_config)
        text = "Hello, world! This is a test."

        encoded = model.encode_text(text, max_length=32, add_special_tokens=True)
        assert encoded.shape == (1, 32)
        assert encoded.dtype == "int32"

        token_list = encoded.numpy().flatten().tolist()
        try:
            start_idx = token_list.index(1) + 1  # 1 is BOS
            end_idx = token_list.index(2)  # 2 is EOS
            decoded_text = model.decode_tokens(np.array(token_list[start_idx:end_idx]))
            assert decoded_text == text
        except ValueError:
            pytest.fail("BOS or EOS token not found in encoded output")


## FIX: Enable integration tests to ensure model-head construction and serialization work.
class TestModernBertBLTIntegration:
    """Test suite for factory functions and model integration."""

    @pytest.fixture
    def test_variant_name(self, monkeypatch) -> str:
        """Adds a small, temporary model variant for fast testing."""
        test_config = {
            'vocab_size': 260,
            'hidden_size': 32,
            'num_layers': 2,
            'num_heads': 2,
            'intermediate_size': 64,
            'use_hash_embeddings': False,
            'max_seq_len': 128,
            "local_attention_window_size": 8, # 8*8 = 64
            "description": "Test variant"
        }
        monkeypatch.setitem(ModernBertBLT.MODEL_VARIANTS, "test", test_config)
        return "test"

    def test_from_variant(self):
        """Tests creating a model from a predefined variant."""
        model_base = ModernBertBLT.from_variant("base")
        assert model_base.hidden_size == 768
        assert model_base.num_layers == 12

        model_large_override = ModernBertBLT.from_variant("large", num_layers=2)
        assert model_large_override.hidden_size == 1024
        assert model_large_override.num_layers == 2

        with pytest.raises(ValueError, match="Unknown variant"):
            ModernBertBLT.from_variant("non_existent_variant")

    def test_create_with_classification_head(self, test_variant_name):
        """Tests the factory for a classification model, including serialization."""
        num_labels = 5
        task_config = NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=num_labels,
        )
        model = create_modern_bert_blt_with_head(
            bert_variant=test_variant_name, task_config=task_config
        )

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 2
        # FIX: The static model.output_shape attribute is unreliable for dict outputs.
        # The dynamic check on the forward pass output below is sufficient and more robust.
        # assert model.output_shape["logits"][1] == num_labels

        batch_size, seq_len = 2, 64 # seq_len must be 64 for window_size=8
        inputs = {
            "input_ids": keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            "attention_mask": keras.ops.ones((batch_size, seq_len), dtype='int32'),
        }
        logits = model(inputs)["logits"]
        assert logits.shape == (batch_size, num_labels)

        # Test serialization cycle
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'classification_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(inputs)["logits"]
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(logits),
                keras.ops.convert_to_numpy(loaded_pred),
                rtol=1e-6, atol=1e-6
            )

    def test_create_with_token_classification_head(self, test_variant_name):
        """Tests the factory for a token classification model."""
        num_labels = 9
        task_config = NLPTaskConfig(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=num_labels,
        )
        model = create_modern_bert_blt_with_head(
            bert_variant=test_variant_name, task_config=task_config
        )

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 2

        batch_size, seq_len = 2, 64 # seq_len must be 64 for window_size=8
        inputs = {
            "input_ids": keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            "attention_mask": keras.ops.ones((batch_size, seq_len), dtype='int32'),
        }
        logits = model(inputs)["logits"]
        assert logits.shape == (batch_size, seq_len, num_labels)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])