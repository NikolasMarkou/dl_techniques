import pytest
import tempfile
import os
from typing import Any, Dict

import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.bert.modern_bert import (
    ModernBERT,
    create_modern_bert_with_head,
)
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType


class TestModernBERT:
    """
    Comprehensive test suite for the refactored ModernBERT model.
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
            'global_attention_interval': 2,
            'local_attention_window_size': 8,
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
        model = ModernBERT(**model_config)
        assert not model.built
        assert isinstance(model.final_norm, keras.layers.LayerNormalization)
        assert len(model.encoder_layers) == model_config['num_layers']
        assert not hasattr(model, 'pooler')
        for i, layer in enumerate(model.encoder_layers):
            is_global_expected = (i + 1) % model_config['global_attention_interval'] == 0
            expected_attn_type = "multi_head" if is_global_expected else "window"
            assert layer.attention_type == expected_attn_type
            if not is_global_expected:
                assert layer.attention_args['window_size'] == model_config['local_attention_window_size']

    def test_forward_pass_output_format(self, model_config, sample_inputs):
        model = ModernBERT(**model_config)
        batch_size, seq_len, hidden_size = 2, 64, model_config['hidden_size']
        outputs_dict = model(sample_inputs)
        assert isinstance(outputs_dict, dict)
        assert "last_hidden_state" in outputs_dict
        assert "attention_mask" in outputs_dict
        assert "pooler_output" not in outputs_dict
        assert outputs_dict["last_hidden_state"].shape == (batch_size, seq_len, hidden_size)
        assert outputs_dict["attention_mask"].shape == sample_inputs["attention_mask"].shape

    def test_forward_pass_input_types(self, model_config, sample_inputs):
        model = ModernBERT(**model_config)
        outputs = model(
            sample_inputs["input_ids"],
            attention_mask=sample_inputs["attention_mask"],
            token_type_ids=sample_inputs["token_type_ids"]
        )
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (2, 64, model_config['hidden_size'])

    def test_serialization_cycle(self, model_config, sample_inputs):
        model = ModernBERT(**model_config)
        original_outputs = model(sample_inputs, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_outputs = loaded_model(sample_inputs, training=False)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs["last_hidden_state"]),
                keras.ops.convert_to_numpy(loaded_outputs["last_hidden_state"]),
                rtol=1e-6, atol=1e-6,
                err_msg="Sequence output differs after serialization"
            )

    def test_config_completeness(self, model_config):
        model = ModernBERT(**model_config)
        config = model.get_config()
        for key, value in model_config.items():
            assert key in config
            assert config[key] == value

    def test_gradients_flow(self, model_config, sample_inputs):
        model = ModernBERT(**model_config)
        with tf.GradientTape() as tape:
            outputs = model(sample_inputs)
            loss = keras.ops.mean(outputs["last_hidden_state"])
        gradients = tape.gradient(loss, model.trainable_variables)
        assert len(gradients) > 0
        assert all(g is not None for g in gradients)

    @pytest.mark.parametrize("training", [True, False])
    def test_training_modes(self, model_config, sample_inputs, training):
        model = ModernBERT(**model_config)
        outputs = model(sample_inputs, training=training)
        assert outputs["last_hidden_state"].shape[0] == sample_inputs["input_ids"].shape[0]

    def test_missing_input_ids_error(self, model_config):
        model = ModernBERT(**model_config)
        with pytest.raises(ValueError, match="input_ids must be provided"):
            model({})


class TestModernBERTBuilders:
    """Test suite for the builder functions for ModernBERT."""

    def test_create_modern_bert_with_head(self):
        """Tests the builder for a complete model with a task-specific head."""
        task_config = NLPTaskConfig(
            name="test_qa",
            task_type=NLPTaskType.QUESTION_ANSWERING,
        )

        model = create_modern_bert_with_head(
            bert_variant="base",
            task_config=task_config,
            bert_config_overrides={
                'num_layers': 2, 'hidden_size': 64, 'num_heads': 2, 'intermediate_size': 128,
                'local_attention_window_size': 4
            }
        )

        assert isinstance(model, keras.Model)
        assert len(model.inputs) == 3

        batch_size, seq_len = 2, 16
        inputs = {
            "input_ids": keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            "attention_mask": keras.ops.zeros((batch_size, seq_len), dtype='int32'),
            "token_type_ids": keras.ops.zeros((batch_size, seq_len), dtype='int32'),
        }
        qa_outputs = model(inputs)

        # --- FIX: Assert on the actual output of the QuestionAnsweringHead ---
        assert isinstance(qa_outputs, dict)
        assert "start_logits" in qa_outputs
        assert "end_logits" in qa_outputs
        assert qa_outputs["start_logits"].shape == (batch_size, seq_len)
        assert qa_outputs["end_logits"].shape == (batch_size, seq_len)

        # Test serialization cycle
        original_pred = model(inputs, training=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'qa_model.keras')
            model.save(filepath)
            loaded_model = keras.models.load_model(filepath)
            loaded_pred = loaded_model(inputs, training=False)

            # --- FIX: Assert on the individual tensors within the output dictionary ---
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred["start_logits"]),
                keras.ops.convert_to_numpy(loaded_pred["start_logits"]),
                rtol=1e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_pred["end_logits"]),
                keras.ops.convert_to_numpy(loaded_pred["end_logits"]),
                rtol=1e-6, atol=1e-6
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])