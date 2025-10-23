"""
Comprehensive pytest test suite for the ModernBERT foundation model.

This module provides extensive testing for the ModernBERT implementation including:
- Foundation model initialization and parameter validation for modern features.
- Architecture building (Pre-LN, GeGLU, Hybrid Attention) and consistent output shape.
- Forward pass functionality with a standardized dictionary output.
- Model variant creation and configuration.
- Serialization and deserialization of the pure encoder.
- Error handling and edge cases.
- Factory function testing for integrating the encoder with task heads.
- End-to-end integration testing for gradient flow and training.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.bert.modern_bert import ModernBERT, create_modern_bert_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType
from dl_techniques.layers.embedding.modern_bert_embeddings import ModernBertEmbeddings


# Define a small window size for all tests to avoid OOM errors.
TEST_WINDOW_SIZE = 16


class TestModernBERTModelInitialization:
    """Test ModernBERT model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic ModernBERT model initialization as a pure encoder."""
        model = ModernBERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            intermediate_size=384,
            global_attention_interval=2,
            local_attention_window_size=TEST_WINDOW_SIZE, # FIX: Use small window size
            use_bias=False
        )

        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert model.use_bias is False
        assert model.global_attention_interval == 2
        assert not model.built

        # Components should be created in __init__
        assert isinstance(model.embeddings, ModernBertEmbeddings)
        assert len(model.encoder_layers) == 6
        assert model.final_norm is not None

        # But should not be built yet
        assert not model.embeddings.built
        for layer in model.encoder_layers:
            assert not layer.built
        assert not model.final_norm.built

    def test_parameter_validation(self):
        """Test ModernBERT parameter validation for invalid values."""
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            ModernBERT(vocab_size=1000, hidden_size=100, num_heads=12, num_layers=4)

        with pytest.raises(ValueError, match="Sizes and layer/head counts must be positive"):
            ModernBERT(vocab_size=1000, hidden_size=-100, num_layers=4, num_heads=8)

        with pytest.raises(ValueError, match="hidden_dropout_prob must be between 0 and 1"):
            ModernBERT(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                hidden_dropout_prob=1.5
            )

        with pytest.raises(ValueError, match="global_attention_interval must be positive"):
            ModernBERT(global_attention_interval=0)

    def test_initialization_with_custom_config(self):
        """Test ModernBERT model initialization with custom configuration."""
        model = ModernBERT(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=1024,
            hidden_dropout_prob=0.2,
            global_attention_interval=4,
            local_attention_window_size=TEST_WINDOW_SIZE, # FIX: Use small window size
        )
        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert model.num_layers == 8
        assert model.hidden_dropout_prob == 0.2
        assert model.global_attention_interval == 4


class TestModernBERTModelVariants:
    """Test ModernBERT model variants and factory methods."""

    def test_bert_base_variant(self):
        # FIX: Override default large window size to prevent OOM
        model = ModernBERT.from_variant("base", local_attention_window_size=TEST_WINDOW_SIZE)
        assert model.hidden_size == 768
        assert model.num_layers == 22
        assert model.num_heads == 12
        assert model.intermediate_size == 1152
        assert model.use_bias is False

    def test_bert_large_variant(self):
        # FIX: Override default large window size to prevent OOM
        model = ModernBERT.from_variant("large", local_attention_window_size=TEST_WINDOW_SIZE)
        assert model.hidden_size == 1024
        assert model.num_layers == 28
        assert model.num_heads == 16
        assert model.intermediate_size == 2624

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            ModernBERT.from_variant("invalid")

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameters."""
        model = ModernBERT.from_variant(
            "base",
            use_bias=True,
            num_layers=10,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        assert model.hidden_size == 768
        assert model.num_layers == 10
        assert model.use_bias is True


class TestModernBERTModelBuilding:
    """Test ModernBERT model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 1000, "hidden_size": 256, "num_layers": 4,
            "num_heads": 8, "intermediate_size": 384, "global_attention_interval": 2,
            "local_attention_window_size": TEST_WINDOW_SIZE # FIX: Use small window size
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = ModernBERT(**basic_config)
        batch_size, seq_length = 2, 32 # Seq length can be larger now
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), maxval=basic_config['vocab_size']),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)
        assert model.built
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, basic_config['hidden_size']
        )
        assert "attention_mask" in outputs
        assert outputs["attention_mask"] is None  # Since it wasn't provided

    def test_transformer_layers_configuration(self, basic_config):
        model = ModernBERT(**basic_config)
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=basic_config['vocab_size']), dtype='int32'
        )
        _ = model(input_ids, training=False)
        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config['hidden_size']
            assert layer.num_heads == basic_config['num_heads']
            assert layer.name == f"encoder_layer_{i}"
            assert layer.normalization_position == 'pre'
            assert layer.ffn_type == 'geglu'


class TestModernBERTModelForwardPass:
    """Test ModernBERT model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> ModernBERT:
        """Create a built ModernBERT model for testing."""
        model = ModernBERT(
            vocab_size=1000, hidden_size=256, num_layers=3, num_heads=8,
            intermediate_size=384,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        sample_input = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=1000), dtype='int32'
        )
        _ = model(sample_input, training=False)
        return model

    def test_forward_pass_input_ids_only(self, built_model):
        batch_size, seq_length = 4, 32
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), maxval=built_model.vocab_size),
            dtype='int32'
        )
        outputs = built_model(input_ids, training=False)
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.hidden_size
        )

    def test_forward_pass_dict_input(self, built_model):
        batch_size, seq_length = 3, 24
        inputs = {
            'input_ids': keras.ops.cast(
                keras.random.uniform((batch_size, seq_length), maxval=built_model.vocab_size),
                dtype='int32'
            ),
            'attention_mask': keras.ops.ones((batch_size, seq_length), dtype='int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), dtype='int32')
        }
        outputs = built_model(inputs, training=False)
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.hidden_size
        )
        assert "attention_mask" in outputs
        assert outputs["attention_mask"] is not None

    def test_invalid_dict_input(self, built_model):
        inputs = {'invalid_key': keras.ops.ones((2, 16), dtype='int32')}
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model(inputs, training=False)


class TestModernBERTModelSerialization:
    """Test ModernBERT model serialization and deserialization."""

    def test_config_serialization(self):
        model = ModernBERT(
            vocab_size=25000, hidden_size=512, num_layers=8, num_heads=8,
            global_attention_interval=3, use_bias=False,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        model_config = model.get_config()
        assert model_config['vocab_size'] == 25000
        assert model_config['hidden_size'] == 512
        assert model_config['global_attention_interval'] == 3
        assert model_config['use_bias'] is False

    def test_model_from_config(self):
        original_model = ModernBERT(
            vocab_size=1000, hidden_size=256, num_layers=4,
            num_heads=8, global_attention_interval=2,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        config = original_model.get_config()
        new_model = ModernBERT.from_config(config)
        assert new_model.hidden_size == original_model.hidden_size
        assert new_model.global_attention_interval == original_model.global_attention_interval

    def test_model_save_load(self):
        model = ModernBERT(
            vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), dtype='int32')
        original_outputs = model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_modern_bert.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs['last_hidden_state']),
                keras.ops.convert_to_numpy(loaded_outputs['last_hidden_state']),
                rtol=1e-5, atol=1e-6, err_msg="Hidden states should match"
            )


class TestModernBERTEdgeCases:
    """Test ModernBERT model edge cases."""

    def test_minimum_sequence_length(self):
        model = ModernBERT(
            vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        input_ids = keras.ops.cast([[42]], dtype='int32')
        outputs = model(input_ids, training=False)
        assert outputs['last_hidden_state'].shape == (1, 1, 128)


class TestModernBERTIntegrationFactory:
    """Test the integration factory `create_modern_bert_with_head`."""

    def test_create_with_classification_head(self):
        """Test creating a ModernBERT model for sequence classification."""
        task_config = NLPTaskConfig(
            name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3
        )
        # FIX: Override default tiny window size to ensure it passes on all systems
        model = create_modern_bert_with_head(
            "tiny",
            task_config,
            bert_config_overrides={"local_attention_window_size": TEST_WINDOW_SIZE}
        )
        assert isinstance(model, keras.Model)

        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((2, 32), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((2, 32), 'int32'),
            'token_type_ids': keras.ops.zeros((2, 32), 'int32'),
        }
        outputs = model(inputs, training=False)
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert outputs["logits"].shape == (2, 3)

    def test_create_with_token_classification_head(self):
        """Test creating a ModernBERT model for token classification."""
        task_config = NLPTaskConfig(
            name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9
        )
        # FIX: Override default tiny window size to ensure it passes on all systems
        model = create_modern_bert_with_head(
            "tiny",
            task_config,
            bert_config_overrides={"local_attention_window_size": TEST_WINDOW_SIZE}
        )
        assert isinstance(model, keras.Model)

        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((2, 32), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((2, 32), 'int32'),
            'token_type_ids': keras.ops.zeros((2, 32), 'int32'),
        }
        outputs = model(inputs, training=False)
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 32, 9)


class TestModernBERTIntegration:
    """Integration tests for the complete model (encoder + head)."""

    @pytest.fixture
    def classification_model(self) -> keras.Model:
        """Create a classification model for integration tests."""
        task_config = NLPTaskConfig(
            name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
        )
        # FIX: Add small window size to overrides
        return create_modern_bert_with_head(
            "tiny",
            task_config,
            bert_config_overrides={
                "hidden_size": 64,
                "intermediate_size": 96,
                "local_attention_window_size": TEST_WINDOW_SIZE
            }
        )

    def test_gradient_flow_integration(self, classification_model):
        """Test that gradients flow through the entire integrated model."""
        batch_size, seq_length = 2, 16
        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((batch_size, seq_length), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((batch_size, seq_length), 'int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), 'int32'),
        }

        with tf.GradientTape() as tape:
            outputs = classification_model(inputs, training=True)
            logits = outputs['logits']
            targets = keras.ops.one_hot(keras.ops.array([0, 2]), 3)
            loss = keras.ops.mean(keras.losses.categorical_crossentropy(targets, logits))

        gradients = tape.gradient(loss, classification_model.trainable_weights)
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0
        assert len(non_none_grads) == len(classification_model.trainable_weights)


class TestModernBERTAdvancedFeatures:
    """Test advanced ModernBERT features."""

    def test_hybrid_attention_mechanism(self):
        """Verify that attention types alternate correctly."""
        interval = 3
        model = ModernBERT(
            vocab_size=1000, hidden_size=128, num_layers=7, num_heads=8,
            global_attention_interval=interval,
            local_attention_window_size=TEST_WINDOW_SIZE # FIX: Use small window size
        )
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
        _ = model(input_ids)  # Build the model

        for i, layer in enumerate(model.encoder_layers):
            is_global = (i + 1) % interval == 0
            expected_type = "multi_head" if is_global else "window"
            assert layer.attention_type == expected_type, f"Layer {i} has wrong attention type"

    def test_pre_ln_and_geglu_config(self):
        """Verify that layers are configured for Pre-LN and GeGLU."""
        model = ModernBERT.from_variant("tiny", local_attention_window_size=TEST_WINDOW_SIZE)
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
        _ = model(input_ids)  # Build the model

        # Check a sample encoder layer
        sample_layer = model.encoder_layers[0]
        assert sample_layer.normalization_position == 'pre'
        assert sample_layer.ffn_type == 'geglu'

    def test_model_summary(self):
        """Test that model summary works without errors."""
        model = ModernBERT.from_variant("tiny", local_attention_window_size=TEST_WINDOW_SIZE)
        input_ids = keras.ops.cast(keras.random.uniform((1, 16), maxval=1000), 'int32')
        _ = model(input_ids, training=False)
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary raised an exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])