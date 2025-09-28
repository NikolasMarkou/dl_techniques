"""
Comprehensive pytest test suite for the refactored BERT foundation model.

This module provides extensive testing for the decoupled BERT implementation including:
- Foundation model initialization and parameter validation.
- Architecture building and consistent output shape.
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

from dl_techniques.models.bert.foundation_model import BERT, create_bert_with_head
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType


class TestBERTModelInitialization:
    """Test BERT model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BERT model initialization as a pure encoder."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            intermediate_size=1024
        )

        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert not hasattr(model, 'add_pooling_layer')
        assert not hasattr(model, 'pooler')
        assert not model.built

        # Components should be created in __init__
        assert model.embeddings is not None
        assert len(model.encoder_layers) == 6

        # But should not be built yet
        assert not model.embeddings.built
        for layer in model.encoder_layers:
            assert not layer.built

    def test_parameter_validation(self):
        """Test BERT parameter validation for invalid values."""
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            BERT(vocab_size=1000, hidden_size=100, num_heads=12, num_layers=4)

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            BERT(vocab_size=1000, hidden_size=-100, num_layers=4, num_heads=8)

        with pytest.raises(ValueError, match="hidden_dropout_prob must be between"):
            BERT(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                hidden_dropout_prob=1.5
            )

    def test_initialization_with_custom_config(self):
        """Test BERT model initialization with custom configuration."""
        model = BERT(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=2048,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.1,
            normalization_type="rms_norm",
            normalization_position="pre"
        )
        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert model.num_layers == 8
        assert model.hidden_dropout_prob == 0.2
        assert model.normalization_type == "rms_norm"


class TestBERTModelVariants:
    """Test BERT model variants and factory methods."""

    def test_bert_base_variant(self):
        model = BERT.from_variant("base")
        assert model.hidden_size == 768
        assert model.num_layers == 12
        assert model.num_heads == 12

    def test_bert_large_variant(self):
        model = BERT.from_variant("large")
        assert model.hidden_size == 1024
        assert model.num_layers == 24
        assert model.num_heads == 16

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            BERT.from_variant("invalid")

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameters."""
        model = BERT.from_variant("base", normalization_type="rms_norm")
        assert model.hidden_size == 768
        assert model.num_layers == 12
        assert model.normalization_type == "rms_norm"
        assert not hasattr(model, 'add_pooling_layer')


class TestBERTModelBuilding:
    """Test BERT model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 1000, "hidden_size": 256, "num_layers": 4,
            "num_heads": 8, "intermediate_size": 1024, "max_position_embeddings": 128
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = BERT(**basic_config)
        batch_size, seq_length = 2, 32
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

    def test_transformer_layers_configuration(self, basic_config):
        model = BERT(**basic_config)
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=basic_config['vocab_size']), dtype='int32'
        )
        _ = model(input_ids, training=False)
        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config['hidden_size']
            assert layer.num_heads == basic_config['num_heads']
            assert layer.name == f"encoder_layer_{i}"


class TestBERTModelForwardPass:
    """Test BERT model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> BERT:
        """Create a built BERT model for testing."""
        model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=3, num_heads=8,
            intermediate_size=1024, max_position_embeddings=128
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

    def test_forward_pass_with_attention_mask(self, built_model):
        batch_size, seq_length = 2, 16
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), maxval=built_model.vocab_size),
            dtype='int32'
        )
        attention_mask = keras.ops.ones((batch_size, seq_length), dtype='int32')
        attention_mask = keras.ops.slice_update(
            attention_mask, [0, seq_length - 4], keras.ops.zeros((batch_size, 4), dtype='int32')
        )
        outputs = built_model(input_ids, attention_mask=attention_mask, training=False)
        assert outputs["last_hidden_state"].shape == (
            batch_size, seq_length, built_model.hidden_size
        )

    def test_invalid_dict_input(self, built_model):
        inputs = {'invalid_key': keras.ops.ones((2, 16), dtype='int32')}
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model(inputs, training=False)


class TestBERTModelSerialization:
    """Test BERT model serialization and deserialization."""

    def test_config_serialization(self):
        model = BERT(
            vocab_size=25000, hidden_size=512, num_layers=8, num_heads=8,
            normalization_type="rms_norm"
        )
        model_config = model.get_config()
        assert model_config['vocab_size'] == 25000
        assert model_config['hidden_size'] == 512
        assert not hasattr(model_config, 'add_pooling_layer')

    def test_model_from_config(self):
        original_model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=4,
            num_heads=8, normalization_type="rms_norm"
        )
        config = original_model.get_config()
        new_model = BERT.from_config(config)
        assert new_model.hidden_size == original_model.hidden_size
        assert new_model.normalization_type == original_model.normalization_type

    def test_model_save_load(self):
        model = BERT(vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8)
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), dtype='int32')
        original_outputs = model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_bert.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs['last_hidden_state']),
                keras.ops.convert_to_numpy(loaded_outputs['last_hidden_state']),
                rtol=1e-5, atol=1e-6, err_msg="Hidden states should match"
            )


class TestBERTEdgeCases:
    """Test BERT model edge cases."""

    def test_minimum_sequence_length(self):
        model = BERT(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        input_ids = keras.ops.cast([[42]], dtype='int32')
        outputs = model(input_ids, training=False)
        assert outputs['last_hidden_state'].shape == (1, 1, 128)

    def test_long_sequence(self):
        model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8,
            max_position_embeddings=128
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 128), maxval=1000), dtype='int32'
        )
        outputs = model(input_ids, training=False)
        assert outputs['last_hidden_state'].shape == (1, 128, 256)


class TestBERTIntegrationFactory:
    """Test the integration factory `create_bert_with_head`."""

    def test_create_with_classification_head(self):
        """Test creating a BERT model for sequence classification."""
        task_config = NLPTaskConfig(
            name="sentiment", task_type=NLPTaskType.SENTIMENT_ANALYSIS, num_classes=3
        )
        model = create_bert_with_head("tiny", task_config)
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
        """Test creating a BERT model for token classification."""
        task_config = NLPTaskConfig(
            name="ner", task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION, num_classes=9
        )
        model = create_bert_with_head("tiny", task_config)
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


class TestBERTIntegration:
    """Integration tests for the complete model (encoder + head)."""

    @pytest.fixture
    def classification_model(self) -> keras.Model:
        """Create a classification model for integration tests."""
        task_config = NLPTaskConfig(
            name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
        )
        return create_bert_with_head(
            "tiny",
            task_config,
            bert_config_overrides={"hidden_size": 64, "intermediate_size": 256}
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
        grad_norms = [keras.ops.sqrt(keras.ops.sum(keras.ops.square(g))) for g in non_none_grads]
        assert all(norm > 0.0 for norm in grad_norms)

    def test_training_integration(self, classification_model):
        """Test the integrated model in a minimal training loop."""
        optimizer = keras.optimizers.Adam(learning_rate=1e-6)
        batch_size, seq_length = 4, 16
        inputs = {
            'input_ids': keras.ops.cast(keras.random.uniform((batch_size, seq_length), maxval=1000), 'int32'),
            'attention_mask': keras.ops.ones((batch_size, seq_length), 'int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), 'int32'),
        }
        labels = keras.ops.cast(keras.random.uniform((batch_size,), maxval=3), 'int32')

        initial_loss = None
        for step in range(10):
            with tf.GradientTape() as tape:
                outputs = classification_model(inputs, training=True)
                loss = keras.ops.mean(
                    keras.losses.sparse_categorical_crossentropy(labels, outputs['logits'])
                )
            if initial_loss is None:
                initial_loss = loss
            gradients = tape.gradient(loss, classification_model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, classification_model.trainable_weights))

        # Loss should decrease (at least not increase significantly)
        assert loss <= initial_loss + 0.1


class TestBERTAdvancedFeatures:
    """Test advanced BERT features from dl-techniques framework."""

    def test_different_normalization_types(self):
        base_config = {"vocab_size": 1000, "hidden_size": 128, "num_layers": 2, "num_heads": 8}
        for norm_type in ["layer_norm", "rms_norm"]:
            model = BERT(**base_config, normalization_type=norm_type)
            input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
            output = model(input_ids, training=False)
            assert output['last_hidden_state'].shape == (2, 16, 128)

    def test_stochastic_depth(self):
        model = BERT(
            vocab_size=1000, hidden_size=256, num_layers=4, num_heads=8,
            use_stochastic_depth=True, stochastic_depth_rate=0.1
        )
        input_ids = keras.ops.cast(keras.random.uniform((2, 16), maxval=1000), 'int32')
        output_train = model(input_ids, training=True)
        output_inference = model(input_ids, training=False)
        assert output_train['last_hidden_state'].shape == (2, 16, 256)
        assert output_inference['last_hidden_state'].shape == (2, 16, 256)

    def test_model_summary(self):
        """Test that model summary works without errors."""
        model = BERT.from_variant("tiny")
        input_ids = keras.ops.cast(keras.random.uniform((1, 16), maxval=1000), 'int32')
        _ = model(input_ids, training=False)
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary raised an exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])