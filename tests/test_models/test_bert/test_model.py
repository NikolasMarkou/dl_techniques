"""
Comprehensive pytest test suite for BERT (Bidirectional Encoder Representations from Transformers) model.

This module provides extensive testing for the BERT implementation including:
- Model initialization and parameter validation
- Architecture building and shape consistency
- Forward pass functionality with different input formats
- Model variant creation and configuration
- Serialization and deserialization
- Error handling and edge cases
- Factory function testing for classification and sequence output models
- Integration testing
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
from typing import Dict, Any

from dl_techniques.models.bert.model import (
    BERT,
    create_bert_for_classification,
    create_bert_for_sequence_output,
    create_bert
)


class TestBERTModelInitialization:
    """Test BERT model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic BERT model initialization."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            intermediate_size=1024
        )

        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert model.add_pooling_layer is True
        assert not model.built

        # Components should be created in __init__
        assert model.embeddings is not None
        assert len(model.encoder_layers) == 6
        assert model.pooler is not None

        # But should not be built yet
        assert not model.embeddings.built
        for layer in model.encoder_layers:
            assert not layer.built

    def test_initialization_without_pooling(self):
        """Test BERT model initialization without pooling layer."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            add_pooling_layer=False
        )

        assert model.add_pooling_layer is False
        assert model.pooler is None

    def test_parameter_validation(self):
        """Test BERT parameter validation for invalid values."""
        # Test invalid hidden_size/num_heads combination
        with pytest.raises(ValueError, match="hidden_size.*must be divisible by num_heads"):
            BERT(
                vocab_size=1000,
                hidden_size=100,
                num_heads=12,
                num_layers=4
            )

        # Test negative hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            BERT(
                vocab_size=1000,
                hidden_size=-100,
                num_layers=4,
                num_heads=8
            )

        # Test invalid dropout rates
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
        """Test creating BERT-base variant."""
        model = BERT.from_variant("base")

        assert model.vocab_size == 30522
        assert model.hidden_size == 768
        assert model.num_layers == 12
        assert model.num_heads == 12
        assert model.intermediate_size == 3072

    def test_bert_large_variant(self):
        """Test creating BERT-large variant."""
        model = BERT.from_variant("large")

        assert model.vocab_size == 30522
        assert model.hidden_size == 1024
        assert model.num_layers == 24
        assert model.num_heads == 16
        assert model.intermediate_size == 4096

    def test_bert_small_variant(self):
        """Test creating BERT-small variant."""
        model = BERT.from_variant("small")

        assert model.vocab_size == 30522
        assert model.hidden_size == 512
        assert model.num_layers == 6
        assert model.num_heads == 8
        assert model.intermediate_size == 2048

    def test_bert_tiny_variant(self):
        """Test creating BERT-tiny variant."""
        model = BERT.from_variant("tiny")

        assert model.vocab_size == 30522
        assert model.hidden_size == 256
        assert model.num_layers == 4
        assert model.num_heads == 4
        assert model.intermediate_size == 1024

    def test_invalid_variant(self):
        """Test error handling for invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            BERT.from_variant("invalid")

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameters."""
        model = BERT.from_variant(
            "base",
            add_pooling_layer=False,
            normalization_type="rms_norm"
        )

        # Should have base variant dimensions
        assert model.hidden_size == 768
        assert model.num_layers == 12
        # But custom parameters should override
        assert model.add_pooling_layer is False
        assert model.normalization_type == "rms_norm"


class TestBERTModelBuilding:
    """Test BERT model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Create a basic BERT config for testing."""
        return {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "intermediate_size": 1024,
            "max_position_embeddings": 128
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality."""
        model = BERT(**basic_config)
        batch_size, seq_length = 2, 32
        input_ids = keras.ops.cast(
            keras.random.uniform(
                (batch_size, seq_length),
                minval=1,
                maxval=basic_config['vocab_size']
            ),
            dtype='int32'
        )

        outputs = model(input_ids, training=False)

        assert model.built
        assert model.embeddings.built
        assert len(model.encoder_layers) == basic_config['num_layers']
        for layer in model.encoder_layers:
            assert layer.built

        sequence_output, pooled_output = outputs
        assert sequence_output.shape == (batch_size, seq_length, basic_config['hidden_size'])
        assert pooled_output.shape == (batch_size, basic_config['hidden_size'])

    def test_build_without_pooling_layer(self, basic_config):
        """Test building BERT model without pooling layer."""
        model = BERT(**basic_config, add_pooling_layer=False)
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=basic_config['vocab_size']),
            dtype='int32'
        )
        output = model(input_ids, training=False)
        assert model.built
        assert model.pooler is None
        assert output.shape == (2, 32, basic_config['hidden_size'])

    def test_transformer_layers_configuration(self, basic_config):
        """Test that TransformerLayers are configured correctly."""
        model = BERT(**basic_config)
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 16), minval=1, maxval=basic_config['vocab_size']),
            dtype='int32'
        )
        _ = model(input_ids, training=False)

        for i, layer in enumerate(model.encoder_layers):
            assert layer.hidden_size == basic_config['hidden_size']
            assert layer.num_heads == basic_config['num_heads']
            assert layer.intermediate_size == basic_config['intermediate_size']
            assert layer.name == f"encoder_layer_{i}"


class TestBERTModelForwardPass:
    """Test BERT model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> BERT:
        """Create a built BERT model for testing."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=3,
            num_heads=8,
            intermediate_size=1024,
            max_position_embeddings=128
        )
        sample_input = keras.ops.cast(
            keras.random.uniform((1, 16), minval=1, maxval=1000),
            dtype='int32'
        )
        _ = model(sample_input, training=False)
        return model

    def test_forward_pass_input_ids_only(self, built_model):
        """Test forward pass with only input IDs."""
        batch_size, seq_length = 4, 32
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=built_model.vocab_size),
            dtype='int32'
        )
        sequence_output, pooled_output = built_model(input_ids, training=False)
        assert sequence_output.shape == (batch_size, seq_length, built_model.hidden_size)
        assert pooled_output.shape == (batch_size, built_model.hidden_size)

    def test_forward_pass_dict_input(self, built_model):
        """Test forward pass with dictionary input."""
        batch_size, seq_length = 3, 24
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=built_model.vocab_size),
            dtype='int32'
        )
        inputs = {
            'input_ids': input_ids,
            'attention_mask': keras.ops.ones((batch_size, seq_length), dtype='int32'),
            'token_type_ids': keras.ops.zeros((batch_size, seq_length), dtype='int32')
        }
        sequence_output, pooled_output = built_model(inputs, training=False)
        assert sequence_output.shape == (batch_size, seq_length, built_model.hidden_size)
        assert pooled_output.shape == (batch_size, built_model.hidden_size)

    def test_forward_pass_return_dict(self, built_model):
        """Test forward pass with return_dict=True."""
        batch_size, seq_length = 2, 16
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=built_model.vocab_size),
            dtype='int32'
        )
        outputs = built_model(input_ids, return_dict=True, training=False)
        assert isinstance(outputs, dict)
        assert 'last_hidden_state' in outputs
        assert 'pooler_output' in outputs
        assert outputs['last_hidden_state'].shape == (batch_size, seq_length, built_model.hidden_size)
        assert outputs['pooler_output'].shape == (batch_size, built_model.hidden_size)

    def test_forward_pass_no_pooling(self):
        """Test forward pass without pooling layer."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            add_pooling_layer=False
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=1000), dtype='int32'
        )
        outputs = model(input_ids, training=False)
        assert outputs.shape == (2, 16, 256)

    def test_forward_pass_with_attention_mask(self, built_model):
        """Test forward pass with attention mask."""
        batch_size, seq_length = 2, 16
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=built_model.vocab_size),
            dtype='int32'
        )
        # Create attention mask with some padding
        attention_mask = keras.ops.ones((batch_size, seq_length), dtype='int32')
        # Mask out last few tokens
        attention_mask = keras.ops.slice_update(
            attention_mask,
            [0, seq_length - 4],
            keras.ops.zeros((batch_size, 4), dtype='int32')
        )

        sequence_output, pooled_output = built_model(
            input_ids,
            attention_mask=attention_mask,
            training=False
        )
        assert sequence_output.shape == (batch_size, seq_length, built_model.hidden_size)
        assert pooled_output.shape == (batch_size, built_model.hidden_size)

    def test_invalid_dict_input(self, built_model):
        """Test error handling for invalid dictionary input."""
        inputs = {
            'invalid_key': keras.ops.ones((2, 16), dtype='int32')
        }
        with pytest.raises(ValueError, match="Dictionary input must contain 'input_ids' key"):
            built_model(inputs, training=False)


class TestBERTModelSerialization:
    """Test BERT model serialization and deserialization."""

    def test_config_serialization(self):
        """Test model configuration serialization."""
        model = BERT(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=2048,
            hidden_dropout_prob=0.15,
            normalization_type="rms_norm"
        )
        model_config = model.get_config()
        assert isinstance(model_config, dict)
        assert model_config['vocab_size'] == 25000
        assert model_config['hidden_size'] == 512
        assert model_config['num_layers'] == 8
        assert model_config['hidden_dropout_prob'] == 0.15
        assert model_config['normalization_type'] == "rms_norm"

    def test_model_from_config(self):
        """Test creating model from configuration."""
        original_model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            normalization_type="rms_norm",
            attention_type="multi_head"
        )
        config = original_model.get_config()
        new_model = BERT.from_config(config)

        assert new_model.vocab_size == original_model.vocab_size
        assert new_model.hidden_size == original_model.hidden_size
        assert new_model.num_layers == original_model.num_layers
        assert new_model.normalization_type == original_model.normalization_type

    def test_model_save_load(self):
        """Test saving and loading complete model."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            intermediate_size=512
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=1000), dtype='int32'
        )
        original_outputs = model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_bert.keras')
            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

            assert isinstance(original_outputs, tuple)
            assert isinstance(loaded_outputs, tuple)
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs[0]),
                keras.ops.convert_to_numpy(loaded_outputs[0]),
                rtol=1e-5, atol=1e-6,
                err_msg="Sequence outputs should match"
            )
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(original_outputs[1]),
                keras.ops.convert_to_numpy(loaded_outputs[1]),
                rtol=1e-5, atol=1e-6,
                err_msg="Pooled outputs should match"
            )


class TestBERTEdgeCases:
    """Test BERT model edge cases and error handling."""

    def test_minimum_sequence_length(self):
        """Test BERT with minimum sequence length."""
        model = BERT(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        input_ids = keras.ops.cast([[42]], dtype='int32')
        sequence_output, pooled_output = model(input_ids, training=False)
        assert sequence_output.shape == (1, 1, 128)
        assert pooled_output.shape == (1, 128)

    def test_single_sample_batch(self):
        """Test BERT with batch size of 1."""
        model = BERT(vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8)
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 32), minval=1, maxval=1000), dtype='int32'
        )
        sequence_output, pooled_output = model(input_ids, training=False)
        assert sequence_output.shape == (1, 32, 128)
        assert pooled_output.shape == (1, 128)

    def test_long_sequence(self):
        """Test BERT with sequence length at maximum position embeddings."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            max_position_embeddings=128
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 128), minval=1, maxval=1000), dtype='int32'
        )
        sequence_output, pooled_output = model(input_ids, training=False)
        assert sequence_output.shape == (1, 128, 256)
        assert pooled_output.shape == (1, 256)


class TestBERTFactoryFunctions:
    """Test BERT factory functions."""

    def test_create_bert_for_classification(self):
        """Test create_bert_for_classification factory function."""
        config_dict = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 3,
            "num_heads": 8,
            "intermediate_size": 1024
        }
        num_labels = 5
        model = create_bert_for_classification(config_dict, num_labels)
        assert isinstance(model, keras.Model)

        input_ids = keras.ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        attention_mask = keras.ops.ones((2, 32), dtype='int32')
        token_type_ids = keras.ops.zeros((2, 32), dtype='int32')
        logits = model([input_ids, attention_mask, token_type_ids], training=False)
        assert logits.shape == (2, num_labels)

    def test_create_bert_for_classification_invalid_labels(self):
        """Test create_bert_for_classification with invalid number of labels."""
        config_dict = {"vocab_size": 1000, "hidden_size": 256, "num_layers": 2, "num_heads": 8}
        with pytest.raises(ValueError, match="num_labels must be positive"):
            create_bert_for_classification(config_dict, num_labels=0)

    def test_create_bert_for_sequence_output(self):
        """Test create_bert_for_sequence_output factory function."""
        config_dict = {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 3,
            "num_heads": 8,
            "intermediate_size": 1024
        }
        model = create_bert_for_sequence_output(config_dict)
        assert isinstance(model, keras.Model)

        input_ids = keras.ops.cast(
            keras.random.uniform((2, 32), minval=1, maxval=config_dict['vocab_size']), dtype='int32'
        )
        attention_mask = keras.ops.ones((2, 32), dtype='int32')
        token_type_ids = keras.ops.zeros((2, 32), dtype='int32')
        sequence_output = model([input_ids, attention_mask, token_type_ids], training=False)
        assert sequence_output.shape == (2, 32, config_dict['hidden_size'])

    def test_create_bert_convenience_function(self):
        """Test the create_bert convenience function."""
        # Test classification task
        model = create_bert(
            variant="small",
            num_classes=3,
            task_type="classification"
        )
        assert isinstance(model, keras.Model)

        # Test sequence output task
        model = create_bert(
            variant="tiny",
            task_type="sequence_output"
        )
        assert isinstance(model, keras.Model)

    def test_create_bert_invalid_variant(self):
        """Test create_bert with invalid variant."""
        with pytest.raises(ValueError, match="Unknown variant: invalid"):
            create_bert(variant="invalid", task_type="classification", num_classes=2)

    def test_create_bert_invalid_task_type(self):
        """Test create_bert with invalid task type."""
        with pytest.raises(ValueError, match="Unknown task_type: invalid"):
            create_bert(variant="base", task_type="invalid")

    def test_create_bert_missing_num_classes(self):
        """Test create_bert classification without num_classes."""
        with pytest.raises(ValueError, match="num_classes must be provided"):
            create_bert(variant="base", task_type="classification")


class TestBERTIntegration:
    """Integration tests for BERT components working together."""

    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire BERT model."""
        config_dict = {
            "vocab_size": 1000,
            "hidden_size": 64,
            "num_layers": 2,
            "num_heads": 8,
            "intermediate_size": 256
        }
        model = create_bert_for_classification(config_dict, num_labels=3)
        batch_size = 2
        seq_length = 16

        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=config_dict['vocab_size']),
            dtype='int32'
        )
        attention_mask = keras.ops.ones((batch_size, seq_length), dtype='int32')
        # Use varied token_type_ids to ensure gradient flow to token type embeddings
        token_type_ids = keras.ops.concatenate([
            keras.ops.zeros((batch_size, seq_length // 2), dtype="int32"),
            keras.ops.ones((batch_size, seq_length - seq_length // 2), dtype="int32"),
        ], axis=1)

        with tf.GradientTape() as tape:
            logits = model([input_ids, attention_mask, token_type_ids], training=True)
            targets = keras.ops.one_hot(keras.ops.array([0, 2]), 3)
            loss = keras.ops.mean(keras.losses.categorical_crossentropy(targets, logits))

        gradients = tape.gradient(loss, model.trainable_weights)
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > len(model.trainable_weights) * 0.5
        grad_norms = [keras.ops.sqrt(keras.ops.sum(keras.ops.square(g))) for g in non_none_grads]
        assert all(norm > 0.0 for norm in grad_norms)

    def test_training_integration(self):
        """Test BERT model in a minimal training loop."""
        model = create_bert_for_classification(
            {"vocab_size": 1000, "hidden_size": 128, "num_layers": 2, "num_heads": 8},
            num_labels=2
        )

        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        batch_size = 4
        seq_length = 16

        # Create synthetic data
        input_ids = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), minval=1, maxval=1000),
            dtype='int32'
        )
        attention_mask = keras.ops.ones((batch_size, seq_length), dtype='int32')
        token_type_ids = keras.ops.zeros((batch_size, seq_length), dtype='int32')
        labels = keras.ops.cast(
            keras.random.uniform((batch_size,), minval=0, maxval=2),
            dtype='int32'
        )

        initial_loss = None
        for step in range(3):
            with tf.GradientTape() as tape:
                logits = model([input_ids, attention_mask, token_type_ids], training=True)
                loss = keras.ops.mean(
                    keras.losses.sparse_categorical_crossentropy(labels, logits)
                )

            if initial_loss is None:
                initial_loss = loss

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        # Loss should decrease (at least not increase significantly)
        assert loss <= initial_loss + 0.1


class TestBERTAdvancedFeatures:
    """Test advanced BERT features from dl-techniques framework."""

    def test_different_normalization_types(self):
        """Test BERT with different normalization types."""
        base_config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 8,
            "intermediate_size": 512
        }

        for norm_type in ["layer_norm", "rms_norm"]:
            model = BERT(**base_config, normalization_type=norm_type, add_pooling_layer=False)
            input_ids = keras.ops.cast(
                keras.random.uniform((2, 16), minval=1, maxval=1000), dtype='int32'
            )
            output = model(input_ids, training=False)
            assert output.shape == (2, 16, 128)

    def test_different_ffn_types(self):
        """Test BERT with different FFN types."""
        base_config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 8,
            "intermediate_size": 512
        }

        for ffn_type in ["mlp", "swiglu"]:
            model = BERT(**base_config, ffn_type=ffn_type, add_pooling_layer=False)
            input_ids = keras.ops.cast(
                keras.random.uniform((2, 16), minval=1, maxval=1000), dtype='int32'
            )
            output = model(input_ids, training=False)
            assert output.shape == (2, 16, 128)

    def test_stochastic_depth(self):
        """Test BERT with stochastic depth enabled."""
        model = BERT(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1,
            add_pooling_layer=False
        )

        input_ids = keras.ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=1000), dtype='int32'
        )

        # Test that stochastic depth works differently in training vs inference
        output_train = model(input_ids, training=True)
        output_inference = model(input_ids, training=False)

        assert output_train.shape == output_inference.shape == (2, 16, 256)
        # Note: Due to randomness, outputs may differ between training and inference

    def test_different_attention_types(self):
        """Test BERT with different attention mechanisms."""
        base_config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 8,
            "intermediate_size": 512
        }

        for attention_type in ["multi_head", "differential"]:
            model = BERT(
                **base_config,
                attention_type=attention_type,
                add_pooling_layer=False
            )
            input_ids = keras.ops.cast(
                keras.random.uniform((2, 16), minval=1, maxval=1000), dtype='int32'
            )
            output = model(input_ids, training=False)
            assert output.shape == (2, 16, 128)

    def test_model_summary(self):
        """Test that model summary works without errors."""
        model = BERT.from_variant("tiny")

        # Build the model first
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 16), minval=1, maxval=1000), dtype='int32'
        )
        _ = model(input_ids, training=False)

        # Summary should not raise an error
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary raised an exception: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])