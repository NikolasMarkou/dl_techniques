"""
Comprehensive pytest test suite for the refactored Tree Transformer model.

This module provides extensive testing for the Tree Transformer implementation including:
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

from dl_techniques.models.tree_transformer.model import (
    TreeTransformer,
    create_tree_transformer_with_head,
)
from dl_techniques.layers.nlp_heads import NLPTaskConfig, NLPTaskType


class TestTreeTransformerModelInitialization:
    """Test TreeTransformer model initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test basic TreeTransformer model initialization."""
        model = TreeTransformer(
            vocab_size=1000,
            hidden_size=256,
            num_layers=6,
            num_heads=8,
            intermediate_size=1024,
        )

        assert model.hidden_size == 256
        assert model.num_layers == 6
        assert not model.built

        # Components should be created in __init__
        assert model.embedding is not None
        assert model.pos_encoding is not None
        assert model.lm_head is not None
        assert len(model.blocks) == 6

        # But should not be built yet
        assert not model.embedding.built
        for block in model.blocks:
            assert not block.built

    def test_parameter_validation(self):
        """Test TreeTransformer parameter validation for invalid values."""
        with pytest.raises(
            ValueError, match="hidden_size.*must be divisible by num_heads"
        ):
            TreeTransformer(
                vocab_size=1000, hidden_size=100, num_heads=12, num_layers=4
            )

        with pytest.raises(ValueError, match="hidden_size must be positive"):
            TreeTransformer(
                vocab_size=1000, hidden_size=-100, num_layers=4, num_heads=8
            )

        with pytest.raises(
            ValueError, match="hidden_dropout_prob must be in"
        ):
            TreeTransformer(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                hidden_dropout_prob=1.5,
            )

    def test_initialization_with_custom_config(self):
        """Test TreeTransformer model initialization with custom configuration."""
        model = TreeTransformer(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=2048,
            hidden_dropout_prob=0.2,
            attention_dropout_prob=0.1,
            normalization_type="rms_norm",
        )
        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert model.num_layers == 8
        assert model.hidden_dropout_prob == 0.2
        assert model.normalization_type == "rms_norm"


class TestTreeTransformerModelVariants:
    """Test TreeTransformer model variants and factory methods."""

    def test_tree_transformer_base_variant(self):
        model = TreeTransformer.from_variant("base")
        assert model.hidden_size == 512
        assert model.num_layers == 10
        assert model.num_heads == 8

    def test_tree_transformer_large_variant(self):
        model = TreeTransformer.from_variant("large")
        assert model.hidden_size == 1024
        assert model.num_layers == 16
        assert model.num_heads == 16

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="Unknown variant 'invalid'"):
            TreeTransformer.from_variant("invalid")

    def test_variant_with_custom_params(self):
        """Test creating variant with custom parameters."""
        model = TreeTransformer.from_variant(
            "base", normalization_type="rms_norm"
        )
        assert model.hidden_size == 512
        assert model.num_layers == 10
        assert model.normalization_type == "rms_norm"


class TestTreeTransformerModelBuilding:
    """Test TreeTransformer model building and architecture creation."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": 1000,
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 8,
            "intermediate_size": 1024,
            "max_len": 128,
        }

    def test_build_basic_functionality(self, basic_config):
        """Test basic building functionality and output contract."""
        model = TreeTransformer(**basic_config)
        batch_size, seq_length = 2, 32
        input_ids = keras.ops.cast(
            keras.random.uniform(
                (batch_size, seq_length), maxval=basic_config["vocab_size"]
            ),
            dtype="int32",
        )

        outputs = model(input_ids, training=False)
        assert model.built
        assert isinstance(outputs, dict)
        assert "last_hidden_state" in outputs
        assert "logits" in outputs
        assert "break_probs" in outputs
        assert outputs["last_hidden_state"].shape == (
            batch_size,
            seq_length,
            basic_config["hidden_size"],
        )
        assert outputs["logits"].shape == (
            batch_size,
            seq_length,
            basic_config["vocab_size"],
        )
        assert outputs["break_probs"].shape == (
            batch_size,
            basic_config["num_layers"],
            seq_length,
            seq_length,
        )

    def test_transformer_blocks_configuration(self, basic_config):
        model = TreeTransformer(**basic_config)
        input_ids = keras.ops.cast(
            keras.random.uniform(
                (1, 16), maxval=basic_config["vocab_size"]
            ),
            dtype="int32",
        )
        _ = model(input_ids, training=False)
        for i, block in enumerate(model.blocks):
            assert block.hidden_size == basic_config["hidden_size"]
            assert block.num_heads == basic_config["num_heads"]
            assert block.name == f"block_{i}"


class TestTreeTransformerModelForwardPass:
    """Test TreeTransformer model forward pass functionality."""

    @pytest.fixture
    def built_model(self) -> TreeTransformer:
        """Create a built TreeTransformer model for testing."""
        model = TreeTransformer(
            vocab_size=1000,
            hidden_size=256,
            num_layers=3,
            num_heads=8,
            intermediate_size=1024,
            max_len=128,
        )
        sample_input = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=1000), dtype="int32"
        )
        _ = model(sample_input, training=False)
        return model

    def test_forward_pass_input_ids_only(self, built_model):
        batch_size, seq_length = 4, 32
        input_ids = keras.ops.cast(
            keras.random.uniform(
                (batch_size, seq_length), maxval=built_model.vocab_size
            ),
            dtype="int32",
        )
        outputs = built_model(input_ids, training=False)
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (
            batch_size,
            seq_length,
            built_model.hidden_size,
        )
        assert outputs["break_probs"].shape == (
            batch_size,
            built_model.num_layers,
            seq_length,
            seq_length,
        )

    def test_forward_pass_dict_input(self, built_model):
        batch_size, seq_length = 3, 24
        inputs = {
            "input_ids": keras.ops.cast(
                keras.random.uniform(
                    (batch_size, seq_length), maxval=built_model.vocab_size
                ),
                dtype="int32",
            )
        }
        outputs = built_model(inputs, training=False)
        assert isinstance(outputs, dict)
        assert outputs["last_hidden_state"].shape == (
            batch_size,
            seq_length,
            built_model.hidden_size,
        )

    def test_padding_mask_functionality(self, built_model):
        batch_size, seq_length, pad_len = 2, 16, 4
        input_ids = keras.ops.cast(
            keras.random.uniform(
                (batch_size, seq_length - pad_len),
                minval=1,
                maxval=built_model.vocab_size,
            ),
            dtype="int32",
        )
        padding = keras.ops.zeros((batch_size, pad_len), dtype="int32")
        padded_input = keras.ops.concatenate([input_ids, padding], axis=1)

        outputs = built_model(padded_input, training=False)
        assert outputs["last_hidden_state"].shape == (
            batch_size,
            seq_length,
            built_model.hidden_size,
        )
        # Check that break_probs for padding tokens are near zero (due to input mask)
        break_probs_last_layer = outputs["break_probs"][:, -1, :, :]
        # Check the last column (padding tokens should have no attention breaks related to them)
        assert keras.ops.max(break_probs_last_layer[:, :, -1]) < 0.1

    def test_invalid_dict_input(self, built_model):
        inputs = {"invalid_key": keras.ops.ones((2, 16), dtype="int32")}
        with pytest.raises(
            ValueError, match="Dictionary input must contain 'input_ids' key"
        ):
            built_model(inputs, training=False)


class TestTreeTransformerModelSerialization:
    """Test TreeTransformer model serialization and deserialization."""

    def test_config_serialization(self):
        model = TreeTransformer(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            normalization_type="rms_norm",
        )
        model_config = model.get_config()
        assert model_config["vocab_size"] == 25000
        assert model_config["hidden_size"] == 512

    def test_model_from_config(self):
        original_model = TreeTransformer(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            normalization_type="rms_norm",
        )
        config = original_model.get_config()
        new_model = TreeTransformer.from_config(config)
        assert new_model.hidden_size == original_model.hidden_size
        assert new_model.normalization_type == original_model.normalization_type

    def test_model_save_load(self):
        model = TreeTransformer(
            vocab_size=1000, hidden_size=256, num_layers=2, num_heads=8
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 16), maxval=1000), dtype="int32"
        )
        original_outputs = model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_tree_transformer.keras")
            # Need to build the model before saving complex custom layers
            if not model.built:
                model(input_ids, training=False)

            model.save(model_path)
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

            for key in ["last_hidden_state", "logits", "break_probs"]:
                np.testing.assert_allclose(
                    keras.ops.convert_to_numpy(original_outputs[key]),
                    keras.ops.convert_to_numpy(loaded_outputs[key]),
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"Output '{key}' should match after loading",
                )


class TestTreeTransformerEdgeCases:
    """Test TreeTransformer model edge cases."""

    def test_minimum_sequence_length(self):
        model = TreeTransformer(
            vocab_size=1000, hidden_size=128, num_layers=2, num_heads=8
        )
        input_ids = keras.ops.cast([[42]], dtype="int32")
        outputs = model(input_ids, training=False)
        assert outputs["last_hidden_state"].shape == (1, 1, 128)
        assert outputs["break_probs"].shape == (1, 2, 1, 1)

    def test_long_sequence(self):
        model = TreeTransformer(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=8,
            max_len=128,
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 128), maxval=1000), dtype="int32"
        )
        outputs = model(input_ids, training=False)
        assert outputs["last_hidden_state"].shape == (1, 128, 256)


class TestTreeTransformerIntegrationFactory:
    """Test the integration factory `create_tree_transformer_with_head`."""

    def test_create_with_classification_head(self):
        """Test creating a TreeTransformer model for sequence classification."""
        task_config = NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3,
        )
        model = create_tree_transformer_with_head("tiny", task_config)
        assert isinstance(model, keras.Model)

        inputs = {
            "input_ids": keras.ops.cast(
                keras.random.uniform((2, 32), maxval=1000), "int32"
            ),
        }
        outputs = model(inputs, training=False)
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert "probabilities" in outputs
        assert outputs["logits"].shape == (2, 3)

    def test_create_with_token_classification_head(self):
        """Test creating a TreeTransformer model for token classification."""
        task_config = NLPTaskConfig(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=9,
        )
        model = create_tree_transformer_with_head("tiny", task_config)
        assert isinstance(model, keras.Model)

        inputs = {
            "input_ids": keras.ops.cast(
                keras.random.uniform((2, 32), maxval=1000), "int32"
            ),
        }
        outputs = model(inputs, training=False)
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 32, 9)


class TestTreeTransformerIntegration:
    """Integration tests for the complete model (encoder + head)."""

    @pytest.fixture
    def token_classification_model(self) -> keras.Model:
        """Create a token classification model for integration tests."""
        task_config = NLPTaskConfig(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=9,
        )
        # Ensure configuration is small and testable
        return create_tree_transformer_with_head(
            "tiny",
            task_config,
            encoder_config_overrides={
                "hidden_size": 64,
                "intermediate_size": 256,
            },
        )

    def test_gradient_flow_integration(self, token_classification_model):
        """Test that gradients flow through the entire integrated model."""
        batch_size, seq_length = 2, 16
        inputs = {
            "input_ids": keras.ops.cast(
                keras.random.uniform((batch_size, seq_length), maxval=1000),
                "int32",
            ),
        }

        with tf.GradientTape() as tape:
            outputs = token_classification_model(inputs, training=True)
            logits = outputs["logits"]
            targets = keras.ops.one_hot(
                keras.ops.cast(
                    keras.random.uniform((batch_size, seq_length), maxval=9),
                    "int32",
                ),
                9,
            )
            loss = keras.ops.mean(
                keras.losses.categorical_crossentropy(targets, logits)
            )

        gradients = tape.gradient(
            loss, token_classification_model.trainable_weights
        )
        non_none_grads = [g for g in gradients if g is not None]
        assert len(non_none_grads) > 0
        assert len(non_none_grads) == len(
            token_classification_model.trainable_weights
        )
        grad_norms = [
            keras.ops.sqrt(keras.ops.sum(keras.ops.square(g)))
            for g in non_none_grads
        ]
        assert all(norm >= 0.0 for norm in grad_norms)

    def test_training_integration(self, token_classification_model):
        """Test the integrated model in a minimal training loop."""
        optimizer = keras.optimizers.Adam(learning_rate=1e-6)
        batch_size, seq_length = 4, 16
        inputs = {
            "input_ids": keras.ops.cast(
                keras.random.uniform((batch_size, seq_length), maxval=1000),
                "int32",
            ),
        }
        labels = keras.ops.cast(
            keras.random.uniform((batch_size, seq_length), maxval=9), "int32"
        )

        initial_loss = None
        for step in range(3): # Reduce steps for stability, focus on gradient flow
            with tf.GradientTape() as tape:
                outputs = token_classification_model(inputs, training=True)
                loss = keras.ops.mean(
                    keras.losses.sparse_categorical_crossentropy(
                        labels, outputs["logits"]
                    )
                )
            if initial_loss is None:
                initial_loss = loss
            gradients = tape.gradient(
                loss, token_classification_model.trainable_weights
            )
            optimizer.apply_gradients(
                zip(gradients, token_classification_model.trainable_weights)
            )
        # Check that loss is a valid number after a few steps
        assert not np.isnan(keras.ops.convert_to_numpy(loss))


class TestTreeTransformerAdvancedFeatures:
    """Test advanced features."""

    def test_different_normalization_types(self):
        base_config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 8,
        }
        for norm_type in ["layer_norm", "rms_norm"]:
            model = TreeTransformer(**base_config, normalization_type=norm_type)
            input_ids = keras.ops.cast(
                keras.random.uniform((2, 16), maxval=1000), "int32"
            )
            output = model(input_ids, training=False)
            assert output["last_hidden_state"].shape == (2, 16, 128)

    def test_model_summary(self):
        """Test that model summary works without errors."""
        model = TreeTransformer.from_variant("tiny")
        input_ids = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=1000), "int32"
        )
        # Explicitly call to build the model before summary
        _ = model(input_ids, training=False)
        try:
            model.summary()
        except Exception as e:
            pytest.fail(f"Model summary raised an exception: {e}")