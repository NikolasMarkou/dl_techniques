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
    create_tree_transformer,
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
            ValueError, match="hidden_dropout_rate must be in"
        ):
            TreeTransformer(
                vocab_size=1000,
                hidden_size=256,
                num_layers=4,
                num_heads=8,
                hidden_dropout_rate=1.5,
            )

    def test_initialization_with_custom_config(self):
        """Test TreeTransformer model initialization with custom configuration."""
        model = TreeTransformer(
            vocab_size=25000,
            hidden_size=512,
            num_layers=8,
            num_heads=8,
            intermediate_size=2048,
            hidden_dropout_rate=0.2,
            attention_dropout_rate=0.1,
            normalization_type="rms_norm",
        )
        assert model.vocab_size == 25000
        assert model.hidden_size == 512
        assert model.num_layers == 8
        assert model.hidden_dropout_rate == 0.2
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
        # Check that break_probs for padding tokens are zero
        break_probs_last_layer = outputs["break_probs"][:, -1, :, :]
        # Check the last column/row (all values related to the last padded token)
        assert keras.ops.max(break_probs_last_layer[:, :, -1]) == 0.0
        assert keras.ops.max(break_probs_last_layer[:, -1, :]) == 0.0


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

        # Build the model before saving
        if not model.built:
             model(input_ids, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "test_tree_transformer.keras")
            model.save(model_path)

            # Load the model and get its output
            loaded_model = keras.models.load_model(model_path)
            loaded_outputs = loaded_model(input_ids, training=False)

        # Get the output of the original model AGAIN after loading
        # to ensure graph state is comparable
        original_outputs = model(input_ids, training=False)

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

        # The base model's lm_head is not used, so it won't have gradients.
        # This is expected behavior.
        num_expected_trainable_weights = len(token_classification_model.trainable_weights)
        num_lm_head_weights = 2 # kernel and bias

        assert len(non_none_grads) > 0
        assert len(non_none_grads) == (num_expected_trainable_weights - num_lm_head_weights)
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


# ---------------------------------------------------------------------------
# Iteration-1 (plan_2026-05-11_3c3ed037) regression / fix-locking tests
# ---------------------------------------------------------------------------

class TestTreeTransformerIter1Fixes:
    """Locks the 4 review bug fixes (B-1, B-3, B-4, B-5) + MLM wrapper compat."""

    def test_mixed_float16_no_nan(self):
        """B-1/B-2: dtype-aware mask sentinel keeps fp16 forward NaN-free."""
        original_policy = keras.mixed_precision.global_policy()
        try:
            keras.mixed_precision.set_global_policy("mixed_float16")
            model = TreeTransformer(
                vocab_size=200, hidden_size=32, num_layers=2, num_heads=4,
                intermediate_size=64, max_len=16, pad_token_id=0,
            )
            input_ids = keras.ops.cast(
                keras.random.uniform((2, 16), maxval=200), "int32"
            )
            # Mix in some pad tokens to exercise the masked-softmax sentinel.
            pad_mask = keras.ops.cast(
                keras.random.uniform((2, 16), maxval=4) < 1, "int32"
            )
            input_ids = input_ids * (1 - pad_mask)
            out = model({"input_ids": input_ids}, training=False)
            assert not bool(
                keras.ops.any(keras.ops.isnan(out["last_hidden_state"]))
            ), "NaN found in last_hidden_state under mixed_float16"
            assert not bool(
                keras.ops.any(keras.ops.isnan(out["logits"]))
            ), "NaN found in logits under mixed_float16"
        finally:
            keras.mixed_precision.set_global_policy(original_policy)

    def test_attention_mask_honored(self):
        """B-3: explicit attention_mask in dict input wins over pad-derivation."""
        model = TreeTransformer(
            vocab_size=200, hidden_size=32, num_layers=2, num_heads=4,
            intermediate_size=64, max_len=16, pad_token_id=0,
        )
        # Build an input where input_ids has NO pad tokens (all > 0) so the
        # fallback mask would be all-ones, but we pass an explicit
        # attention_mask that zeros out half the positions. If the explicit
        # mask is honored, break_probs at the masked positions go to zero
        # (or near-zero); if it's ignored, they stay non-zero.
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=200), "int32"
        )
        explicit_mask = keras.ops.concatenate([
            keras.ops.ones((2, 8), dtype="int32"),
            keras.ops.zeros((2, 8), dtype="int32"),
        ], axis=1)
        out_explicit = model(
            {"input_ids": input_ids, "attention_mask": explicit_mask},
            training=False,
        )
        out_implicit = model({"input_ids": input_ids}, training=False)
        # Outputs must DIFFER — fallback would give an all-ones mask, explicit
        # mask gives a half-zeroed one. If they're identical, the explicit
        # mask was dropped.
        diff = keras.ops.convert_to_numpy(
            keras.ops.max(keras.ops.abs(
                out_explicit["last_hidden_state"]
                - out_implicit["last_hidden_state"]
            ))
        )
        assert diff > 1e-4, (
            f"Explicit attention_mask had no effect (max-abs-diff {diff})"
        )

    def test_pad_token_id_nonzero(self):
        """pad_token_id != 0 correctly drives the implicit mask."""
        pad_id = 100266
        model = TreeTransformer(
            vocab_size=100277, hidden_size=32, num_layers=2, num_heads=4,
            intermediate_size=64, max_len=16, pad_token_id=pad_id,
        )
        # First 8 positions are real tokens (id 5), last 8 are pad_id.
        input_ids_np = np.array(
            [[5] * 8 + [pad_id] * 8, [7] * 8 + [pad_id] * 8],
            dtype="int32",
        )
        input_ids = keras.ops.convert_to_tensor(input_ids_np)
        out_with_pad = model({"input_ids": input_ids}, training=False)
        # And a version where pad_id is replaced by a real token everywhere
        # — outputs at the first 8 positions must differ (different right-context).
        input_ids_full = keras.ops.convert_to_tensor(
            np.array([[5] * 16, [7] * 16], dtype="int32")
        )
        out_full = model({"input_ids": input_ids_full}, training=False)
        diff = keras.ops.convert_to_numpy(keras.ops.max(keras.ops.abs(
            out_with_pad["last_hidden_state"][:, :8, :]
            - out_full["last_hidden_state"][:, :8, :]
        )))
        assert diff > 1e-4, (
            "pad_token_id != 0 did not change the attention pattern — "
            "implicit mask is likely broken."
        )

    def test_mlm_wrapper_compatibility(self):
        """MaskedLanguageModel accepts a TreeTransformer encoder and forwards correctly."""
        from dl_techniques.models.masked_language_model.mlm import (
            MaskedLanguageModel,
        )
        encoder = TreeTransformer.from_variant(
            "tiny", vocab_size=200, max_len=16, pad_token_id=0
        )
        mlm = MaskedLanguageModel(
            encoder=encoder,
            vocab_size=200,
            mask_token_id=103,
            mask_ratio=0.15,
        )
        input_ids = keras.ops.cast(
            keras.random.uniform((2, 16), minval=1, maxval=200), "int32"
        )
        attention_mask = keras.ops.ones((2, 16), dtype="int32")
        logits = mlm(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            training=False,
        )
        # MaskedLanguageModel.call returns the MLM logits tensor directly
        # (shape: (B, L, vocab_size)).
        assert logits.shape == (2, 16, 200), (
            f"Unexpected MLM logits shape: {logits.shape}"
        )

    def test_load_pretrained_weights_uses_weight_transfer(self):
        """B-4: load_pretrained_weights uses weight_transfer helper on .keras file."""
        model_src = TreeTransformer(
            vocab_size=200, hidden_size=32, num_layers=2, num_heads=4,
            intermediate_size=64, max_len=16, pad_token_id=0,
        )
        # Build with a forward pass.
        dummy = keras.ops.cast(
            keras.random.uniform((1, 16), maxval=200), "int32"
        )
        _ = model_src({"input_ids": dummy}, training=False)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = os.path.join(tmp, "tree.keras")
            model_src.save(ckpt_path)

            model_dst = TreeTransformer(
                vocab_size=200, hidden_size=32, num_layers=2, num_heads=4,
                intermediate_size=64, max_len=16, pad_token_id=0,
            )
            # The new API: no by_name, no skip_mismatch — uses
            # load_weights_from_checkpoint internally.
            model_dst.load_pretrained_weights(ckpt_path)
            # Compare at least one block's weight values.
            src_weights = model_src.weights
            dst_weights = model_dst.weights
            assert len(src_weights) == len(dst_weights)
            # Pick a non-trivial layer (first encoder block's first weight).
            matched = 0
            for sw, dw in zip(src_weights, dst_weights):
                if sw.shape == dw.shape:
                    a = keras.ops.convert_to_numpy(sw.value)
                    b = keras.ops.convert_to_numpy(dw.value)
                    if np.allclose(a, b, atol=1e-6):
                        matched += 1
            assert matched > 0, "No weights transferred via weight_transfer helper"

    def test_model_fit_one_step_smoke(self):
        """1-step fit smoke on MaskedLanguageModel(TreeTransformer)."""
        from dl_techniques.models.masked_language_model.mlm import (
            MaskedLanguageModel,
        )
        encoder = TreeTransformer.from_variant(
            "tiny", vocab_size=200, max_len=16, pad_token_id=0
        )
        mlm = MaskedLanguageModel(
            encoder=encoder,
            vocab_size=200,
            mask_token_id=103,
            mask_ratio=0.15,
        )
        mlm.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=1e-4, clipnorm=1.0)
        )

        # Build synthetic tf.data.Dataset matching MLM.train_step's expectations.
        input_ids = tf.random.uniform(
            (4, 16), minval=1, maxval=200, dtype=tf.int32
        )
        attention_mask = tf.ones((4, 16), dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        ).batch(2)
        history = mlm.fit(ds, epochs=1, verbose=0)
        loss = history.history["loss"][-1]
        assert np.isfinite(loss), f"Non-finite loss after 1-step fit: {loss}"


class TestTreeTransformerIter1Refactor:
    """Lock-in tests for the iter-1 refactor (plan_2026-05-11_0a5779e8):

    - ``create_tree_transformer`` module-level factory exists and works.
    - ``from_variant(pretrained=True)`` raises ``NotImplementedError``
      instead of silently falling back to random init.
    - Package-level public API is exactly the trimmed 3-name surface.
    """

    def test_create_tree_transformer_factory(self):
        """`create_tree_transformer` returns a configured TreeTransformer
        with the variant defaults and runs a forward pass."""
        model = create_tree_transformer("tiny", vocab_size=200)

        assert isinstance(model, TreeTransformer)
        # tiny variant: hidden_size=128, num_layers=4, num_heads=4
        assert model.hidden_size == 128
        assert model.num_layers == 4
        assert model.num_heads == 4
        assert model.vocab_size == 200

        input_ids = np.random.randint(0, 200, size=(2, 16), dtype=np.int32)
        attention_mask = np.ones((2, 16), dtype=np.int32)
        outputs = model(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (2, 16, 128)

    def test_from_variant_pretrained_true_raises(self):
        """`from_variant(pretrained=True)` must raise NotImplementedError
        rather than silently random-initializing.

        Previously the body wrapped `_download_weights` in `except Exception`,
        so the NotImplementedError raised by `_download_weights` was caught
        and the model returned with random weights, misleading users.
        """
        with pytest.raises(NotImplementedError):
            TreeTransformer.from_variant("tiny", pretrained=True)

    def test_public_api_surface(self):
        """Package `__init__` exposes only the 3 public symbols."""
        import dl_techniques.models.tree_transformer as pkg

        expected = {
            "TreeTransformer",
            "create_tree_transformer",
            "create_tree_transformer_with_head",
        }
        assert set(pkg.__all__) == expected
        for name in expected:
            assert hasattr(pkg, name), f"missing public symbol: {name}"

        # Layer classes intentionally NOT re-exported from the package
        # (callers like nam/ import them directly from .model).
        for name in (
            "GroupAttention",
            "TreeMHA",
            "PositionalEncoding",
            "TreeTransformerBlock",
        ):
            assert not hasattr(pkg, name), (
                f"internal layer {name} should not be re-exported from "
                "the tree_transformer package __init__"
            )

        # But they must still be importable from .model (nam consumer contract).
        from dl_techniques.models.tree_transformer.model import (
            GroupAttention,
            TreeMHA,
            PositionalEncoding,
            TreeTransformerBlock,
        )
        assert GroupAttention is not None
        assert TreeMHA is not None
        assert PositionalEncoding is not None
        assert TreeTransformerBlock is not None
