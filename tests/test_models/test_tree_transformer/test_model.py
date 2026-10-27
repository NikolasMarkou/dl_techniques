"""
Comprehensive pytest test suite for the refactored Tree Transformer model.

This module provides extensive testing for the Tree Transformer implementation including:
- Foundation model initialization and parameter validation.
- Architecture building and consistent output shape.
- Forward pass functionality with a standardized dictionary output.
- Model variant creation and configuration.
- Serialization and deserialization of the pure encoder.
- Error handling and edge cases.
- The in-model task head (``include_top=True``) that replaced the functional
  ``create_tree_transformer_with_head`` wrapper.
- End-to-end integration testing for gradient flow and training.
"""

import pytest
import numpy as np
import keras
import tensorflow as tf
import tempfile
import os
import re
from typing import Dict, Any

from dl_techniques.models.language.tree_transformer.model import (
    TreeTransformer,
    create_tree_transformer,
)
from dl_techniques.layers.heads.nlp import NLPTaskConfig, NLPTaskType

from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights


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
    """``create_tree_transformer(..., include_top=True)``: the in-model task head.

    The functional ``create_tree_transformer_with_head`` this replaced returned
    the head's raw dict and gave the head no attention mask; the in-model head
    returns the finalized output (bare logits here) and pools under the mask
    the blocks use.
    """

    def test_create_with_classification_head(self):
        """Sequence classification: bare logits, ``probabilities`` dropped."""
        task_config = NLPTaskConfig(
            name="sentiment",
            task_type=NLPTaskType.SENTIMENT_ANALYSIS,
            num_classes=3,
        )
        model = create_tree_transformer(
            "tiny", include_top=True, task_config=task_config)
        assert isinstance(model, TreeTransformer)

        inputs = {
            "input_ids": keras.ops.cast(
                keras.random.uniform((2, 32), maxval=1000), "int32"
            ),
        }
        outputs = model(inputs, training=False)
        assert not isinstance(outputs, dict)
        assert outputs.shape == (2, 3)

    def test_create_with_token_classification_head(self):
        """Token classification: bare logits, ``predictions`` dropped."""
        task_config = NLPTaskConfig(
            name="ner",
            task_type=NLPTaskType.NAMED_ENTITY_RECOGNITION,
            num_classes=9,
        )
        model = create_tree_transformer(
            "tiny", include_top=True, task_config=task_config)
        assert isinstance(model, TreeTransformer)

        inputs = {
            "input_ids": keras.ops.cast(
                keras.random.uniform((2, 32), maxval=1000), "int32"
            ),
        }
        outputs = model(inputs, training=False)
        assert not isinstance(outputs, dict)
        assert outputs.shape == (2, 32, 9)


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
        return create_tree_transformer(
            "tiny",
            include_top=True,
            task_config=task_config,
            hidden_size=64,
            intermediate_size=256,
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
            logits = token_classification_model(inputs, training=True)
            targets = keras.ops.one_hot(
                keras.ops.cast(
                    keras.random.uniform((batch_size, seq_length), maxval=9),
                    "int32",
                ),
                9,
            )
            loss = keras.ops.mean(
                # from_logits=True: the head emits logits, and the default
                # renormalize-and-clip path can zero every gradient (measured
                # on BERT's identical test; see test_bert.py:359).
                keras.losses.categorical_crossentropy(
                    targets, logits, from_logits=True
                )
            )

        gradients = tape.gradient(
            loss, token_classification_model.trainable_weights
        )
        non_none_grads = [g for g in gradients if g is not None]

        # With a task head the model creates no lm_head, so EVERY trainable
        # weight is on the loss path. The wrapper this replaced carried an
        # unused lm_head and excused its 2 weights (kernel, bias) from here.
        assert len(non_none_grads) > 0
        assert len(non_none_grads) == len(
            token_classification_model.trainable_weights)
        # A Euclidean norm is never negative: the previous
        # `all(norm >= 0.0)` was true of an all-zero gradient, i.e. of a model
        # in which nothing trains. Per-weight positive floor instead, so one
        # dead tensor cannot hide behind the live ones
        # (`test_beit/test_model.py::test_gradients_reach_every_trainable_weight`).
        # An ABSOLUTE floor is the wrong instrument, exactly as
        # `test_mamba/test_mamba_v1.py:726-733` records: MEASURED here,
        # `block_0/group_attention/key_projection/bias` has gradient norm
        # 1.357e-09 -- genuinely non-zero, but below a 1e-8 floor. The claim is
        # "not identically zero and finite", per weight, which is the form
        # `test_beit/test_model.py::test_gradients_reach_every_trainable_weight`
        # uses.
        for weight, grad in zip(
            token_classification_model.trainable_weights, gradients
        ):
            g = keras.ops.convert_to_numpy(grad)
            assert np.isfinite(g).all(), f"non-finite gradient at {weight.path}"
            assert float(np.max(np.abs(g))) > 0.0, (
                f"gradient at {weight.path} is identically zero"
            )

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
                        labels, outputs, from_logits=True
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
        """``normalization_type`` must reach the normalization layers.

        ``last_hidden_state`` is ``(2, 16, 128)`` under both settings, so the
        old assertion held whether or not the kwarg was honoured. RMSNorm has no
        centering shift, so it holds strictly FEWER weights than LayerNorm --
        a structural difference the weight-shape signature pins directly.
        """
        base_config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_layers": 2,
            "num_heads": 8,
        }

        def _build(norm_type):
            model = TreeTransformer(**base_config, normalization_type=norm_type)
            model(keras.ops.zeros((1, 16), dtype="int32"), training=False)
            return model

        builders = {n: (lambda n=n: _build(n)) for n in ["layer_norm", "rms_norm"]}
        sigs = assert_structural_knob_changes_weights(
            builders, knob="normalization_type"
        )
        assert len(sigs["rms_norm"]) < len(sigs["layer_norm"]), (
            "rms_norm should drop one weight per norm layer (no beta), but held "
            f"{len(sigs['rms_norm'])} vs layer_norm's {len(sigs['layer_norm'])}"
        )

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
        from dl_techniques.models.common.lm_pretraining.mlm import (
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
        from dl_techniques.models.common.lm_pretraining.mlm import (
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
    - Package-level public API is exactly the trimmed 2-name surface (the
      third, ``create_tree_transformer_with_head``, was replaced by
      ``include_top`` in plan-2026-10-11T004932-b83935e1).
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
        """Package `__init__` exposes only the 2 public symbols."""
        import dl_techniques.models.language.tree_transformer as pkg

        expected = {
            "TreeTransformer",
            "create_tree_transformer",
        }
        assert set(pkg.__all__) == expected
        assert not hasattr(pkg, "create_tree_transformer_with_head")
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
        from dl_techniques.models.language.tree_transformer.model import (
            GroupAttention,
            TreeMHA,
            PositionalEncoding,
            TreeTransformerBlock,
        )
        assert GroupAttention is not None
        assert TreeMHA is not None
        assert PositionalEncoding is not None
        assert TreeTransformerBlock is not None


class TestTreeTransformerComponentsSplitLockIn:
    """Lock in the model.py/components.py split.

    Reference: plan_2026-05-11_46ecfa0b — model.py contains only the
    top-level model + factories; sub-layers live in components.py and are
    re-exported from model.py for backward compatibility.
    """

    def test_sublayers_importable_from_components_module(self):
        from dl_techniques.models.language.tree_transformer.components import (
            GroupAttention, TreeMHA, PositionalEncoding, TreeTransformerBlock,
        )
        for cls in (GroupAttention, TreeMHA, PositionalEncoding, TreeTransformerBlock):
            assert cls is not None

    def test_sublayers_re_exported_from_model_module(self):
        # Deep-import contract for nam/ and train/nam/.
        from dl_techniques.models.language.tree_transformer.model import (
            GroupAttention, TreeMHA, PositionalEncoding, TreeTransformerBlock,
        )
        # Same object identity — they ARE the components.py classes.
        from dl_techniques.models.language.tree_transformer import components as comp
        assert GroupAttention is comp.GroupAttention
        assert TreeMHA is comp.TreeMHA
        assert PositionalEncoding is comp.PositionalEncoding
        assert TreeTransformerBlock is comp.TreeTransformerBlock

    def test_model_module_has_only_top_level_class(self):
        import dl_techniques.models.language.tree_transformer.model as m
        # Top-level + factories live here.
        assert hasattr(m, "TreeTransformer")
        assert callable(getattr(m, "create_tree_transformer", None))
        # Replaced by `include_top` on the class (plan-2026-10-11T004932-b83935e1).
        assert not hasattr(m, "create_tree_transformer_with_head")
        # Sub-layers are re-exports — verify they live in components, not model.
        assert m.GroupAttention.__module__ == "dl_techniques.models.language.tree_transformer.components"
        assert m.TreeTransformer.__module__ == "dl_techniques.models.language.tree_transformer.model"


# ---------------------------------------------------------------------
# group_prob: the "core innovation" line, pinned by ORIENTATION
# (plan-2026-08-19-a616f581 step 12)
# ---------------------------------------------------------------------

from dl_techniques.models.language.tree_transformer.components import GroupAttention, TreeMHA


# R-038 closure -- plan-2026-08-22T035419-a11304c8 / D-251.
# Keras `ops/nn.py:907` advises that a softmax over a size-1 axis always returns
# exactly 1.0. Every site in this module feeds that axis a size of 1 ON PURPOSE
# -- single class, single token, single head, single anchor, single cluster,
# minimum sequence length -- so the advisory describes the test's own input, not
# a defect. Suppressed HERE rather than in `pyproject.toml` so an ACCIDENTAL
# size-1 softmax anywhere else still fails under `error::UserWarning`.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:You are using a softmax over axis:UserWarning"),
]


class TestGroupProbIsAppliedOnTheKeyAxis:
    """`attention_weights *= expand_dims(group_prob, axis=1)` -- components.py:516.

    Two mutations were run against this package at HEAD, out of tree:

    * `group_prob -> ones_like(group_prob)` (the line DELETED): **1 failed, 40
      passed**. `TestTreeTransformerIntegration::test_gradient_flow_integration`
      catches it, because all 24 `GroupAttention` weights leave the backward
      graph (75 live gradients instead of 99). So F-58's claim that deleting the
      line is undetectable is REFUTED at HEAD.
    * `group_prob -> transpose(group_prob, (0, 2, 1))` (the line kept, its
      matrix TRANSPOSED): **41 passed** -- the whole suite green, including the
      gradient test, since a transpose changes no gradient's support. That is
      the gap this class closes, and it is this repository's most expensive
      documented failure mode (a transposed attention bias once passed 219
      tests).

    The oracle is derived from the CLAIM, not from the implementation: axis 2 of
    `group_prob` indexes KEYS, so changing WHICH key position `group_prob`
    selects must move EVERY query row of the attention output. Feed a
    column-selector matrix (all-zero except column `j`, which is 1) for two
    different `j` and compare.

        correct orientation  -> every row moves (MEASURED min row delta 0.625)
        transposed           -> only rows j0 and j1 move; the other 4 are 0.0
        line deleted         -> no row moves at all; all 6 are exactly 0.0

    One test, RED under both mutations.
    """

    # DECISION plan-2026-08-19T070627-a616f581/D-017
    # The oracle is a COLUMN SELECTOR compared across two key positions, and the
    # assertion is on the MINIMUM row delta. Do NOT "simplify" this to "the
    # output differs when group_prob is replaced by ones": that claim is already
    # made (indirectly) by TestTreeTransformerIntegration::
    # test_gradient_flow_integration, and it stays GREEN on a product whose
    # group_prob matrix is TRANSPOSED -- measured, 41 passed. Do NOT average the
    # row deltas either: the transposed mutant moves 2 of 6 rows a lot, so a mean
    # would stay green. See decisions.md D-017.
    BATCH, SEQ, HIDDEN, HEADS = 1, 6, 32, 4
    J0, J1 = 2, 4

    def _built_mha(self):
        keras.utils.set_random_seed(7)
        mha = TreeMHA(num_heads=self.HEADS, hidden_size=self.HIDDEN,
                      attention_dropout_rate=0.0)
        shape = (self.BATCH, self.SEQ, self.HIDDEN)
        mha.build((shape, shape, shape, None, (self.BATCH, 1, self.SEQ)))
        return mha

    def _column_selector(self, j):
        gp = np.zeros((self.BATCH, self.SEQ, self.SEQ), dtype="float32")
        gp[:, :, j] = 1.0
        return keras.ops.convert_to_tensor(gp)

    def test_moving_the_selected_key_moves_every_query_row(self):
        mha = self._built_mha()
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(11).standard_normal(
                (self.BATCH, self.SEQ, self.HIDDEN)).astype("float32")
        )
        mask = keras.ops.ones((self.BATCH, 1, self.SEQ))

        def run(j):
            return keras.ops.convert_to_numpy(
                mha((x, x, x, self._column_selector(j), mask), training=False)
            )

        per_row = np.max(np.abs(run(self.J0) - run(self.J1)), axis=-1)[0]

        assert per_row.shape == (self.SEQ,)
        # MEASURED min 0.625 in the correct orientation; the two failure modes
        # both produce EXACT zeros (structural, not rounding), so the bound is
        # not a calibrated threshold -- it separates "moved" from "0.0".
        assert float(per_row.min()) > 1e-3, (
            "group_prob is not being applied on the key axis: moving the "
            "selected key from "
            f"{self.J0} to {self.J1} left {int((per_row <= 1e-3).sum())} of "
            f"{self.SEQ} query rows unmoved (per-row deltas {per_row})"
        )

    def test_group_prob_is_non_uniform_across_positions(self):
        """A constant `group_prob` would make the multiply a no-op up to scale.

        The test above cannot see that: it feeds `group_prob` by hand. This one
        asserts the matrix `GroupAttention` actually produces is structured.
        """
        keras.utils.set_random_seed(7)
        attn = GroupAttention(hidden_size=self.HIDDEN)
        shape = (self.BATCH, self.SEQ, self.HIDDEN)
        mask_shape = (self.BATCH, 1, self.SEQ)
        attn.build((shape, mask_shape, (self.BATCH, self.SEQ, self.SEQ)))

        x = keras.ops.convert_to_tensor(
            np.random.default_rng(12).standard_normal(
                (self.BATCH, self.SEQ, self.HIDDEN)).astype("float32")
        )
        mask = keras.ops.ones(mask_shape)
        prior = keras.ops.zeros((self.BATCH, self.SEQ, self.SEQ))

        group_prob, _break_prob = attn((x, mask, prior), training=False)
        gp = keras.ops.convert_to_numpy(group_prob)

        assert gp.shape == (self.BATCH, self.SEQ, self.SEQ), gp.shape
        assert float(np.std(gp)) > 0.0, (
            "group_prob is constant, so multiplying the attention weights by it "
            "is a global rescale and carries no constituency structure"
        )


# ---------------------------------------------------------------------
# include_top: the in-model task head (plan-2026-10-11T004932-b83935e1)
# ---------------------------------------------------------------------

#: Small enough that every arm below runs in about a second.
_HEAD_KW: Dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_layers": 2,
    "num_heads": 2,
    "intermediate_size": 64,
    "max_len": 16,
}
#: The factory-route transfer arms use the CUSTOM vocabulary of ``_HEAD_KW``
#: (64, not the class default): a same-vocabulary checkpoint must restore the
#: embedding by value (D-022 of plan-2026-10-11T004932-b83935e1; before it,
#: any non-default ``vocab_size`` dropped the embedding and these arms had to
#: keep the default to pass).
_XFER_KW: Dict[str, Any] = dict(_HEAD_KW)
_SEQ = 8
_CLF = NLPTaskConfig(
    name="clf", task_type=NLPTaskType.TEXT_CLASSIFICATION, num_classes=3
)
_NER = NLPTaskConfig(
    name="ner", task_type=NLPTaskType.TOKEN_CLASSIFICATION, num_classes=9
)
#: Same head class and weight shapes as ``_CLF``, a different task.
_SPAM = NLPTaskConfig(
    name="spam", task_type=NLPTaskType.SPAM_DETECTION, num_classes=3
)
#: The masked-LM projection: NOT backbone. It exists only headless, and a
#: task head replaces it (D-011 of plan-2026-10-11T004932-b83935e1).
_LM_HEAD = "lm_head_projection"
#: ``GroupAttention``, its norm and ``TreeMHA`` carry no explicit ``name=``,
#: so their path segment embeds the process-global Keras auto-name counter
#: (``group_attention_16``) and differs between two models built in one
#: process, head or no head. Measured; the same three stems are the package's
#: ``AUTONAME_STEMS`` entry in ``test_roundtrip_instrument_family.py``. The
#: counter is stripped from exactly these segments before paths are compared.
_AUTONAMED = re.compile(r"^(group_attention|layer_normalization|tree_mha)_\d+$")


def _canonical(path: str) -> str:
    return "/".join(_AUTONAMED.sub(r"\1", part) for part in path.split("/"))


def _np(tensor) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(tensor))


def _head_batch() -> Dict[str, np.ndarray]:
    ids = np.random.default_rng(0).integers(1, 64, (2, _SEQ)).astype("int32")
    mask = np.ones_like(ids)
    mask[1, 5:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _relative_paths(model: keras.Model) -> list:
    """Weight paths without the model-root segment (guide 8.3), canonical."""
    return sorted(_canonical(w.path.split("/", 1)[-1]) for w in model.weights)


def _backbone_hidden(model: TreeTransformer, batch: Dict[str, np.ndarray]) -> np.ndarray:
    """``last_hidden_state`` of ``model``'s backbone, whatever its top.

    Read through a headless twin rebuilt from the model's own config and given
    the model's own backbone weights, layer by layer; the twin's ``lm_head``
    has no counterpart in a headed model and does not feed
    ``last_hidden_state``.
    """
    config = model.get_config()
    config.update(include_top=False, task_config=None, head_config_overrides=None)
    config.pop("name", None)
    twin = TreeTransformer.from_config(config)
    twin(batch, training=False)
    for layer in twin.layers:
        if layer.name != _LM_HEAD:
            layer.set_weights(model.get_layer(layer.name).get_weights())
    return _np(twin(batch, training=False)["last_hidden_state"])


def _randomized(model: TreeTransformer, seed: int) -> TreeTransformer:
    """Overwrite every weight, so a restored value cannot match by init luck."""
    assert model.weights, "build the model before randomizing it"
    rng = np.random.default_rng(seed)
    for w in model.weights:
        w.assign(rng.normal(0.0, 0.1, size=tuple(w.shape)).astype(w.dtype))
    return model


class TestTreeTransformerIncludeTop:
    """The ``include_top`` / ``task_config`` / ``head_config_overrides`` knobs."""

    def test_head_output_shapes(self):
        """Bare logits for single-output heads, a dict for QA; nothing else."""
        batch = _head_batch()
        out = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_CLF)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, 3)

        out = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_NER)(
            batch, training=False)
        assert not isinstance(out, dict) and tuple(out.shape) == (2, _SEQ, 9)

        qa = TreeTransformer(**_HEAD_KW, include_top=True, task_config=NLPTaskConfig(
            name="qa", task_type=NLPTaskType.QUESTION_ANSWERING, num_classes=2))
        out = qa(batch, training=False)
        # No `logits` / `break_probs` / `last_hidden_state` beside the head's keys.
        assert set(out) == {"start_logits", "end_logits"}
        assert tuple(out["start_logits"].shape) == (2, _SEQ)

    def test_a_task_head_replaces_the_lm_head(self):
        """``build()`` must not create the LM head ``call()`` no longer runs."""
        headed = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert headed.lm_head is None
        assert not [w.path for w in headed.weights if _LM_HEAD in w.path]

        headless = TreeTransformer(**_HEAD_KW)
        headless(_head_batch(), training=False)
        assert headless.lm_head is not None
        assert sorted(w.path.split("/", 1)[-1] for w in headless.weights
                      if _LM_HEAD in w.path) == [
            f"{_LM_HEAD}/bias", f"{_LM_HEAD}/kernel"]

    def test_symbolic_output_matches_the_eager_one(self):
        """A symbolic call reports what ``call`` returns, headless or headed."""
        model = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_NER)
        symbolic = model(keras.Input(shape=(None,), dtype="int32"))
        assert not isinstance(symbolic, dict)
        assert tuple(symbolic.shape) == (None, None, 9)
        headless = TreeTransformer(**_HEAD_KW)
        symbolic = headless(keras.Input(shape=(None,), dtype="int32"))
        assert set(symbolic) == {"last_hidden_state", "logits", "break_probs"}

    def test_the_head_pools_under_the_pad_derived_mask(self):
        """Without an ``attention_mask`` the head sees ``input_ids != pad_token_id``.

        The wrapper this replaced gave the head no mask at all. Mean pooling
        makes the head's output depend on its mask, so the model's output is
        compared against its own head applied by hand to the backbone's hidden
        states, once under the pad-derived mask and once under none.
        """
        model = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_CLF,
                                head_config_overrides={"pooling_type": "mean"})
        ids = np.random.default_rng(1).integers(1, 64, (2, _SEQ)).astype("int32")
        ids[1, 5:] = 0
        out = _np(model(ids, training=False))

        hidden = _backbone_hidden(model, {"input_ids": ids})
        masked = _np(model.task_head(
            {"hidden_states": hidden,
             "attention_mask": (ids != 0).astype("float32")}, training=False
        )["logits"])
        unmasked = _np(model.task_head(
            {"hidden_states": hidden}, training=False)["logits"])
        np.testing.assert_allclose(out, masked, atol=1e-6, rtol=0)
        assert not np.allclose(masked[1], unmasked[1], atol=1e-6, rtol=0), (
            "fixture invalid: the head's mask makes no difference")

    def test_config_round_trip_with_a_head(self):
        """Every knob is serialized; ``task_config`` as a plain JSON dict."""
        import json

        overrides = {"pooling_type": "mean"}
        model = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_CLF,
                                head_config_overrides=overrides)
        config = model.get_config()
        assert config["include_top"] is True
        assert config["head_config_overrides"] == overrides
        assert isinstance(config["task_config"], dict)
        assert config["task_config"]["task_type"] == "text_classification"
        json.dumps(config)  # plain data, no enum or dataclass inside

        from_dict = TreeTransformer(**_HEAD_KW, include_top=True,
                                    task_config=dict(config["task_config"]),
                                    head_config_overrides=overrides)
        assert from_dict.get_config()["task_config"] == config["task_config"]

        rebuilt = TreeTransformer.from_config(config)
        assert rebuilt.get_config() == config
        assert rebuilt.task_head.pooling_type == "mean"

        batch = _head_batch()
        model(batch, training=False)
        rebuilt(batch, training=False)
        assert _relative_paths(rebuilt) == _relative_paths(model)
        rebuilt.set_weights(model.get_weights())
        np.testing.assert_array_equal(
            _np(rebuilt(batch, training=False)), _np(model(batch, training=False)))

    def test_a_config_without_the_head_keys_still_loads(self):
        """A config written before the knobs existed rebuilds the headless model."""
        config = TreeTransformer(**_HEAD_KW).get_config()
        for key in ("include_top", "task_config", "head_config_overrides"):
            config.pop(key)
        model = TreeTransformer.from_config(config)
        assert model.include_top is False and model.task_head is None
        assert model.lm_head is not None

    def test_keras_round_trip_restores_the_head_by_value(self, tmp_path):
        """Weights compared BEFORE the first call on the loaded model (guide 8.4)."""
        model = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_CLF)
        batch = _head_batch()
        before = _np(model(batch, training=False))
        saved = [_np(w).copy() for w in model.weights]
        head_saved = [_np(w).copy() for w in model.task_head.weights]
        assert head_saved, "the head has no weights to compare"

        path = str(tmp_path / "tree_head.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        assert len(loaded.weights) == len(saved)
        for s, w in zip(saved, loaded.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0,
                                       err_msg=f"{w.path} not restored")
        for s, w in zip(head_saved, loaded.task_head.weights):
            np.testing.assert_allclose(s, _np(w), atol=0.0, rtol=0.0)
        np.testing.assert_array_equal(_np(loaded(batch, training=False)), before)

    @pytest.mark.parametrize("include_top", [False, True])
    def test_explicit_build_matches_lazy_build(self, include_top):
        """``build()`` materializes exactly the tree ``call()`` runs (guide 8.1)."""
        kw = dict(_HEAD_KW, include_top=include_top,
                  task_config=_CLF if include_top else None)
        explicit = TreeTransformer(**kw)
        explicit.build((None, _SEQ))
        lazy = TreeTransformer(**kw)
        lazy(_head_batch(), training=False)
        assert _relative_paths(explicit) == _relative_paths(lazy)
        assert explicit.count_params() == lazy.count_params() > 0

    def test_a_disabled_head_builds_no_head_weights(self):
        """Anti-vacuity sibling of the parity arm (guide 8.3)."""
        headless = TreeTransformer(**_HEAD_KW)
        headless(_head_batch(), training=False)
        assert headless.task_head is None
        assert not [w.path for w in headless.weights if "task_head" in w.path]

        headed = TreeTransformer(**_HEAD_KW, include_top=True, task_config=_CLF)
        headed(_head_batch(), training=False)
        assert [w.path for w in headed.weights if "/task_head/" in w.path]

    def test_the_encoder_weight_paths_do_not_depend_on_the_head(self):
        """The backbone's paths are identical across tops (D-005 of the plan).

        The backbone is every weight except the two tops: ``lm_head_projection/``
        (headless only, the masked-LM projection) and ``task_head/`` (headed
        only). Measured on this config: 48 backbone paths, plus 2 for the LM
        head headless (50), plus 8 for a TEXT_CLASSIFICATION head headed (56).
        """
        def weights(**kw):
            m = TreeTransformer(**_HEAD_KW, name="tree", **kw)
            m(_head_batch(), training=False)
            return [_canonical(w.path) for w in m.weights]

        headless = weights()
        backbone = [p for p in headless if not p.startswith(f"tree/{_LM_HEAD}/")]
        assert (len(backbone), len(headless)) == (48, 50)
        for task in (_CLF, _NER):
            headed = weights(include_top=True, task_config=task)
            assert [p for p in headed if not p.startswith("tree/task_head/")] == (
                backbone), task.name
            assert all(p.startswith("tree/task_head/")
                       for p in headed[len(backbone):]), task.name
            assert len(headed) > len(backbone)

    def test_backbone_weights_transfer_across_heads(self, tmp_path):
        """A backbone trained under one head is reused, same weights, under another.

        The supported route is a local ``pretrained`` path, which copies the
        backbone by layer name and never the checkpoint's ``task_head``
        (D-007 of plan-2026-10-11T004932-b83935e1). Exercised into a
        TOKEN_CLASSIFICATION model and into a headless one, whose ``lm_head``
        has no source in a headed checkpoint and keeps its initialization.
        """
        source = create_tree_transformer("tiny", include_top=True,
                                         task_config=_CLF, **_XFER_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 1)
        path = str(tmp_path / "trained_under_clf.keras")
        source.save(path)
        reference = _backbone_hidden(source, batch)

        token = create_tree_transformer("tiny", include_top=True, task_config=_NER,
                                        pretrained=path, **_XFER_KW)
        headless = create_tree_transformer("tiny", pretrained=path, **_XFER_KW)

        np.testing.assert_array_equal(_backbone_hidden(token, batch), reference)
        np.testing.assert_array_equal(
            _np(headless(batch, training=False)["last_hidden_state"]), reference)
        assert tuple(token(batch, training=False).shape) == (2, _SEQ, 9)

        # Anti-vacuity: a fresh backbone does NOT reproduce the reference.
        fresh = create_tree_transformer("tiny", **_XFER_KW)
        assert not np.array_equal(
            _np(fresh(batch, training=False)["last_hidden_state"]), reference)

    def test_an_equal_shape_head_of_another_task_is_not_copied(self, tmp_path):
        """A SPAM head is not filled from a CLF checkpoint's head of equal shape.

        With the old ``skip_prefixes=()`` default the transfer matched the head
        by layer name and copied it silently, because the shapes agree
        (plan-2026-10-11T004932-b83935e1 D-007).
        """
        source = create_tree_transformer("tiny", include_top=True,
                                         task_config=_CLF, **_XFER_KW)
        batch = _head_batch()
        source(batch, training=False)
        _randomized(source, 2)
        path = str(tmp_path / "clf.keras")
        source.save(path)

        spam = create_tree_transformer("tiny", include_top=True, task_config=_SPAM,
                                       pretrained=path, **_XFER_KW)
        src_head = [_np(w) for w in source.task_head.weights]
        dst_head = [_np(w) for w in spam.task_head.weights]
        assert [a.shape for a in src_head] == [b.shape for b in dst_head], (
            "fixture invalid: the two heads must have equal weight shapes")
        assert not any(np.array_equal(a, b) for a, b in zip(src_head, dst_head))
        np.testing.assert_array_equal(
            _backbone_hidden(spam, batch), _backbone_hidden(source, batch))

    def test_pretrained_headless_checkpoint_into_a_headed_model(self, tmp_path):
        """A headless (MLM) checkpoint restores the backbone of a headed model.

        The checkpoint's ``lm_head`` has no target there and is left out.
        """
        encoder = create_tree_transformer("tiny", **_XFER_KW)
        batch = _head_batch()
        encoder(batch, training=False)
        _randomized(encoder, 3)
        expected = _np(encoder(batch, training=False)["last_hidden_state"])
        path = str(tmp_path / "encoder.keras")
        encoder.save(path)

        headed = create_tree_transformer("tiny", include_top=True, task_config=_CLF,
                                         pretrained=path, **_XFER_KW)
        assert headed.lm_head is None
        np.testing.assert_array_equal(_backbone_hidden(headed, batch), expected)

    @pytest.mark.parametrize("headed", [True, False], ids=["headed", "headless"])
    def test_a_same_custom_vocabulary_restores_the_embedding(self, tmp_path, headed):
        """Reviewer W2: a non-default but EQUAL vocabulary is not a vocab change.

        Before D-022 ``from_variant`` compared ``vocab_size`` with
        ``DEFAULT_VOCAB_SIZE`` and skipped ``embedding`` (and ``lm_head``) for
        any custom vocabulary, so this checkpoint's embedding was dropped.
        """
        encoder = create_tree_transformer("tiny", **_HEAD_KW)
        encoder(_head_batch(), training=False)
        _randomized(encoder, 5)
        path = str(tmp_path / "encoder_v64.keras")
        encoder.save(path)

        head = {"include_top": True, "task_config": _CLF} if headed else {}
        target = create_tree_transformer("tiny", pretrained=path, **_HEAD_KW, **head)
        names = ["embedding", "pos_encoding", "block_0", "block_1", "final_norm"]
        if not headed:
            names.append("lm_head_projection")
        for name in names:
            for a, b in zip(target.get_layer(name).weights,
                            encoder.get_layer(name).weights):
                np.testing.assert_array_equal(_np(a), _np(b), err_msg=a.path)

    @pytest.mark.parametrize("headed", [True, False], ids=["headed", "headless"])
    def test_a_different_vocabulary_skips_only_the_vocab_layers(self, tmp_path, headed):
        """A checkpoint of another vocabulary: everything but embedding/LM head loads."""
        encoder = create_tree_transformer("tiny", **_HEAD_KW)
        encoder(_head_batch(), training=False)
        _randomized(encoder, 4)
        path = str(tmp_path / "encoder_v64.keras")
        encoder.save(path)

        head = {"include_top": True, "task_config": _CLF} if headed else {}
        target = create_tree_transformer(
            "tiny", pretrained=path, **dict(_HEAD_KW, vocab_size=80), **head)
        for name in ["pos_encoding", "block_0", "block_1", "final_norm"]:
            for a, b in zip(target.get_layer(name).weights,
                            encoder.get_layer(name).weights):
                np.testing.assert_array_equal(_np(a), _np(b), err_msg=a.path)
        assert tuple(target.get_layer("embedding").weights[0].shape) == (80, 32)

    @pytest.mark.parametrize("headed", [True, False], ids=["headed", "headless"])
    @pytest.mark.parametrize("override", [
        {"hidden_size": 64}, {"num_layers": 3},
    ], ids=["hidden_size", "deeper"])
    def test_a_backbone_that_does_not_fit_raises(self, tmp_path, headed, override):
        """The D-020 rule: a backbone that does not fit is refused, never half-loaded."""
        encoder = create_tree_transformer("tiny", **_HEAD_KW)
        encoder(_head_batch(), training=False)
        path = str(tmp_path / "encoder.keras")
        encoder.save(path)

        head = {"include_top": True, "task_config": _CLF} if headed else {}
        with pytest.raises(ValueError, match="unloaded|[Ss]hape mismatch"):
            create_tree_transformer(
                "tiny", pretrained=path, **dict(_HEAD_KW, **override), **head)

    def test_include_top_without_task_config_raises(self):
        with pytest.raises(ValueError, match="include_top=True needs a task_config"):
            TreeTransformer(**_HEAD_KW, include_top=True)

    @pytest.mark.parametrize("dead", [
        {"task_config": _CLF},
        {"head_config_overrides": {"pooling_type": "mean"}},
    ], ids=["task_config", "head_config_overrides"])
    def test_head_knobs_without_include_top_raise(self, dead):
        with pytest.raises(ValueError, match="only take effect with include_top=True"):
            TreeTransformer(**_HEAD_KW, **dead)

    def test_pretrained_true_still_raises_with_a_head(self):
        with pytest.raises(NotImplementedError):
            create_tree_transformer("tiny", pretrained=True, include_top=True,
                                    task_config=_CLF)
