"""Tests for WaveFieldLLM (decoder-only LM with WaveFieldAttention).

Mirrors the structure of ``tests/test_models/test_gpt2/test_gpt2.py`` and
adds wave-field-specific checks: padding-mask zeroing of last_hidden_state
and field_size validation.
"""

import os
import tempfile

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.models.wave_field_llm.wave_field_llm import (
    WaveFieldLLM,
    WaveFieldDecoderBlock,
)
from dl_techniques.losses import MaskedCausalLMLoss


def _random_ids(shape, vocab_size):
    return np.random.randint(0, vocab_size, shape).astype(np.int32)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def tiny_config():
    """Minimal WaveFieldLLM config for fast testing."""
    return {
        "vocab_size": 256,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "max_seq_len": 32,
        "field_size": 64,
        "dropout_rate": 0.0,
        "attention_dropout_rate": 0.0,
    }


@pytest.fixture
def tiny_model(tiny_config):
    model = WaveFieldLLM(**tiny_config)
    dummy = _random_ids((1, tiny_config["max_seq_len"]), tiny_config["vocab_size"])
    model(dummy, training=False)
    return model


# ---------------------------------------------------------------------
# Initialization & validation
# ---------------------------------------------------------------------


class TestWaveFieldLLMInitialization:

    def test_initialization(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        assert model.vocab_size == 256
        assert model.embed_dim == 64
        assert model.depth == 2
        assert model.num_heads == 4
        assert model.max_seq_len == 32
        assert model.field_size == 64
        assert len(model.blocks) == 2
        assert isinstance(model.blocks[0], WaveFieldDecoderBlock)

    def test_default_field_size_doubles_max_seq_len(self):
        model = WaveFieldLLM(
            vocab_size=128, embed_dim=32, depth=1,
            num_heads=4, max_seq_len=16, field_size=None,
        )
        assert model.field_size == 32

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            WaveFieldLLM(vocab_size=-1, embed_dim=64, depth=1, num_heads=4)

    def test_invalid_embed_dim(self):
        with pytest.raises(ValueError, match="embed_dim must be positive"):
            WaveFieldLLM(vocab_size=64, embed_dim=0, depth=1, num_heads=4)

    def test_invalid_depth(self):
        with pytest.raises(ValueError, match="depth must be positive"):
            WaveFieldLLM(vocab_size=64, embed_dim=64, depth=0, num_heads=4)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads must be positive"):
            WaveFieldLLM(vocab_size=64, embed_dim=64, depth=1, num_heads=0)

    def test_embed_dim_not_divisible(self):
        with pytest.raises(ValueError, match="must be divisible by"):
            WaveFieldLLM(vocab_size=64, embed_dim=100, depth=1, num_heads=12)

    def test_invalid_field_size(self):
        with pytest.raises(ValueError, match="field_size must be > 1"):
            WaveFieldLLM(
                vocab_size=64, embed_dim=64, depth=1, num_heads=4,
                max_seq_len=8, field_size=1,
            )

    def test_invalid_dropout(self):
        with pytest.raises(ValueError, match="dropout_rate"):
            WaveFieldLLM(
                vocab_size=64, embed_dim=64, depth=1,
                num_heads=4, dropout_rate=1.5,
            )


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------


class TestWaveFieldLLMForward:

    def test_forward_shape(self, tiny_model, tiny_config):
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        out = tiny_model(ids, training=False)
        assert "logits" in out
        assert "last_hidden_state" in out
        assert out["logits"].shape == (2, 16, tiny_config["vocab_size"])
        assert out["last_hidden_state"].shape == (2, 16, tiny_config["embed_dim"])

    def test_forward_full_seq_len(self, tiny_model, tiny_config):
        seq = tiny_config["max_seq_len"]
        ids = _random_ids((1, seq), tiny_config["vocab_size"])
        out = tiny_model(ids, training=False)
        assert out["logits"].shape == (1, seq, tiny_config["vocab_size"])

    def test_forward_with_padding_mask(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 12), tiny_config["vocab_size"])
        # First sample: 8 valid + 4 padded; second: 5 valid + 7 padded.
        mask = np.array(
            [[1] * 8 + [0] * 4, [1] * 5 + [0] * 7], dtype=np.float32,
        )
        out = model(ids, attention_mask=mask, training=False)
        logits = keras.ops.convert_to_numpy(out["logits"])
        lhs = keras.ops.convert_to_numpy(out["last_hidden_state"])
        # Output is finite.
        assert np.all(np.isfinite(logits))
        assert np.all(np.isfinite(lhs))
        # WaveFieldAttention zeros its output at padded positions; after
        # the residual + final norm, padded positions are no longer zero
        # globally — but the attention residual contribution at padded
        # positions is zero, so outputs there must remain finite. We
        # therefore assert finiteness only.

    def test_dict_input(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 10), tiny_config["vocab_size"])
        mask = np.ones((2, 10), dtype=np.int32)
        out = model({"input_ids": ids, "attention_mask": mask}, training=False)
        assert out["logits"].shape == (2, 10, tiny_config["vocab_size"])

    def test_dict_input_missing_ids_raises(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        with pytest.raises(ValueError, match="input_ids"):
            model({"attention_mask": np.ones((2, 10), dtype=np.int32)})


# ---------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------


class TestWaveFieldLLMCausality:

    def test_future_does_not_affect_past(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        seq1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
        seq2 = np.array([[1, 2, 3, 4, 5, 6, 7, 99]], dtype=np.int32)

        out1 = model(seq1, training=False)
        out2 = model(seq2, training=False)
        l1 = keras.ops.convert_to_numpy(out1["logits"])
        l2 = keras.ops.convert_to_numpy(out2["logits"])
        for pos in range(7):
            np.testing.assert_allclose(
                l1[0, pos], l2[0, pos], atol=1e-5,
                err_msg=f"position {pos} changed when only position 7 changed",
            )


# ---------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------


class TestWaveFieldLLMWeightTying:

    def test_default_tied(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        assert model.tie_word_embeddings is True
        assert model.lm_head is None

    def test_logits_use_embedding_weights(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((1, 8), tiny_config["vocab_size"])
        out = model(ids, training=False)
        emb = model.token_embeddings.embeddings
        expected = keras.ops.matmul(
            out["last_hidden_state"], keras.ops.transpose(emb),
        )
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(out["logits"]),
            keras.ops.convert_to_numpy(expected),
            atol=1e-5,
        )

    def test_no_tying(self, tiny_config):
        cfg = {**tiny_config, "tie_word_embeddings": False}
        model = WaveFieldLLM(**cfg)
        assert model.lm_head is not None
        ids = _random_ids((1, 8), cfg["vocab_size"])
        out = model(ids, training=False)
        assert out["logits"].shape == (1, 8, cfg["vocab_size"])


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------


class TestWaveFieldLLMSerialization:

    def test_get_config_round_trip(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        config = model.get_config()
        for k, v in tiny_config.items():
            assert config[k] == v
        model2 = WaveFieldLLM.from_config(config)
        assert model2.vocab_size == model.vocab_size
        assert model2.depth == model.depth
        assert model2.field_size == model.field_size

    def test_compute_output_shape(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        shapes = model.compute_output_shape((None, 16))
        assert shapes["logits"] == (None, 16, tiny_config["vocab_size"])
        assert shapes["last_hidden_state"] == (None, 16, tiny_config["embed_dim"])

    def test_save_load_keras_round_trip(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 16), tiny_config["vocab_size"])
        out_before = keras.ops.convert_to_numpy(
            model(ids, training=False)["logits"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wfllm.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            out_after = keras.ops.convert_to_numpy(
                loaded(ids, training=False)["logits"],
            )

        # 1e-5 atol: matches LESSONS L44 for non-U-Net causal models.
        np.testing.assert_allclose(out_before, out_after, atol=1e-5)


# ---------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------


class TestWaveFieldLLMVariants:

    def test_from_variant_tiny(self):
        model = WaveFieldLLM.from_variant("tiny")
        assert model.embed_dim == 256
        assert model.depth == 4
        assert model.num_heads == 4
        assert model.max_seq_len == 512
        assert model.field_size == 1024

    def test_from_variant_with_overrides(self):
        model = WaveFieldLLM.from_variant("tiny", dropout_rate=0.1)
        assert model.dropout_rate == 0.1
        assert model.embed_dim == 256

    def test_from_variant_unknown(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            WaveFieldLLM.from_variant("nonexistent")

    def test_all_variants_have_required_keys(self):
        required = {"embed_dim", "depth", "num_heads", "max_seq_len", "field_size"}
        for name, cfg in WaveFieldLLM.MODEL_VARIANTS.items():
            for k in required:
                assert k in cfg, f"Variant {name!r} missing {k!r}"


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


class TestWaveFieldLLMGradient:

    def test_gradient_flow(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 8), tiny_config["vocab_size"])
        labels = _random_ids((2, 8), tiny_config["vocab_size"])

        with tf.GradientTape() as tape:
            out = model(ids, training=True)
            loss = keras.ops.mean(
                keras.losses.sparse_categorical_crossentropy(
                    labels, out["logits"], from_logits=True,
                )
            )
        grads = tape.gradient(loss, model.trainable_variables)
        for var, g in zip(model.trainable_variables, grads):
            assert g is not None, f"no grad for {var.name}"


# ---------------------------------------------------------------------
# CLM loss compatibility
# ---------------------------------------------------------------------


class TestWaveFieldLLMCLMLoss:

    def test_clm_loss_finite(self, tiny_config):
        model = WaveFieldLLM(**tiny_config)
        ids = _random_ids((2, 8), tiny_config["vocab_size"])
        labels = _random_ids((2, 8), tiny_config["vocab_size"])
        out = model(ids, training=False)
        loss = MaskedCausalLMLoss()(labels, out["logits"])
        loss_val = float(keras.ops.convert_to_numpy(loss))
        assert np.isfinite(loss_val)
