"""Tests for the AsciiConvNextBert arm."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.ascii_char import VOCAB_SIZE
from dl_techniques.models.embeddings_experimental.ascii_bert import AsciiBert
from dl_techniques.models.embeddings_experimental.ascii_clifford_bert import (
    AsciiCliffordBert,
)
from dl_techniques.models.embeddings_experimental.ascii_convnext_bert import (
    AsciiConvNextBert,
    create_ascii_convnext_bert,
)

SEQ_LEN = 24


@pytest.fixture
def token_ids():
    rng = np.random.default_rng(5)
    return rng.integers(6, VOCAB_SIZE, size=(2, SEQ_LEN)).astype("int32")


def tiny(**overrides):
    config = dict(
        hidden_size=32,
        num_layers=2,
        kernel_size=3,
        max_position_embeddings=SEQ_LEN * 2,
    )
    config.update(overrides)
    model = AsciiConvNextBert(**config)
    model.build((None, SEQ_LEN))
    return model


class TestConstruction:
    def test_block_type_is_convnext(self):
        assert tiny().block_type == "convnext"

    def test_there_is_no_attention_layer_anywhere(self):
        """The arm's defining property: attention-free."""
        names = [w.path.lower() for w in tiny().weights]
        assert not [n for n in names if "attention" in n and "pool" not in n]

    def test_layer_scale_starts_at_one_not_at_1e_minus_5(self):
        """Unlike the Clifford arm, this block contributes at full magnitude."""
        assert tiny().gamma_initial_value == 1.0

    def test_forward_pass_shapes(self, token_ids):
        out = tiny()({"input_ids": token_ids})
        assert tuple(out["last_hidden_state"].shape) == (2, SEQ_LEN, 32)
        assert tuple(out["pooled_output"].shape) == (2, 32)

    def test_the_sequence_is_convolved_along_length_only(self, token_ids):
        """The (1, K) kernel must not mix across the lifted height axis.

        The lift introduces a singleton height, so a kernel wider than 1 there
        would convolve across padding that is not part of the sequence at all.
        """
        model = tiny(kernel_size=5)
        depthwise = [
            w for w in model.weights if "depthwise" in w.path.lower()
        ]
        assert depthwise, "no depthwise kernel found"
        # DepthwiseConv2D kernel is (kh, kw, channels, depth_multiplier).
        assert depthwise[0].shape[0] == 1, depthwise[0].shape
        assert depthwise[0].shape[1] == 5, depthwise[0].shape


class TestReceptiveField:
    def test_matches_the_one_convolution_per_block_formula(self):
        assert tiny(num_layers=2, kernel_size=7).receptive_field == 13
        assert tiny(num_layers=4, kernel_size=3).receptive_field == 9

    def test_the_span_is_half_the_clifford_arms_at_equal_depth_and_kernel(self):
        """Matching on kernel_size does NOT match the two arms on span."""
        conv = AsciiConvNextBert(
            hidden_size=32, num_layers=4, kernel_size=7,
            max_position_embeddings=64,
        )
        clifford = AsciiCliffordBert(
            hidden_size=32, num_layers=4, context_kernel_size=7,
            max_position_embeddings=64,
        )
        assert clifford.receptive_field == 2 * conv.receptive_field - 1

    def test_a_short_span_is_warned_about(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="dl"):
            AsciiConvNextBert(
                hidden_size=32, num_layers=2, kernel_size=3,
                max_position_embeddings=512,
            )
        assert any("Token-mixing span" in r.message for r in caplog.records)


class TestVariants:
    def test_every_variant_builds_and_runs(self, token_ids):
        for variant in AsciiConvNextBert.MODEL_VARIANTS:
            model = AsciiConvNextBert.from_variant(
                variant, max_position_embeddings=SEQ_LEN * 2
            )
            model.build((None, SEQ_LEN))
            out = model({"input_ids": token_ids})
            assert np.isfinite(
                keras.ops.convert_to_numpy(out["pooled_output"])
            ).all(), variant
            keras.backend.clear_session()

    def test_the_ladder_is_depth_and_width_matched_to_the_other_arms(self):
        """All three arms must line up on the size axis."""
        for variant in ("tiny", "small", "base"):
            conv = AsciiConvNextBert.MODEL_VARIANTS[variant]
            transformer = AsciiBert.MODEL_VARIANTS[variant]
            clifford = AsciiCliffordBert.MODEL_VARIANTS[variant]
            assert conv["hidden_size"] == transformer["hidden_size"], variant
            assert conv["num_layers"] == transformer["num_layers"], variant
            assert conv["hidden_size"] == clifford["hidden_size"], variant
            assert conv["num_layers"] == clifford["num_layers"], variant

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            AsciiConvNextBert.from_variant("enormous")


class TestFactory:
    def test_returns_the_requested_variant(self):
        model = create_ascii_convnext_bert("tiny", max_position_embeddings=32)
        assert isinstance(model, AsciiConvNextBert)

    def test_pretrained_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="No pretrained weights"):
            create_ascii_convnext_bert("tiny", pretrained=True)


class TestKnobsAreWired:
    def test_kernel_size_changes_the_weight_shape_signature(self):
        def signature(**kwargs):
            model = tiny(**kwargs)
            sig = sorted(tuple(w.shape) for w in model.weights)
            keras.backend.clear_session()
            return sig

        assert signature(kernel_size=3) != signature(kernel_size=7)

    def test_use_gamma_changes_the_weight_count(self):
        with_gamma = tiny(use_gamma=True)
        n_with = len(with_gamma.weights)
        keras.backend.clear_session()
        without = tiny(use_gamma=False)
        n_without = len(without.weights)
        keras.backend.clear_session()
        assert n_with > n_without

    def test_gamma_initial_value_changes_the_output(self, token_ids):
        outputs = {}
        for gamma in (1.0, 0.01):
            model = tiny(gamma_initial_value=gamma)
            outputs[gamma] = keras.ops.convert_to_numpy(
                model({"input_ids": token_ids}, training=False)[
                    "last_hidden_state"
                ]
            )
            keras.backend.clear_session()
        assert not np.allclose(outputs[1.0], outputs[0.01], atol=1e-6)


class TestSerialization:
    def test_config_round_trip_preserves_the_arms_own_arguments(self):
        model = tiny(kernel_size=5, block_activation="relu", use_gamma=False)
        config = model.get_config()
        assert "block_type" not in config
        assert "block_config" not in config

        restored = AsciiConvNextBert.from_config(config)
        assert restored.kernel_size == 5
        assert restored.block_activation == "relu"
        assert restored.use_gamma is False

    def test_keras_round_trip_preserves_outputs(self, token_ids):
        model = tiny()
        expected = keras.ops.convert_to_numpy(
            model({"input_ids": token_ids}, training=False)["pooled_output"]
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "m.keras")
            model.save(path)
            actual = keras.ops.convert_to_numpy(
                keras.models.load_model(path)(
                    {"input_ids": token_ids}, training=False
                )["pooled_output"]
            )
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)


class TestGradientFlow:
    def test_every_trainable_weight_receives_a_gradient(self, token_ids):
        import tensorflow as tf

        model = tiny()
        with tf.GradientTape() as tape:
            pooled = model({"input_ids": token_ids}, training=True)["pooled_output"]
            loss = keras.ops.mean(keras.ops.square(pooled))
        grads = tape.gradient(loss, model.trainable_weights)
        dead = [w.path for w, g in zip(model.trainable_weights, grads) if g is None]
        assert dead == []
