"""Tests for the AsciiCliffordBert arm."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.ascii_char import VOCAB_SIZE
from dl_techniques.models.embeddings_experimental.ascii_bert import AsciiBert
from dl_techniques.models.embeddings_experimental.ascii_clifford_bert import (
    AsciiCliffordBert,
    create_ascii_clifford_bert,
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
        shifts=[1, 2],
        max_position_embeddings=SEQ_LEN * 2,
    )
    config.update(overrides)
    model = AsciiCliffordBert(**config)
    model.build((None, SEQ_LEN))
    return model


class TestConstruction:
    def test_block_type_is_clifford(self):
        assert tiny().block_type == "clifford"

    def test_there_is_no_attention_layer_anywhere(self):
        """The arm's defining property: attention-free."""
        names = [w.path.lower() for w in tiny().weights]
        assert not [n for n in names if "attention" in n and "pool" not in n]

    def test_global_context_is_off_by_default(self):
        assert tiny().use_global_context is False

    def test_context_kernel_defaults_to_seven_not_the_layers_three(self):
        """A character-level encoder needs a wider span than K=3 gives."""
        assert AsciiCliffordBert(hidden_size=32, num_layers=2).context_kernel_size == 7

    def test_forward_pass_shapes(self, token_ids):
        out = tiny()({"input_ids": token_ids})
        assert tuple(out["last_hidden_state"].shape) == (2, SEQ_LEN, 32)
        assert tuple(out["pooled_output"].shape) == (2, 32)


class TestReceptiveField:
    def test_matches_the_two_convolutions_per_block_formula(self):
        assert tiny(num_layers=2, context_kernel_size=7).receptive_field == 25
        assert tiny(num_layers=4, context_kernel_size=3).receptive_field == 17

    def test_a_short_span_is_warned_about(self, caplog):
        """A span shorter than the sequence must be visible in the run log."""
        import logging

        with caplog.at_level(logging.WARNING, logger="dl"):
            AsciiCliffordBert(
                hidden_size=32,
                num_layers=2,
                context_kernel_size=3,
                max_position_embeddings=512,
            )
        assert any("Token-mixing span" in r.message for r in caplog.records)

    def test_no_warning_when_the_span_covers_the_sequence(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="dl"):
            AsciiCliffordBert(
                hidden_size=32,
                num_layers=4,
                context_kernel_size=7,
                max_position_embeddings=32,
            )
        assert not [r for r in caplog.records if "Token-mixing span" in r.message]


class TestVariants:
    def test_every_variant_builds_and_runs(self, token_ids):
        for variant in AsciiCliffordBert.MODEL_VARIANTS:
            model = AsciiCliffordBert.from_variant(
                variant, max_position_embeddings=SEQ_LEN * 2
            )
            model.build((None, SEQ_LEN))
            out = model({"input_ids": token_ids})
            assert np.isfinite(
                keras.ops.convert_to_numpy(out["pooled_output"])
            ).all(), variant
            keras.backend.clear_session()

    def test_the_ladder_is_depth_and_width_matched_to_the_baseline_arm(self):
        """The size axis must line up, or the comparison is not controlled."""
        for variant in ("tiny", "small", "base"):
            clifford = AsciiCliffordBert.MODEL_VARIANTS[variant]
            transformer = AsciiBert.MODEL_VARIANTS[variant]
            assert clifford["hidden_size"] == transformer["hidden_size"], variant
            assert clifford["num_layers"] == transformer["num_layers"], variant

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            AsciiCliffordBert.from_variant("enormous")


class TestFactory:
    def test_returns_the_requested_variant(self):
        model = create_ascii_clifford_bert("tiny", max_position_embeddings=32)
        assert isinstance(model, AsciiCliffordBert)

    def test_pretrained_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="No pretrained weights"):
            create_ascii_clifford_bert("tiny", pretrained=True)


class TestKnobsAreWired:
    """A configuration knob that changes nothing is a dead knob."""

    def test_shifts_changes_the_weight_shape_signature(self):
        """Structural knob: pinned on shapes, not on output values.

        Different shapes consume different RNG draws, so an output-difference
        assertion here would be satisfied by initialization luck alone.
        """
        def signature(**kwargs):
            model = tiny(**kwargs)
            sig = sorted((w.path.split("/")[-1], tuple(w.shape)) for w in model.weights)
            keras.backend.clear_session()
            return sig

        assert signature(shifts=[1, 2]) != signature(shifts=[1, 2, 4, 8])

    def test_context_kernel_size_changes_the_weight_shape_signature(self):
        def signature(**kwargs):
            model = tiny(**kwargs)
            sig = sorted(tuple(w.shape) for w in model.weights)
            keras.backend.clear_session()
            return sig

        assert signature(context_kernel_size=3) != signature(context_kernel_size=7)

    def test_cli_mode_changes_the_output(self, token_ids):
        """Value knob: same shapes, so the output must move."""
        outputs = {}
        for mode in ("full", "inner"):
            model = tiny(cli_mode=mode, layer_scale_init=1.0)
            keras.utils.set_random_seed(0)
            outputs[mode] = keras.ops.convert_to_numpy(
                model({"input_ids": token_ids}, training=False)["last_hidden_state"]
            )
            keras.backend.clear_session()
        assert outputs["full"].shape == outputs["inner"].shape
        assert not np.allclose(outputs["full"], outputs["inner"], atol=1e-6)


class TestSerialization:
    def test_config_round_trip_preserves_the_arms_own_arguments(self):
        model = tiny(shifts=[1, 2, 4], cli_mode="wedge", context_kernel_size=5)
        config = model.get_config()
        assert "block_type" not in config
        assert "block_config" not in config

        restored = AsciiCliffordBert.from_config(config)
        assert restored.shifts == [1, 2, 4]
        assert restored.cli_mode == "wedge"
        assert restored.context_kernel_size == 5

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
