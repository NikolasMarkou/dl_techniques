"""Tests for the AsciiConvNextV2Bert arm.

The V1 and V2 arms differ by exactly one thing -- Global Response
Normalization -- so most of this file is about that difference, and about the
padding hazard GRN introduces.
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.ascii_char import PAD_ID, VOCAB_SIZE
from dl_techniques.models.embeddings_experimental.ascii_bert import AsciiBert
from dl_techniques.models.embeddings_experimental.ascii_convnext_bert import (
    AsciiConvNextBert,
)
from dl_techniques.models.embeddings_experimental.ascii_convnext_v2_bert import (
    AsciiConvNextV2Bert,
    create_ascii_convnext_v2_bert,
)
from dl_techniques.models.embeddings_experimental.shared import EmbeddingEncoder

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
    model = AsciiConvNextV2Bert(**config)
    model.build((None, SEQ_LEN))
    return model


class TestConstruction:
    def test_block_type_is_convnext_v2(self):
        assert tiny().block_type == "convnext_v2"

    def test_there_is_no_attention_layer_anywhere(self):
        names = [w.path.lower() for w in tiny().weights]
        assert not [n for n in names if "attention" in n and "pool" not in n]

    def test_forward_pass_shapes(self, token_ids):
        out = tiny()({"input_ids": token_ids})
        assert tuple(out["last_hidden_state"].shape) == (2, SEQ_LEN, 32)
        assert tuple(out["pooled_output"].shape) == (2, 32)


class TestGrnIsTheOnlyDifferenceFromV1:
    """The arms are a matched pair around exactly one variable."""

    def test_v2_has_grn_weights_and_v1_has_none(self):
        v2 = [w.path for w in tiny().weights if "response" in w.path.lower()]
        keras.backend.clear_session()
        v1_model = AsciiConvNextBert(
            hidden_size=32, num_layers=2, kernel_size=3,
            max_position_embeddings=SEQ_LEN * 2,
        )
        v1_model.build((None, SEQ_LEN))
        v1 = [w.path for w in v1_model.weights if "response" in w.path.lower()]
        keras.backend.clear_session()
        assert v2, "V2 must carry GRN weights"
        assert not v1, "V1 must carry none"

    def test_v2_is_larger_than_v1_by_the_grn_parameters_only(self):
        def params(cls):
            model = cls(
                hidden_size=32, num_layers=2, kernel_size=3,
                max_position_embeddings=SEQ_LEN * 2,
            )
            model.build((None, SEQ_LEN))
            n = model.count_params()
            grn = sum(
                int(np.prod(w.shape))
                for w in model.weights
                if "response" in w.path.lower()
            )
            keras.backend.clear_session()
            return n, grn

        n_v1, grn_v1 = params(AsciiConvNextBert)
        n_v2, grn_v2 = params(AsciiConvNextV2Bert)
        assert grn_v1 == 0
        assert n_v2 - n_v1 == grn_v2, (
            "V2 should differ from V1 by exactly the GRN parameters; a larger "
            "gap means something else changed too and the pair is no longer "
            "a controlled comparison"
        )

    def test_the_two_arms_share_the_size_ladder(self):
        for variant in ("tiny", "small", "base"):
            v2 = AsciiConvNextV2Bert.MODEL_VARIANTS[variant]
            v1 = AsciiConvNextBert.MODEL_VARIANTS[variant]
            transformer = AsciiBert.MODEL_VARIANTS[variant]
            assert v2["hidden_size"] == v1["hidden_size"] == transformer["hidden_size"]
            assert v2["num_layers"] == v1["num_layers"] == transformer["num_layers"]
            assert v2["kernel_size"] == v1["kernel_size"]


class TestGrnMakesPadLengthMatterAfterTraining:
    """Pinned known behaviour, and a trap a smoke test would miss.

    GRN scores each channel by its L2 magnitude over the SEQUENCE, so whatever
    sits in the padded region enters every real position's normalizer.

    The wrapper zeroes masked positions and GRN's score is an L2 SUM --
    ``sqrt(sum(x**2) + eps)`` -- so exact zeros contribute nothing. At
    initialization every bias is zero, so the padded region stays exactly zero
    through conv_1, the norm, conv_2 and the activation (each verified), and
    pad length is exactly inert.

    Training moves the biases off zero, the padded region stops being zero, and
    GRN's sum picks it up. MEASURED on a 6-token prefix, ``hidden_size=32``,
    2 blocks, ``K=3``, pad-8 against pad-12:

        arm          at initialization    with non-zero biases
        convnext     0.000e+00            0.000e+00
        convnext_v2  0.000e+00            3.215e-03

    So a probe of a freshly constructed model reports this arm as padding-safe
    and is wrong about the model that is actually trained. Same shape as
    LayerScale hiding the Clifford arm's boundary effect.
    """

    @staticmethod
    def _pad_length_delta(block_type, *, nonzero_biases):
        rng = np.random.default_rng(11)
        prefix = rng.integers(6, VOCAB_SIZE, size=(1, 6)).astype("int32")

        model = EmbeddingEncoder(
            hidden_size=32,
            num_layers=2,
            block_type=block_type,
            block_config={"kernel_size": 3},
            max_position_embeddings=64,
            hidden_dropout_rate=0.0,
        )
        model.build((None, 6))

        if nonzero_biases:
            for weight in model.weights:
                if "bias" in weight.path.lower():
                    weight.assign(keras.ops.ones_like(weight) * 0.1)

        def states(length):
            ids = (
                prefix
                if length == 6
                else np.concatenate(
                    [prefix, np.full((1, length - 6), PAD_ID, dtype="int32")],
                    axis=1,
                )
            )
            return keras.ops.convert_to_numpy(
                model({"input_ids": ids})["last_hidden_state"]
            )[0, :6]

        delta = float(np.abs(states(8) - states(12)).max())
        keras.backend.clear_session()
        return delta

    def test_pad_length_is_inert_at_initialization(self):
        assert self._pad_length_delta(
            "convnext_v2", nonzero_biases=False
        ) == pytest.approx(0.0, abs=1e-9)

    def test_pad_length_moves_the_output_once_biases_are_non_zero(self):
        delta = self._pad_length_delta("convnext_v2", nonzero_biases=True)
        assert delta > 1e-4, (
            "expected GRN to make pad length matter once biases leave zero; "
            f"got {delta:.3e}. If this is now zero, GRN has stopped reducing "
            "over the sequence and this arm's packing rationale needs rewriting."
        )

    def test_the_v1_arm_is_the_control_and_stays_inert_either_way(self):
        """Isolates the effect to GRN rather than to the convolution."""
        assert self._pad_length_delta(
            "convnext", nonzero_biases=False
        ) == pytest.approx(0.0, abs=1e-9)
        assert self._pad_length_delta(
            "convnext", nonzero_biases=True
        ) == pytest.approx(0.0, abs=1e-9)


class TestReceptiveField:
    def test_uses_the_one_convolution_per_block_formula(self):
        assert tiny(num_layers=2, kernel_size=7).receptive_field == 13
        assert tiny(num_layers=4, kernel_size=3).receptive_field == 9


class TestVariants:
    def test_every_variant_builds_and_runs(self, token_ids):
        for variant in AsciiConvNextV2Bert.MODEL_VARIANTS:
            model = AsciiConvNextV2Bert.from_variant(
                variant, max_position_embeddings=SEQ_LEN * 2
            )
            model.build((None, SEQ_LEN))
            out = model({"input_ids": token_ids})
            assert np.isfinite(
                keras.ops.convert_to_numpy(out["pooled_output"])
            ).all(), variant
            keras.backend.clear_session()

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            AsciiConvNextV2Bert.from_variant("enormous")


class TestFactory:
    def test_returns_the_requested_variant(self):
        assert isinstance(
            create_ascii_convnext_v2_bert("tiny", max_position_embeddings=32),
            AsciiConvNextV2Bert,
        )

    def test_pretrained_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="No pretrained weights"):
            create_ascii_convnext_v2_bert("tiny", pretrained=True)


class TestSerialization:
    def test_config_round_trip(self):
        model = tiny(kernel_size=5, block_activation="relu")
        config = model.get_config()
        assert "block_type" not in config
        restored = AsciiConvNextV2Bert.from_config(config)
        assert restored.kernel_size == 5
        assert restored.block_activation == "relu"

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

    def test_the_grn_weights_receive_a_gradient(self, token_ids):
        """GRN is the arm's whole reason to exist; it must actually train."""
        import tensorflow as tf

        model = tiny()
        grn = [w for w in model.trainable_weights if "response" in w.path.lower()]
        assert grn
        with tf.GradientTape() as tape:
            pooled = model({"input_ids": token_ids}, training=True)["pooled_output"]
            loss = keras.ops.mean(keras.ops.square(pooled))
        grads = tape.gradient(loss, grn)
        assert all(
            g is not None
            and float(np.abs(keras.ops.convert_to_numpy(g)).max()) > 0.0
            for g in grads
        )
