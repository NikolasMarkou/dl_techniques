"""Tests for the AsciiBert arm."""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.tokenizers.ascii_char import VOCAB_SIZE
from dl_techniques.models.embeddings_experimental.ascii_bert import (
    AsciiBert,
    create_ascii_bert,
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
        num_heads=4,
        intermediate_size=64,
        max_position_embeddings=SEQ_LEN * 2,
    )
    config.update(overrides)
    model = AsciiBert(**config)
    model.build((None, SEQ_LEN))
    return model


class TestConstruction:
    def test_defaults_to_the_ascii_vocabulary(self):
        assert tiny().vocab_size == VOCAB_SIZE

    def test_block_type_is_the_transformer(self):
        assert tiny().block_type == "transformer"

    def test_indivisible_hidden_size_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            AsciiBert(hidden_size=30, num_heads=4)

    def test_forward_pass_shapes(self, token_ids):
        out = tiny()({"input_ids": token_ids})
        assert tuple(out["last_hidden_state"].shape) == (2, SEQ_LEN, 32)
        assert tuple(out["pooled_output"].shape) == (2, 32)


class TestVariants:
    def test_every_variant_builds_and_runs(self, token_ids):
        for variant in AsciiBert.MODEL_VARIANTS:
            model = AsciiBert.from_variant(
                variant, max_position_embeddings=SEQ_LEN * 2
            )
            model.build((None, SEQ_LEN))
            out = model({"input_ids": token_ids})
            assert np.isfinite(
                keras.ops.convert_to_numpy(out["pooled_output"])
            ).all(), variant
            keras.backend.clear_session()

    def test_the_ladder_is_monotonically_larger(self, token_ids):
        sizes = []
        for variant in ("tiny", "small", "base"):
            model = AsciiBert.from_variant(
                variant, max_position_embeddings=SEQ_LEN * 2
            )
            model.build((None, SEQ_LEN))
            sizes.append(model.count_params())
            keras.backend.clear_session()
        assert sizes == sorted(sizes)
        assert len(set(sizes)) == 3

    def test_unknown_variant_raises_and_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="Unknown variant"):
            AsciiBert.from_variant("enormous")

    def test_description_is_not_forwarded_to_the_constructor(self):
        # MODEL_VARIANTS doubles as a kwargs dict, so `description` must be
        # popped; if it leaks through, construction raises.
        model = AsciiBert.from_variant("tiny", max_position_embeddings=32)
        assert not hasattr(model, "description")

    def test_overrides_beat_the_variant(self):
        model = AsciiBert.from_variant("tiny", num_layers=1)
        assert model.num_layers == 1


class TestFactory:
    def test_returns_the_requested_variant(self):
        model = create_ascii_bert("tiny", max_position_embeddings=32)
        assert isinstance(model, AsciiBert)
        assert model.hidden_size == AsciiBert.MODEL_VARIANTS["tiny"]["hidden_size"]

    def test_pretrained_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="No pretrained weights"):
            create_ascii_bert("tiny", pretrained=True)


class TestSerialization:
    def test_config_round_trip_preserves_the_arms_own_arguments(self):
        model = tiny(num_heads=2, intermediate_size=48, hidden_act="relu")
        config = model.get_config()
        # The generic block plumbing must not leak into the subclass config,
        # which does not accept it.
        assert "block_type" not in config
        assert "block_config" not in config

        restored = AsciiBert.from_config(config)
        assert restored.num_heads == 2
        assert restored.intermediate_size == 48
        assert restored.hidden_act == "relu"

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
        dead = [
            w.path
            for w, g in zip(model.trainable_weights, grads)
            if g is None
        ]
        assert dead == []
