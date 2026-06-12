"""Tests for the Ideogram4Transformer model + factory (step 6).

Uses the TINY preset throughout to stay fast. Builds a valid packed batch
(text tokens carrying ``llm_features`` + image tokens carrying noise ``x``),
verifies the forward shape / float32 velocity dtype, gradient finiteness,
the masked-add gating semantics, the ``.keras`` deterministic-velocity
round-trip, and ``get_config`` / ``from_config`` round-trip.
"""

import os
import keras
import numpy as np
import pytest

from dl_techniques.models.ideogram4.config import (
    Ideogram4Config,
    get_ideogram4_config,
)
from dl_techniques.models.ideogram4.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from dl_techniques.models.ideogram4.transformer import (
    Ideogram4Transformer,
    create_ideogram4_transformer,
)


# ---------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------

TEXT_LEN = 3
IMAGE_LEN = 4
SEQ_LEN = TEXT_LEN + IMAGE_LEN  # 7
BATCH = 2


@pytest.fixture(scope="module")
def tiny_config() -> Ideogram4Config:
    config, _ = get_ideogram4_config("tiny")
    return config


def _make_batch(config: Ideogram4Config, seed: int = 0) -> dict:
    """Build a valid packed TINY batch (text tokens then image tokens)."""
    rng = np.random.default_rng(seed)

    # indicator: first TEXT_LEN are text, remaining IMAGE_LEN are image.
    indicator = np.empty((BATCH, SEQ_LEN), dtype="int32")
    indicator[:, :TEXT_LEN] = LLM_TOKEN_INDICATOR
    indicator[:, TEXT_LEN:] = OUTPUT_IMAGE_INDICATOR

    # segment_ids: all tokens in one segment (full attention).
    segment_ids = np.zeros((BATCH, SEQ_LEN), dtype="int32")

    # position_ids (B, L, 3): simple monotone (t, h, w) coordinates.
    position_ids = np.zeros((BATCH, SEQ_LEN, 3), dtype="int32")
    for b in range(BATCH):
        for l in range(SEQ_LEN):
            position_ids[b, l, 0] = l  # t axis
            position_ids[b, l, 1] = l % 2  # h axis
            position_ids[b, l, 2] = l % 3  # w axis

    llm_features = rng.standard_normal(
        (BATCH, SEQ_LEN, config.llm_features_dim)
    ).astype("float32")
    x = rng.standard_normal((BATCH, SEQ_LEN, config.in_channels)).astype("float32")
    t = rng.uniform(0.0, 1.0, size=(BATCH,)).astype("float32")

    return {
        "llm_features": keras.ops.convert_to_tensor(llm_features),
        "x": keras.ops.convert_to_tensor(x),
        "t": keras.ops.convert_to_tensor(t),
        "position_ids": keras.ops.convert_to_tensor(position_ids),
        "segment_ids": keras.ops.convert_to_tensor(segment_ids),
        "indicator": keras.ops.convert_to_tensor(indicator),
    }


# ---------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------


class TestIdeogram4Transformer:
    def test_forward_shape_and_dtype(self, tiny_config):
        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config)
        out = model(batch)
        assert tuple(out.shape) == (BATCH, SEQ_LEN, tiny_config.in_channels)
        # Velocity output must be float32 (PyTorch returns .float()).
        assert keras.backend.standardize_dtype(out.dtype) == "float32"
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_t_per_token(self, tiny_config):
        """t given per-token (B, L) is accepted and yields the right shape."""
        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config)
        rng = np.random.default_rng(1)
        batch["t"] = keras.ops.convert_to_tensor(
            rng.uniform(0.0, 1.0, size=(BATCH, SEQ_LEN)).astype("float32")
        )
        out = model(batch)
        assert tuple(out.shape) == (BATCH, SEQ_LEN, tiny_config.in_channels)
        assert np.all(np.isfinite(keras.ops.convert_to_numpy(out)))

    def test_gradients_finite(self, tiny_config):
        import tensorflow as tf

        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config)
        # build weights
        _ = model(batch)
        with tf.GradientTape() as tape:
            out = model(batch, training=True)
            loss = tf.reduce_mean(tf.square(out))
        grads = tape.gradient(loss, model.trainable_variables)
        assert len(grads) > 0
        non_none = [g for g in grads if g is not None]
        assert len(non_none) == len(grads), "some grads are None"
        for g in non_none:
            assert np.all(np.isfinite(keras.ops.convert_to_numpy(g)))

    def test_llm_features_masked_at_image_positions(self, tiny_config):
        """Changing llm_features at NON-text (image) positions must not change
        the output -- llm_token_mask gates them out."""
        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config)
        out_ref = keras.ops.convert_to_numpy(model(batch))

        # Perturb llm_features ONLY at image positions (indices >= TEXT_LEN).
        llm = keras.ops.convert_to_numpy(batch["llm_features"]).copy()
        llm[:, TEXT_LEN:, :] += 100.0
        batch2 = dict(batch)
        batch2["llm_features"] = keras.ops.convert_to_tensor(llm)
        out_pert = keras.ops.convert_to_numpy(model(batch2))

        np.testing.assert_allclose(out_ref, out_pert, atol=1e-5)

    def test_x_masked_at_text_positions(self, tiny_config):
        """Changing x at text positions must not change the output --
        output_image_mask gates x to image tokens only."""
        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config)
        out_ref = keras.ops.convert_to_numpy(model(batch))

        x = keras.ops.convert_to_numpy(batch["x"]).copy()
        x[:, :TEXT_LEN, :] += 100.0
        batch2 = dict(batch)
        batch2["x"] = keras.ops.convert_to_tensor(x)
        out_pert = keras.ops.convert_to_numpy(model(batch2))

        np.testing.assert_allclose(out_ref, out_pert, atol=1e-5)

    def test_compute_output_shape(self, tiny_config):
        model = Ideogram4Transformer(config=tiny_config)
        in_shapes = {
            "llm_features": (BATCH, SEQ_LEN, tiny_config.llm_features_dim),
            "x": (BATCH, SEQ_LEN, tiny_config.in_channels),
            "t": (BATCH,),
            "position_ids": (BATCH, SEQ_LEN, 3),
            "segment_ids": (BATCH, SEQ_LEN),
            "indicator": (BATCH, SEQ_LEN),
        }
        out_shape = model.compute_output_shape(in_shapes)
        assert out_shape == (BATCH, SEQ_LEN, tiny_config.in_channels)

    def test_get_config_round_trip(self, tiny_config):
        model = Ideogram4Transformer(config=tiny_config)
        cfg = model.get_config()
        rebuilt = Ideogram4Transformer.from_config(cfg)
        assert rebuilt.config.to_dict() == tiny_config.to_dict()

    def test_keras_round_trip_deterministic_velocity(self, tiny_config, tmp_path):
        """The key serialization gate: save/reload yields IDENTICAL velocity at
        fixed dict inputs (the transformer is deterministic -- no sampling)."""
        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config, seed=7)
        out_before = keras.ops.convert_to_numpy(model(batch))

        path = os.path.join(str(tmp_path), "ideogram4_tiny.keras")
        model.save(path)
        reloaded = keras.models.load_model(path)
        out_after = keras.ops.convert_to_numpy(reloaded(batch))

        np.testing.assert_allclose(out_before, out_after, atol=1e-5)


class TestFactory:
    def test_factory_tiny_works(self, tiny_config):
        model = create_ideogram4_transformer("tiny")
        assert isinstance(model, Ideogram4Transformer)
        batch = _make_batch(tiny_config)
        out = model(batch)
        assert tuple(out.shape) == (BATCH, SEQ_LEN, tiny_config.in_channels)

    def test_factory_overrides(self):
        model = create_ideogram4_transformer("tiny", num_layers=1)
        assert model.config.num_layers == 1
        assert len(model.blocks) == 1

    def test_factory_full_constructs(self):
        """The full model must CONSTRUCT and report the expected config (34
        layers). Do NOT forward it -- construction only."""
        model = create_ideogram4_transformer("full")
        assert isinstance(model, Ideogram4Transformer)
        assert model.config.num_layers == 34
        assert model.config.emb_dim == 4608
        assert model.config.num_heads == 18
        assert len(model.blocks) == 34
